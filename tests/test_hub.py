"""Offline tests for the hub scanner and auditor (network calls are replaced)."""

import json
import shutil

from rdq import hub
from rdq import synthetic as syn
from tests.test_checks import TEMPLATE_CARD


def test_scan_records_provenance_and_facts(tmp_path, monkeypatch):
    listing = [
        {"id": "alice/pick", "card_license": "apache-2.0", "gated": False},
        {"id": "bob/broken", "card_license": None, "gated": False},
    ]
    monkeypatch.setattr(hub, "list_datasets", lambda limit, sort="downloads": listing)
    root = syn.write_dataset(tmp_path / "src", syn.make_episodes(5), names=syn.MOTORS[:5])
    info_text = (root / "meta" / "info.json").read_text()

    def fake_read(repo_id, filename, cache):
        if repo_id == "bob/broken":
            return None
        return TEMPLATE_CARD if filename == "README.md" else info_text

    monkeypatch.setattr(hub, "_read_hub_file", fake_read)
    path = hub.scan(10, tmp_path / "out", pause=0, log=lambda *a: None)
    rows = [json.loads(l) for l in open(path)]
    assert len(rows) == 2
    alice = rows[0]
    assert alice["provenance"]["verdict"] == "default licence, unconfirmed"
    assert alice["info"]["codebase_version"] == "v3.0"
    assert alice["info"]["action_dim"] == 6
    assert alice["info"]["names_shape_mismatch"] == ["action", "observation.state"]
    assert rows[1]["provenance"]["verdict"] == "no licence"
    assert rows[1]["info"] is None
    # Resuming does not duplicate rows.
    hub.scan(10, tmp_path / "out", pause=0, log=lambda *a: None)
    assert len(open(path).readlines()) == 2


def test_audit_repos_downloads_audits_and_cleans_up(tmp_path, monkeypatch):
    src = syn.write_dataset(tmp_path / "src", syn.make_episodes(12))
    (src / "README.md").write_text(TEMPLATE_CARD)

    class FakeHF:
        class errors:
            class EntryNotFoundError(Exception):
                pass

            class HfHubHTTPError(Exception):
                pass

        @staticmethod
        def snapshot_download(repo_id, repo_type, local_dir, allow_patterns):
            shutil.copytree(src, local_dir)
            return local_dir

    monkeypatch.setattr(hub, "_hf", lambda: FakeHF)
    monkeypatch.setattr(hub, "_data_size_mb", lambda repo_id: 1.0 if repo_id != "big/one" else 9999.0)
    index = hub.audit_repos(["alice/pick", "big/one"], tmp_path / "out", max_mb=100, log=lambda *a: None)
    rows = [json.loads(l) for l in open(index)]
    assert rows[0]["episodes"] == 12 and rows[0]["score"] >= 90
    assert "skipped" in rows[1]
    report = json.loads((tmp_path / "out" / rows[0]["report"]).read_text())
    assert report["dataset"]["path"] == "alice/pick"
    assert report["provenance"]["default_license_unconfirmed"] is True
    assert not (tmp_path / "out" / ".downloads" / "alice__pick").exists()


def test_backfill_raw(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "raw" / "a--x").mkdir(parents=True)
    (out / "scan.jsonl").write_text('{"id": "a/x"}\n{"id": "b/y"}\n')
    calls = []

    def fake_read(repo_id, filename, cache):
        calls.append((repo_id, filename))
        return "{}" if filename.endswith(".json") else "---\nlicense: mit\n---\n"

    monkeypatch.setattr(hub, "_read_hub_file", fake_read)
    assert hub.backfill_raw(out, log=lambda *a: None) == 1
    assert {c[0] for c in calls} == {"b/y"}
    assert (out / "raw" / "b--y" / "info.json").exists()


def test_network_errors_are_retried_not_recorded(tmp_path, monkeypatch):
    src = syn.write_dataset(tmp_path / "src", syn.make_episodes(12))
    calls = {"n": 0}

    class ConnectError(Exception):
        pass

    class FakeHF:
        class errors:
            class EntryNotFoundError(Exception):
                pass

            class HfHubHTTPError(Exception):
                pass

        @staticmethod
        def snapshot_download(repo_id, repo_type, local_dir, allow_patterns):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectError("[Errno 8] nodename nor servname provided, or not known")
            shutil.copytree(src, local_dir)
            return local_dir

    monkeypatch.setattr(hub, "_hf", lambda: FakeHF)
    monkeypatch.setattr(hub, "_data_size_mb", lambda repo_id: 1.0)
    monkeypatch.setattr(hub, "_wait_for_network", lambda log=None, **k: True)
    out = tmp_path / "out"
    # A previous run recorded an error for this dataset; it must be retried.
    out.mkdir()
    (out / "audit_index.jsonl").write_text('{"id": "alice/pick", "error": "ConnectError: down"}\n')
    index = hub.audit_repos(["alice/pick"], out, max_mb=100, log=lambda *a: None)
    rows = [json.loads(l) for l in open(index)]
    assert rows[-1]["id"] == "alice/pick" and "score" in rows[-1]
    assert calls["n"] == 2


def test_check_accepts_a_hub_id(tmp_path, monkeypatch, capsys):
    from rdq import cli

    src = syn.write_dataset(tmp_path / "src", syn.make_episodes(10))
    (src / "README.md").write_text("---\nlicense: apache-2.0\n---\n" + TEMPLATE_CARD)

    class File:
        def __init__(self, name, size):
            self.rfilename, self.size = name, size

    class FakeHF:
        @staticmethod
        def snapshot_download(repo_id, repo_type, allow_patterns):
            assert "videos/**" not in allow_patterns
            return str(src)

        class HfApi:
            def dataset_info(self, repo_id, files_metadata):
                return type("Info", (), {"siblings": [File("data/x.parquet", 5_000_000), File("videos/a.mp4", 9e9)]})()

    monkeypatch.setattr(hub, "_hf", lambda: FakeHF)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "r.json"
    assert cli.main(["check", "someone/droid_subset", "--json", str(out), "--no-color"]) in (0, 1)
    # videos are not counted without --video; a small --max-mb refuses the download
    assert cli.main(["check", "someone/droid_subset", "--max-mb", "1"]) == 2
    rep = json.loads(out.read_text())
    assert rep["dataset"]["path"] == "someone/droid_subset"
    assert rep["provenance"]["upstream"]["status"] == "relabelled"


def test_summary_keeps_names_and_licence_stats_separate():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from analyze_hub import summarise_scan

    rows = []
    for i in range(10):
        rows.append({
            "id": f"o{i}/droid_{i}" if i < 6 else f"o{i}/mine_{i}",
            "downloads": 10,
            "provenance": hub.analyse_card("---\nlicense: apache-2.0\n---\n" + TEMPLATE_CARD,
                                           repo_id=f"o{i}/droid_{i}" if i < 6 else f"o{i}/mine_{i}").to_dict(),
            "info": {"codebase_version": "v3.0", "robot_type": "so101", "has_action": True,
                     "names_shape_mismatch": ["action"] if i == 0 else [], "names_placeholder": [], "names_missing": []},
        })
    s = summarise_scan(rows)
    assert s["names_true_mismatch"].startswith("1/10")
    assert s["upstream_licences"]["declared_differs_from_upstream"].startswith("6/6")
