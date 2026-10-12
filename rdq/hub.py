"""Scan and audit LeRobot datasets on the Hugging Face Hub.

Two stages, both resumable:

1. scan:  list datasets tagged "LeRobot", download only README.md and
          meta/info.json for each, and record licence/provenance and metadata
          facts. Cheap: two small files per dataset.
2. audit: for a chosen subset, download meta/ and data/ (never videos), run
          the full checks, save the JSON report, then delete the data.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .checks import _declared_dim, names_status
from .provenance import analyse_card

TAG = "LeRobot"


def _hf():
    try:
        import huggingface_hub as hf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install huggingface_hub") from exc
    return hf


def _retry(fn, *args, attempts: int = 6, **kwargs):
    hf = _hf()
    delay = 5.0
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except hf.errors.EntryNotFoundError:
            return None
        except hf.errors.HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403, 404):
                return None
            if i == attempts - 1:
                raise
            time.sleep(delay if status != 429 else max(delay, 60))
            delay *= 2
        except (OSError, ValueError):
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return None


def list_datasets(limit: int, sort: str = "downloads") -> list[dict]:
    hf = _hf()
    api = hf.HfApi()
    out = []
    for info in api.list_datasets(
        filter=TAG,
        sort=sort,
        limit=limit,
        expand=["cardData", "downloads", "likes", "tags", "createdAt", "lastModified", "gated", "author"],
    ):
        card = info.card_data.to_dict() if getattr(info, "card_data", None) else {}
        out.append(
            {
                "id": info.id,
                "author": getattr(info, "author", None),
                "downloads": getattr(info, "downloads", None),
                "likes": getattr(info, "likes", None),
                "gated": getattr(info, "gated", None),
                "created_at": str(getattr(info, "created_at", "") or ""),
                "last_modified": str(getattr(info, "last_modified", "") or ""),
                "tags": list(getattr(info, "tags", None) or []),
                "card_license": card.get("license"),
            }
        )
    return out


def _read_hub_file(repo_id: str, filename: str, cache: Path) -> str | None:
    hf = _hf()
    path = _retry(hf.hf_hub_download, repo_id, filename, repo_type="dataset", cache_dir=str(cache))
    if not path:
        return None
    return Path(path).read_text(errors="replace")


def info_facts(info: dict) -> dict:
    """Metadata-level facts that need only meta/info.json."""
    features = info.get("features") or {}
    facts = {
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
        "cameras": [k for k, f in features.items() if (f or {}).get("dtype") == "video"],
        "has_action": "action" in features,
        "has_state": any(k.startswith("observation.state") for k in features),
        "names_shape_mismatch": [],
        "names_placeholder": [],
        "names_missing": [],
    }
    for key in ("action", "observation.state"):
        if key in features:
            facts[f"{key}_dim"] = _declared_dim(features[key])
            status = names_status(features[key])
            if status == "mismatch":
                facts["names_shape_mismatch"].append(key)
            elif status == "placeholder":
                facts["names_placeholder"].append(key)
            elif status == "unnamed":
                facts["names_missing"].append(key)
    return facts


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def scan(limit: int, out_dir: str | Path, sort: str = "downloads", pause: float = 0.2, log=print) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing_path = out_dir / "listing.json"
    if listing_path.exists():
        listing = json.loads(listing_path.read_text())
        log(f"using saved listing of {len(listing)} datasets")
    else:
        log(f"listing up to {limit} datasets tagged {TAG} (sorted by {sort})…")
        listing = list_datasets(limit, sort=sort)
        listing_path.write_text(json.dumps(listing, indent=1))
        log(f"found {len(listing)}")
    rows_path = out_dir / "scan.jsonl"
    done = _done_ids(rows_path)
    cache = out_dir / ".hf_cache"
    with open(rows_path, "a") as out:
        for n, entry in enumerate(listing, 1):
            if entry["id"] in done:
                continue
            row = dict(entry)
            try:
                readme = _read_hub_file(entry["id"], "README.md", cache)
                raw_info = _read_hub_file(entry["id"], "meta/info.json", cache)
                # Keep the raw files (a few KB each) so results can be recomputed without re-downloading.
                raw_dir = out_dir / "raw" / entry["id"].replace("/", "--")
                raw_dir.mkdir(parents=True, exist_ok=True)
                if readme is not None:
                    (raw_dir / "README.md").write_text(readme)
                if raw_info is not None:
                    (raw_dir / "info.json").write_text(raw_info)
                row["has_readme"] = readme is not None
                row["provenance"] = analyse_card(readme, card_license=entry.get("card_license")).to_dict()
                if raw_info:
                    try:
                        row["info"] = info_facts(json.loads(raw_info))
                    except json.JSONDecodeError:
                        row["info"] = None
                        row["error"] = "meta/info.json is not valid JSON"
                else:
                    row["info"] = None
            except Exception as exc:  # keep going; record what failed
                row["error"] = f"{type(exc).__name__}: {exc}"
            out.write(json.dumps(row) + "\n")
            out.flush()
            if n % 50 == 0:
                log(f"  {n}/{len(listing)} scanned")
            time.sleep(pause)
    shutil.rmtree(cache, ignore_errors=True)
    log(f"scan written to {rows_path}; raw cards and info.json in {out_dir / 'raw'}")
    return rows_path


def backfill_raw(out_dir: str | Path, log=print) -> int:
    """Download README.md and meta/info.json for scanned datasets whose raw files are missing."""
    out_dir = Path(out_dir)
    rows = [json.loads(l) for l in open(out_dir / "scan.jsonl")]
    cache = out_dir / ".hf_cache_backfill"
    missing = [r["id"] for r in rows if not (out_dir / "raw" / r["id"].replace("/", "--")).exists()]
    if missing:
        log(f"saving raw cards for {len(missing)} datasets scanned before raw saving existed…")
    for n, repo_id in enumerate(missing, 1):
        raw_dir = out_dir / "raw" / repo_id.replace("/", "--")
        try:
            readme = _read_hub_file(repo_id, "README.md", cache)
            info = _read_hub_file(repo_id, "meta/info.json", cache)
        except Exception as exc:  # keep going
            log(f"  {repo_id}: {type(exc).__name__}")
            continue
        raw_dir.mkdir(parents=True, exist_ok=True)
        if readme is not None:
            (raw_dir / "README.md").write_text(readme)
        if info is not None:
            (raw_dir / "info.json").write_text(info)
        if n % 50 == 0:
            log(f"  {n}/{len(missing)}")
    shutil.rmtree(cache, ignore_errors=True)
    return len(missing)


def _data_size_mb(repo_id: str) -> float | None:
    hf = _hf()
    info = _retry(hf.HfApi().dataset_info, repo_id, files_metadata=True)
    if info is None or not getattr(info, "siblings", None):
        return None
    total = sum((s.size or 0) for s in info.siblings if s.rfilename.startswith(("data/", "meta/")))
    return total / 1e6


NETWORK_HINTS = (
    "nodename nor servname",
    "name resolution",
    "error sending request",
    "connection reset",
    "connection refused",
    "timed out",
    "cas client error",
    "network is unreachable",
)


def _is_network_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"ConnectError", "ConnectTimeout", "ReadTimeout", "ConnectionError", "RemoteProtocolError", "ReadError"}:
        return True
    text = str(exc).lower()
    return any(h in text for h in NETWORK_HINTS)


def _wait_for_network(log=print, max_wait: float = 1800, host: str = "huggingface.co") -> bool:
    """Block until the Hub's hostname resolves again (or give up after max_wait seconds)."""
    import socket

    waited = 0.0
    announced = False
    while waited <= max_wait:
        try:
            socket.getaddrinfo(host, 443)
            if announced:
                log("  network is back, continuing")
            return True
        except OSError:
            if not announced:
                log("  network is down; waiting for it to come back (Ctrl-C to stop)…")
                announced = True
            time.sleep(30)
            waited += 30
    return False


def _finished_ids(path: Path) -> set[str]:
    """Datasets already audited or deliberately skipped. Errors are retried."""
    if not path.exists():
        return set()
    last: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                last[row["id"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return {i for i, r in last.items() if "score" in r or "skipped" in r}


def audit_repos(
    repo_ids: list[str],
    out_dir: str | Path,
    max_mb: float = 300,
    keep_data: bool = False,
    log=print,
) -> Path:
    from .report import audit as run_audit
    from .report import write_json

    hf = _hf()
    out_dir = Path(out_dir)
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    work = out_dir / ".downloads"
    index_path = out_dir / "audit_index.jsonl"
    done = _finished_ids(index_path)
    with open(index_path, "a") as index:
        for n, repo_id in enumerate(repo_ids, 1):
            if repo_id in done:
                continue
            row = _audit_one(repo_id, work, reports, out_dir, max_mb, keep_data, run_audit, write_json, hf, log)
            index.write(json.dumps(row) + "\n")
            index.flush()
            status = row.get("score", row.get("skipped", row.get("error", "?")))
            log(f"  [{n}/{len(repo_ids)}] {repo_id}: {status}")
    return index_path


def _audit_one(repo_id, work, reports, out_dir, max_mb, keep_data, run_audit, write_json, hf, log) -> dict:
    """Audit one dataset, waiting out network drops instead of recording them as failures."""
    for attempt in range(4):
        row = {"id": repo_id}
        local = work / repo_id.replace("/", "__")
        try:
            size = _data_size_mb(repo_id)
            row["data_mb"] = round(size, 1) if size is not None else None
            if size is not None and size > max_mb:
                row["skipped"] = f"data+meta is {size:.0f} MB (limit {max_mb:.0f} MB)"
            else:
                _retry(
                    hf.snapshot_download,
                    repo_id,
                    repo_type="dataset",
                    local_dir=str(local),
                    allow_patterns=["meta/*", "meta/**", "data/**", "README.md"],
                )
                result = run_audit(local)
                result["dataset"]["path"] = repo_id
                report_path = reports / f"{repo_id.replace('/', '__')}.json"
                write_json(result, report_path)
                row.update(
                    {
                        "score": result["score"]["dataset_score"],
                        "usable_fraction": result["score"]["usable_fraction"],
                        "episodes": result["dataset"]["episodes"],
                        "frames": result["dataset"]["frames"],
                        "severity_counts": result["score"]["severity_counts"],
                        "checks": sorted({f["check"] for f in result["findings"]}),
                        "report": str(report_path.relative_to(out_dir)),
                    }
                )
            return row
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if not _is_network_error(exc) or attempt == 3:
                return row
            if not _wait_for_network(log):
                return row
        finally:
            if not keep_data:
                shutil.rmtree(local, ignore_errors=True)
    return row
