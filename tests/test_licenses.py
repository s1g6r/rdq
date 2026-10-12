"""Upstream licence matching and declared-vs-upstream assessment."""

from rdq.licenses import assess_card, match_sources
from rdq.provenance import analyse_card
from tests.test_checks import TEMPLATE_CARD


def test_mirror_of_cc_by_dataset_is_relabelled():
    a = assess_card("someone/droid_1.0.1", TEMPLATE_CARD, "apache-2.0")
    assert a.status == "relabelled" and a.severity == "warning"
    assert a.concluded == "CC-BY-4.0"
    assert "attribution" in a.restrictions and "indicate-modifications" in a.restrictions
    assert a.upstream[0]["evidence"] == "repo-id"


def test_matching_license_is_consistent():
    a = assess_card("lab/bridge_v2_lerobot", "See https://rail-berkeley.github.io/bridgedata/", "cc-by-4.0")
    assert a.status == "consistent" and a.severity == "ok"
    assert a.upstream[0]["evidence"] == "card-link"


def test_noncommercial_upstream_relabelled_permissive_is_critical():
    a = assess_card("someone/agibot_alpha_v30", TEMPLATE_CARD, "apache-2.0")
    assert a.status == "upstream-conflict" and a.severity == "critical"
    assert a.commercial_training is False
    p = analyse_card("---\nlicense: apache-2.0\n---\n" + TEMPLATE_CARD, repo_id="someone/agibot_alpha_v30")
    assert p.verdict == "licence conflicts with upstream"


def test_robot_name_is_not_a_dataset_source():
    # Data recorded on an AgiBot robot is not AgiBot World.
    keys = {m.key for m in match_sources("RoboCOIN/AgiBot-g1_pick_up_battery", "")}
    assert "agibot_world" not in keys and keys <= {"robocoin"}
    assert match_sources("user/android_app_demos", "") == []


def test_no_redistribution_source():
    a = assess_card("user/ego4d_lerobot", "", "mit")
    assert a.status == "upstream-conflict" and a.severity == "critical"
    assert a.redistribution is False


def test_unknown_upstream_and_missing():
    assert assess_card("user/my_so101_pick", "", "apache-2.0").status == "no-known-upstream"
    assert assess_card("user/my_so101_pick", "", None).status == "missing"
    assert assess_card("user/droid_subset", "", None).status == "missing"


def test_card_mention_alone_is_only_a_warning():
    a = assess_card("user/hand_poses", "Poses estimated from EPIC-KITCHENS clips.", "mit")
    assert a.status == "upstream-conflict" and a.severity == "warning"
    assert a.upstream[0]["evidence"] == "card-mention"


def test_correctly_tagged_no_derivatives_source():
    a = assess_card("user/egodex-test", "", "cc-by-nc-nd-4.0")
    assert a.status == "restricted-redistribution" and a.severity == "warning"


# --------------------------------------------------------------------------- #
# Data passport
# --------------------------------------------------------------------------- #

import json  # noqa: E402

import pytest  # noqa: E402

from rdq import synthetic as syn  # noqa: E402
from rdq.passport import build_passport, content_hash, load_answers, to_datasheet, to_spdx  # noqa: E402
from rdq.report import audit  # noqa: E402


def _ds(tmp_path, license_tag="apache-2.0", card=TEMPLATE_CARD):
    root = syn.write_dataset(tmp_path / "ds", syn.make_episodes(10))
    readme = f"---\nlicense: {license_tag}\n---\n{card}"
    (root / "README.md").write_text(readme)
    return root, readme


def test_passport_for_relabelled_mirror(tmp_path):
    root, readme = _ds(tmp_path)
    result = audit(root, readme=readme, repo_id="someone/droid_subset")
    p = build_passport(result, readme, root, repo_id="someone/droid_subset")
    assert p["rdq:license"]["status"] == "relabelled"
    assert p["rdq:license"]["concluded"] == "CC-BY-4.0"
    assert p["license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert p["prov:wasDerivedFrom"][0]["name"] == "DROID"
    assert "collection_method" in p["rdq:unanswered"]
    assert p["rdq:structure"]["episodes"] == 10
    json.dumps(p)  # serialisable


def test_passport_default_licence_without_upstream_is_noassertion(tmp_path):
    root, readme = _ds(tmp_path)
    p = build_passport(audit(root, readme=readme, repo_id="me/so101_pick"), readme, root, repo_id="me/so101_pick")
    assert p["rdq:license"]["status"] == "default-unconfirmed"
    assert p["license"] == "NOASSERTION"
    answers = {"license_confirmed_by": "Jane Doe, data owner", "collection_method": "leader-arm teleop"}
    p2 = build_passport(audit(root, readme=readme, repo_id="me/so101_pick"), readme, root, repo_id="me/so101_pick", answers=answers)
    assert p2["rdq:license"]["status"] == "owner-confirmed"
    assert p2["license"] == "https://spdx.org/licenses/Apache-2.0.html"
    assert p2["rai:dataCollection"] == "leader-arm teleop"


def test_noncommercial_upstream_adds_usage_prohibition(tmp_path):
    root, readme = _ds(tmp_path)
    p = build_passport(audit(root, readme=readme, repo_id="x/agibot_alpha_v30"), readme, root, repo_id="x/agibot_alpha_v30")
    assert p["rdq:license"]["commercialTraining"] is False
    assert {"odrl:action": "odrl:commercialize"} in p["usageInfo"]["odrl:prohibition"]


def test_spdx_and_datasheet(tmp_path):
    root, readme = _ds(tmp_path)
    p = build_passport(audit(root, readme=readme, repo_id="someone/droid_subset"), readme, root, repo_id="someone/droid_subset")
    spdx = to_spdx(p)
    rels = {n["relationshipType"]: n for n in spdx["@graph"] if n.get("type") == "Relationship"}
    assert {"hasDeclaredLicense", "hasConcludedLicense", "descendantOf"} <= set(rels)
    exprs = {n["simplelicensing_licenseExpression"] for n in spdx["@graph"] if n.get("type") == "simplelicensing_LicenseExpression"}
    assert exprs == {"Apache-2.0", "CC-BY-4.0"}
    md = to_datasheet(p)
    assert "## Licence" in md and "CC-BY-4.0" in md and "_not provided_" in md


def test_content_hash_changes_with_data(tmp_path):
    root, _ = _ds(tmp_path)
    h1 = content_hash(root)["value"]
    (root / "meta" / "info.json").write_text((root / "meta" / "info.json").read_text() + " ")
    assert content_hash(root)["value"] != h1


def test_answers_reject_unknown_keys(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("collection_method: teleop\nfavourite_colour: blue\n")
    with pytest.raises(ValueError):
        load_answers(f)


def test_passport_validates_with_mlcroissant(tmp_path):
    mlc = pytest.importorskip("mlcroissant")
    root, readme = _ds(tmp_path)
    p = build_passport(audit(root, readme=readme, repo_id="someone/droid_subset"), readme, root, repo_id="someone/droid_subset")
    mlc.Dataset(jsonld=p)  # raises on validation errors


def test_upstream_without_a_data_licence():
    a = assess_card("lerobot/aloha_sim_insertion_human", "", "mit")
    assert a.status == "no-upstream-grant" and a.severity == "warning"
    assert a.concluded is None


def test_umbrella_licence_caps_severity():
    a = assess_card("IPEC-COMMUNITY/language_table_lerobot", "", "apache-2.0")
    assert a.status == "relabelled" and a.severity == "warning"
    assert any("umbrella" in n for n in a.notes)


def test_knowledge_base_patterns_do_not_overlap_on_known_ids():
    from rdq.licenses import UPSTREAM_SOURCES
    import re

    ids = ["IPEC-COMMUNITY/bc_z_lerobot", "lerobot/berkeley_cable_routing", "ember-lab-berkeley/robocasa365-pretrain-mg",
           "cadene/droid_1.0.1", "IPEC-COMMUNITY/bridge_orig_lerobot", "lerobot/pusht", "lerobot/libero"]
    for rid in ids:
        hits = [s.key for s in UPSTREAM_SOURCES if re.search(s.id_pattern, rid.lower())]
        assert len(hits) == 1, (rid, hits)


def test_passport_for_mirror_of_unlicensed_original(tmp_path):
    root, readme = _ds(tmp_path, license_tag="mit")
    p = build_passport(audit(root, readme=readme, repo_id="lerobot/aloha_sim_insertion_human"), readme, root,
                       repo_id="lerobot/aloha_sim_insertion_human")
    assert p["rdq:license"]["status"] == "no-upstream-grant"
    assert p["license"] == "NOASSERTION"
