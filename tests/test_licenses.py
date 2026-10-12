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
    a = assess_card("lab/bridge_v2_lerobot", "Converted from https://rail-berkeley.github.io/bridgedata/", "cc-by-4.0")
    assert a.status == "consistent" and a.severity == "ok"
    assert a.upstream[0]["evidence"] == "card-link"


def test_noncommercial_upstream_relabelled_permissive_is_critical():
    # A repo name alone is a lead: the conflict is reported, but not as critical.
    a = assess_card("someone/agibot_alpha_v30", TEMPLATE_CARD, "apache-2.0")
    assert a.status == "upstream-conflict" and a.severity == "warning"
    assert a.commercial_training is False
    # The card declaring the source makes it critical.
    card = "---\nlicense: apache-2.0\nsource_datasets:\n- agibot-world/AgiBotWorld-Alpha\n---\n" + TEMPLATE_CARD
    a = assess_card("someone/agibot_alpha_v30", card, "apache-2.0")
    assert a.severity == "critical"
    p = analyse_card(card, repo_id="someone/agibot_alpha_v30")
    assert p.verdict == "licence conflicts with upstream"


def test_robot_name_is_not_a_dataset_source():
    # Data recorded on an AgiBot robot is not AgiBot World.
    keys = {m.key for m in match_sources("RoboCOIN/AgiBot-g1_pick_up_battery", "")}
    assert "agibot_world" not in keys and keys <= {"robocoin"}
    assert match_sources("user/android_app_demos", "") == []


def test_no_redistribution_source():
    a = assess_card("user/ego4d_lerobot", "", "mit")
    assert a.status == "upstream-conflict" and a.severity == "warning"  # repo name only
    assert a.redistribution is False
    a = assess_card("user/ego4d_lerobot", "Converted from Ego4D (https://ego4d-data.org).", "mit")
    assert a.severity == "critical"


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


# --------------------------------------------------------------------------- #
# Regressions from the pre-release review (R7)


def _m(key, evidence):
    from rdq.licenses import _BY_KEY, SourceMatch

    return SourceMatch(key, _BY_KEY[key].name, evidence, corroborated=True)


def test_noncommercial_source_wins_over_unknown_terms():
    from rdq.licenses import assess

    a = assess("apache-2.0", [_m("epic_kitchens", "card-link"), _m("berkeley_fanuc_manipulation", "card-link")])
    assert a.severity == "critical" and a.commercial_training is False
    a = assess("apache-2.0", [_m("epic_kitchens", "card-link"), _m("rh20t", "card-link")])
    assert a.severity == "critical" and a.commercial_training is False


def test_weak_second_source_does_not_hide_a_strong_conflict():
    from rdq.licenses import assess

    a = assess("apache-2.0", [_m("epic_kitchens", "card-link"), _m("oxe_language_table", "repo-id")])
    assert a.status == "upstream-conflict" and a.severity == "critical"
    joined = " ".join(a.notes)
    assert "links to EPIC-KITCHENS" in joined and "repo name suggests it is derived from Language-Table" in joined


def test_unrelated_repo_ids_are_not_upstream_mirrors():
    assert "bridge_v2" not in {m.key for m in match_sources("cambridge-robotics/cambridge_v2_pick", "")}
    assert "unitree_g1_datasets" not in {m.key for m in match_sources("unitreerobotics/Z1_Dual_Dex1_StackBox", "")}
    # A generic id match is a lead, not a critical finding.
    for rid in ("acme/epic_kitchen_assistant", "acme/hoi4d_retarget", "acme/egodex_style_handover", "acme/kai0_test"):
        a = assess_card(rid, "", "apache-2.0")
        assert a.severity != "critical", rid
    # Nor is a name that merely mentions LeRobot; a derivation link plus the name is.
    assert assess_card("someone/epic_kitchens_lerobot", "", "apache-2.0").severity == "warning"
    card = "LeRobot conversion of EPIC-KITCHENS (https://epic-kitchens.github.io)."
    assert assess_card("someone/epic_kitchens_lerobot", card, "apache-2.0").severity == "critical"


def test_citation_link_is_not_derivation_evidence():
    a = assess_card("acme/inhouse_pick", "Our episode format follows AgiBot World (https://agibot-world.com).", "apache-2.0")
    assert a.severity != "critical"
    assert not a.upstream or a.upstream[0]["evidence"] == "card-mention"
    a = assess_card("acme/inhouse_pick", "Converted from AgiBot World: https://agibot-world.com", "apache-2.0")
    assert a.severity == "warning" and a.upstream[0]["evidence"] == "card-link"  # one signal
    a = assess_card("acme/agibot_world_subset", "Converted from AgiBot World: https://agibot-world.com", "apache-2.0")
    assert a.severity == "critical"
    # "open source" and LeRobot's template Homepage label next to a cited link are not derivation evidence
    a = assess_card("acme/kitchen", "Our open source teleop data, in the style of EPIC-KITCHENS (https://epic-kitchens.github.io).", "apache-2.0")
    assert a.severity != "critical"


def test_model_pretraining_mention_is_not_an_upstream():
    a = assess_card("acme/pick", "Policies use an R3M encoder pretrained on Ego4D.", "apache-2.0")
    assert a.status == "no-known-upstream"
    a = assess_card("acme/hands", "Our own hand poses, compared against EgoDex.", "mit")
    assert a.status == "no-known-upstream"


def test_source_datasets_is_the_strongest_evidence():
    card = "---\nlicense: apache-2.0\nsource_datasets:\n- agibot-world/AgiBotWorld-Alpha\n---\nhello"
    a = assess_card("acme/pick", card, "apache-2.0")
    assert a.upstream[0]["evidence"] == "card-source" and a.severity == "critical"


def test_publisher_upload_is_informational():
    a = assess_card("RoboCOIN/some_task", "", "apache-2.0")
    assert a.severity in ("ok", "info") and a.status == "publisher-upload"
    assert a.upstream and a.upstream[0]["evidence"] == "publisher"
    a = assess_card("RoboCOIN/some_task", "", "mit")
    assert a.status == "publisher-upload" and a.severity == "info" and a.concluded is None


def test_stricter_declared_licence_is_not_a_conflict():
    a = assess_card("acme/droid_nc_subset", "", "cc-by-nc-4.0")
    assert a.status == "stricter-than-upstream" and a.severity == "info"


def test_passport_keeps_a_publisher_added_restriction(tmp_path):
    from rdq import synthetic as syn
    from rdq.passport import build_passport
    from rdq.report import audit

    root = tmp_path / "ds"
    syn.write_dataset(root, syn.make_episodes(4, 60), with_video=False)
    card = "---\nlicense: cc-by-nc-4.0\n---\nA subset of DROID."
    doc = build_passport(audit(root, readme=card, repo_id="acme/droid_nc_subset"), card, root, "acme/droid_nc_subset")
    lic = doc["rdq:license"]
    assert lic["commercialTraining"] is False
    assert "CC-BY-NC-4.0" in lic["concluded"]
    assert doc["usageInfo"]["odrl:prohibition"]
    assert not str(doc["url"]).startswith("file:")


def test_share_alike_added_to_cc_by_is_stricter():
    assert assess_card("acme/droid_sa_subset", "", "cc-by-sa-4.0").status == "stricter-than-upstream"


def test_list_valued_licence_uses_the_most_restrictive_tag():
    p = analyse_card("---\nlicense:\n- mit\n- cc-by-nc-4.0\n---\nhello")
    assert p.license == "cc-by-nc-4.0" and p.verdict == "non-commercial"


def test_bottom_line_says_what_to_do():
    p = analyse_card("---\nlicense: apache-2.0\n---\nhi", repo_id="someone/droid_1.0.1")
    assert p.bottom_level == "go" and "credit DROID" in p.bottom_line
    p = analyse_card("---\nlicense: cc-by-nc-4.0\n---\nhi")
    assert p.bottom_level == "stop"
    p = analyse_card("no front matter")
    assert p.bottom_level == "stop" and "permission" in p.bottom_line


def test_a_source_the_card_says_it_did_not_use_is_not_matched():
    card = ("---\nlicense: apache-2.0\n---\nAn independent SO-ARM dataset inspired by their collection strategy; "
            "it is not the WidowX data from the paper, and we do not use BridgeData V2 here.")
    assert not match_sources("someone/cube-cup-100", card)


def test_a_mention_alone_does_not_say_go():
    card = "---\nlicense: apache-2.0\n---\nTeleoperated pick-and-place. See also BridgeData V2."
    p = analyse_card(card, repo_id="someone/pick_place")
    assert p.upstream["status"] == "relabelled" and not p.upstream["concluded"]
    assert p.bottom_level == "check" and "None" not in p.bottom_line and "CC-BY-4.0" in p.bottom_line


def test_odd_source_datasets_values_do_not_crash():
    for value in ("7", "true", "{a: 1}", "[]", "null"):
        analyse_card(f"---\nlicense: mit\nsource_datasets: {value}\n---\nbody", repo_id="x/y").to_dict()
