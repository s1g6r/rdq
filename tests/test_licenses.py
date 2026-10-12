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
    assert match_sources("RoboCOIN/AgiBot-g1_pick_up_battery", "") == []
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
