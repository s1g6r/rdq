"""MCAP recordings (ROS 2 JointState and JSON) audited with the signal-level checks."""

import json

import numpy as np
import pytest

from rdq import synthetic as syn
from rdq.cli import main

pytest.importorskip("mcap")
pytest.importorskip("mcap_ros2")

JOINT_STATE = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


def _write_ros2(path, state, action, fps=30):
    from mcap_ros2.writer import Writer

    with open(path, "wb") as fh:
        w = Writer(fh)
        schema = w.register_msgdef("sensor_msgs/msg/JointState", JOINT_STATE)
        for t in range(len(state)):
            ns = int(1e9 * t / fps)
            for topic, vec in (("/follower/joint_states", state[t]), ("/leader/joint_states", action[t])):
                msg = {"header": {"stamp": {"sec": ns // 10**9, "nanosec": ns % 10**9}, "frame_id": ""},
                       "name": [f"j{i}" for i in range(len(vec))], "position": [float(x) for x in vec], "velocity": [], "effort": []}
                w.write_message(topic=topic, schema=schema, message=msg, log_time=ns, publish_time=ns)
        w.finish()


def test_mcap_ros2_audit_finds_planted_faults(tmp_path):
    eps = syn.make_episodes(12, length=150, seed=3)
    syn.make_idle(eps[4])
    syn.delay_action(eps[7], frames=10)
    files = []
    for i, e in enumerate(eps):
        p = tmp_path / f"episode_{i:03d}.mcap"
        _write_ros2(p, e["state"], e["action"])
        files.append(str(p))
    out = tmp_path / "r.json"
    assert main(["check-mcap", *files, "--state-topic", "/follower/joint_states", "--action-topic",
                 "/leader/joint_states", "--json", str(out), "--no-color"]) == 0
    rep = json.loads(out.read_text())
    assert rep["dataset"]["episodes"] == 12 and abs(rep["dataset"]["fps"] - 30) < 0.5
    assert set(rep["score"]["usable_episodes"]) == set(range(12)) - {4, 7}
    checks = {f["check"] for f in rep["findings"]}
    assert "episodes.idle" in checks and "sync.action_state_lag" in checks
    assert not any(c.startswith(("stats.", "counts.", "info.")) for c in checks)  # LeRobot-format checks skipped


def test_mcap_json_messages(tmp_path):
    from mcap.writer import Writer

    p = tmp_path / "ep.mcap"
    with open(p, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name="Joints", encoding="jsonschema", data=b"{}")
        cid = w.register_channel(topic="/state", message_encoding="json", schema_id=sid)
        for t in range(60):
            ns = int(t * 1e9 / 20)
            w.add_message(channel_id=cid, log_time=ns, publish_time=ns, data=json.dumps({"data": [np.sin(t / 10), t / 60]}).encode())
        w.finish()
    from rdq.mcap_reader import load_mcap

    ds = load_mcap([p], "/state")
    assert ds.episodes[0].length == 60 and ds.fps == 20.0
    assert ds.numeric_keys == ["observation.state"]
