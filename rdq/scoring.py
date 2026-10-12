"""Turn findings into per-episode and per-dataset scores.

The scores are deliberately simple and transparent: every point lost can be
traced to a named finding. Whether they predict training outcomes is an
empirical question the experiment in experiments/ is designed to answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .checks import CRITICAL, INFO, WARNING, Finding

EPISODE_PENALTY = {CRITICAL: 60, WARNING: 20, INFO: 5}
DATASET_PENALTY = {CRITICAL: 15, WARNING: 5, INFO: 1}


@dataclass
class Score:
    dataset_score: int
    episode_scores: dict[int, int]
    usable_episodes: list[int]  # no critical or warning finding
    flagged_episodes: dict[int, list[str]]  # episode -> checks that flagged it
    severity_counts: dict[str, int]

    @property
    def usable_fraction(self) -> float:
        total = len(self.episode_scores)
        return len(self.usable_episodes) / total if total else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["episode_scores"] = {str(k): v for k, v in self.episode_scores.items()}
        d["flagged_episodes"] = {str(k): v for k, v in self.flagged_episodes.items()}
        d["usable_fraction"] = round(self.usable_fraction, 4)
        return d


def score(findings: list[Finding], episode_ids: list[int]) -> Score:
    ep_scores = {i: 100 for i in episode_ids}
    flagged: dict[int, list[str]] = {}
    blocking: set[int] = set()
    dataset_penalty = 0
    counts = {CRITICAL: 0, WARNING: 0, INFO: 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
        if f.episodes:
            for ep in f.episodes:
                if ep not in ep_scores:
                    continue
                ep_scores[ep] = max(0, ep_scores[ep] - EPISODE_PENALTY[f.severity])
                flagged.setdefault(ep, []).append(f.check)
                if f.severity in (CRITICAL, WARNING):
                    blocking.add(ep)
        else:
            dataset_penalty += DATASET_PENALTY[f.severity]
    mean_ep = sum(ep_scores.values()) / len(ep_scores) if ep_scores else 0
    dataset_score = int(round(max(0.0, mean_ep - dataset_penalty)))
    usable = [i for i in episode_ids if i not in blocking]
    return Score(
        dataset_score=dataset_score,
        episode_scores=ep_scores,
        usable_episodes=usable,
        flagged_episodes=flagged,
        severity_counts=counts,
    )
