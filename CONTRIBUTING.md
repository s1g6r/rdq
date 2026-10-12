# Contributing

Bug reports with a dataset id (or a small sample) are the most useful contribution.

- **Setup:** `pip install -e ".[dev,croissant,video]"`, then run `pytest -q` and `python tests/false_alarms.py`.
- **A new check** goes in `rdq/checks.py`. Plant the fault in a synthetic dataset (`rdq/synthetic.py`) and test that the right episodes are flagged. The false-alarm script must stay at zero warnings on clean data.
- **A new upstream licence entry** goes in `rdq/data/upstream_licences.json`. It needs a primary-source statement URL, a ≤25-word quote, the check date and a confidence level. Make sure `id_pattern` does not match unrelated repos (a robot brand is not a dataset).
- **Thresholds** live in `Thresholds` in `rdq/checks.py` and are written into every JSON report. Change them only with evidence.
