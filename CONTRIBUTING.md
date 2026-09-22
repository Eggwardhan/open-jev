# Contributing to open-jev

Thanks for helping improve open-jev.

## Before opening a pull request

1. Explain the user problem and the decision contract being changed.
2. Add focused tests for new behavior or bug fixes.
3. Run `ruff check src tests scripts` and `pytest -q`.
4. Include a small reproducible example when adding a primitive, benchmark, or integration.
5. Keep external source attribution and license boundaries explicit.

Prefer small pull requests that change one decision primitive or evaluation path
at a time. Claims about accuracy, calibration, latency, or cost should include
the dataset split, baseline, sample count, and hardware used.
