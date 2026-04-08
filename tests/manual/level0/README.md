# Manual Level 0 Fixtures

This directory contains sample files for manual table-detection checks.

## Current Purpose

Use these files to compare:

- `gemma_first` output
- `heuristic_only` output

Example:

```bash
gridgulp detect tests/manual/level0/sample.xlsx --json
GRIDGULP_DETECTION_STRATEGY=heuristic_only \
gridgulp detect tests/manual/level0/sample.xlsx --json
```

These fixtures are no longer tied to a vision pipeline. They are now general regression inputs for complex layout and semantic-boundary checks.
