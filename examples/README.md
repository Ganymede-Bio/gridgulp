# GridGulp Examples

These examples exercise the current Gemma-first and heuristic-only workflows.

## Recommended Flow

1. Configure Gemma in `gridgulp.toml`
2. Run `gridgulp doctor`
3. Run `gridgulp schematize ...`
4. Compare against `heuristic_only` when debugging proposal quality

## Example Commands

Basic CSV run:

```bash
gridgulp schematize examples/spreadsheets/simple/basic_table.csv --json
```

Heuristic-only comparison:

```bash
GRIDGULP_DETECTION_STRATEGY=heuristic_only \
gridgulp schematize examples/spreadsheets/simple/basic_table.csv --json
```

## Python Example

```python
from gridgulp import GridGulp
from gridgulp.config import Config

gg = GridGulp(config=Config.from_toml("gridgulp.toml"))
result = gg.schematize_sync("examples/spreadsheets/simple/basic_table.csv")

for entity in result.entities:
    print(entity.suggested_name, entity.record_count_estimate)
    for field in entity.fields:
        print(" ", field.id, field.semantic_type)
```

## What Gemma Sees

Gemma always receives structured text first. When a sheet is visually ambiguous, GridGulp also attaches rendered sheet images as fallback evidence:

- sheet density maps
- sampled rows
- heuristic proposal ranges and headers
- nearby boundary rows
- merged-cell summaries
- multi-row header hints
- semantic row summaries
- normalized overview and proposal crop PNGs when vision fallback triggers

## Notes

- Old Qwen, DeepSeek, Ollama-specific, and vision-first examples are no longer the supported path.
- The `heuristic_only` mode remains useful for debugging and regression comparison, but it is not the primary workflow.
