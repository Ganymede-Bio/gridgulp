# GridGulp Usage Guide

## Execution Modes

GridGulp supports two explicit modes:

- `gemma_first`: heuristic proposals plus Gemma finalization, with multimodal image fallback for visually ambiguous sheets, followed by canonical workbook schematization
- `heuristic_only`: deterministic-only debugging mode

`gemma_first` is the intended production path.

## Configure Gemma

Create `gridgulp.toml` from the example file:

```toml
detection_strategy = "gemma_first"

[gemma]
enabled = true
base_url = "http://localhost:11434/v1"
model_id = "gemma4:26b"
timeout_seconds = 60
max_retries = 2
max_tokens = 1200

[vision]
enabled = true
mode = "fallback_only"
max_proposal_crops = 4
crop_margin_cells = 1
```

Or override via env:

```bash
export GRIDGULP_DETECTION_STRATEGY=gemma_first
export GRIDGULP_GEMMA_BASE_URL=http://localhost:11434/v1
export GRIDGULP_GEMMA_MODEL_ID=gemma4:26b
export GRIDGULP_VISION_MODE=fallback_only
```

## CLI Workflow

Validate config and endpoint health:

```bash
gridgulp doctor
```

This checks both `/v1/models` and a tiny multimodal probe when vision fallback is enabled.

Schematize one file:

```bash
gridgulp schematize data/report.xlsx
```

Schematize one file as JSON:

```bash
gridgulp schematize data/report.xlsx --json
```

Inspect raw table detection and save the rendered overview and crop PNGs used for fallback inspection:

```bash
gridgulp detect data/report.xlsx --save-debug-images .gridgulp-debug
```

Schematize a directory:

```bash
gridgulp schematize data/reports --recursive
```

## Python Workflow

```python
from gridgulp import GridGulp
from gridgulp.config import Config

config = Config.from_toml("gridgulp.toml")
gg = GridGulp(config=config)

result = gg.schematize_sync("data/report.xlsx")
print(result.total_entities)
for entity in result.entities:
    print(entity.suggested_name, entity.record_count_estimate)
    for field in entity.fields:
        print(" ", field.id, field.semantic_type)
```

## Heuristic-Only Debugging

Use this when you want to compare against the deterministic proposal layer:

```python
from gridgulp import GridGulp
from gridgulp.config import Config, DetectionStrategy

gg = GridGulp(config=Config(detection_strategy=DetectionStrategy.HEURISTIC_ONLY))
result = gg.schematize_sync("data/report.xlsx")
```

Or via CLI:

```bash
GRIDGULP_DETECTION_STRATEGY=heuristic_only gridgulp schematize data/report.xlsx --json
```

## Output Notes

Gemma-first schematization attaches:

- workbook-level canonical entities, fields, record groups, and relationships
- per-sheet `regions` with semantic roles like `data_table`, `title_block`, `notes_block`, and `matrix_block`
- workbook warnings for ambiguous or low-confidence schema decisions
- detection metadata remains available through the lower-level `detect` command and API

## Policy

- Only Gemma-family model ids are accepted.
- GridGulp does not allow per-request model switching.
- `gemma_first` fails closed when Gemma config is missing or the endpoint is unavailable.
