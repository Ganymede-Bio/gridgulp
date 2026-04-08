# GridGulp

[![PyPI version](https://badge.fury.io/py/gridgulp.svg)](https://pypi.org/project/gridgulp/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gridgulp.svg)](https://pypi.org/project/gridgulp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Gemma-first spreadsheet schematization for Excel, CSV, TSV, and text files.

## What Changed

GridGulp now supports two explicit execution modes:

- `gemma_first`: the supported primary path. Heuristics propose candidate regions, then Gemma finalizes boundaries, names, and merges/splits/additions. Structured text runs first; normalized sheet images are used only as a Gemma fallback for visually ambiguous sheets. The resulting tables are then lifted into canonical entities, fields, records, and relationships.
- `heuristic_only`: an explicit offline/debug escape hatch. This is useful for local debugging and for legacy test coverage, but it is not the main product path.

Only Gemma-family models are accepted for LLM execution.

## Installation

```bash
pip install gridgulp
```

For local development:

```bash
uv pip install -e ".[dev]"
```

## Configuration

GridGulp loads config in this order:

1. `gridgulp.toml`
2. `GRIDGULP_*` environment variables
3. CLI flags

Start from [gridgulp.toml.example](/Users/bensonlee/dev/gridgulp/gridgulp.toml.example).

Minimal Gemma-first config:

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

## CLI

Validate your config first:

```bash
gridgulp doctor
```

Validate multimodal support explicitly against Ollama:

```bash
gridgulp --gemma-base-url http://localhost:11434/v1 --gemma-model-id gemma4:26b doctor
```

Schematize a file:

```bash
gridgulp schematize path/to/report.xlsx
```

Emit JSON:

```bash
gridgulp schematize path/to/report.xlsx --json
```

Keep the lower-level table detector available when you need raw table boundaries:

```bash
gridgulp detect path/to/report.xlsx --save-debug-images .gridgulp-debug
```

Process a directory:

```bash
gridgulp schematize path/to/reports --recursive
```

## Python API

Gemma-first schematization:

```python
from gridgulp import GridGulp
from gridgulp.config import Config

config = Config.from_toml("gridgulp.toml")
gg = GridGulp(config=config)

result = gg.schematize_sync("sales_report.xlsx")
print(result.total_entities)
for entity in result.entities:
    print(entity.suggested_name, entity.record_count_estimate)
    for field in entity.fields:
        print(" ", field.id, field.semantic_type, field.physical_type)
```

Lower-level table debugging:

```python
from gridgulp import GridGulp
from gridgulp.config import Config, DetectionStrategy

config = Config(detection_strategy=DetectionStrategy.HEURISTIC_ONLY)
gg = GridGulp(config=config)

result = gg.detect_tables_sync("sales_report.xlsx")
```

## Schema Output

The primary output is `WorkbookSchemaResult`, which includes:

- workbook metadata
- per-sheet classified regions
- canonical entities
- normalized fields
- extracted record groups
- inferred relationships
- provenance and warnings

`DetectionResult` remains available as a lower-level primitive for debugging table boundaries.

## Detection And Schematization Architecture

For `gemma_first`, GridGulp runs:

1. file typing and file reading
2. deterministic proposal generation from the existing detector stack
3. structured-text prompt construction from sheet maps, boundary context, merged cells, multi-row header hints, and semantic row patterns
4. Gemma text reasoning over final `keep`, `merge`, `split`, `add`, and `reject` decisions
5. multimodal Gemma fallback with rendered PNG overview/crops when the text pass is low-confidence or visually ambiguous
6. post-processing into `TableInfo` objects with clipped ranges, deduplicated overlaps, header metadata, and semantic summaries
7. canonical schematization into regions, entities, fields, records, and relationships

## Supported Formats

- `.xlsx`
- `.xls`
- `.xlsm`
- `.csv`
- `.tsv`
- `.txt`

`.xlsb` is detected but not supported for reading.

## Development Workflow

```bash
gridgulp doctor
gridgulp schematize examples/spreadsheets/simple/basic_table.csv --json
pytest -q tests/test_detection_agent.py
```

To compare Gemma-first schema output against the deterministic baseline, switch `detection_strategy` to `heuristic_only` in config or via env:

```bash
GRIDGULP_DETECTION_STRATEGY=heuristic_only gridgulp schematize file.xlsx --json
```

## Docs

- [Usage Guide](/Users/bensonlee/dev/gridgulp/docs/USAGE_GUIDE.md)
- [Semantic Understanding](/Users/bensonlee/dev/gridgulp/docs/SEMANTIC_UNDERSTANDING.md)
- [Examples](/Users/bensonlee/dev/gridgulp/examples/README.md)
