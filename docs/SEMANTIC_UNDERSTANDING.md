# Semantic Understanding in GridGulp

GridGulp now uses structured-text Gemma reasoning for semantic table understanding, then lifts the final tables into canonical workbook schema objects.

## What Gemma Adds

The deterministic detector stack still finds candidate table regions, but Gemma is responsible for the final semantic decision layer:

- keep a good proposal
- merge multiple proposals into one logical table
- split one proposal into multiple tables
- add a missed table outside the heuristic output
- reject a bad proposal

Gemma also suggests semantic table names and operates over structured hints such as merged cells, multi-row headers, and subtotal patterns.

## Prompt Inputs

The Gemma prompt is built from:

- sheet dimensions and density
- row samples
- heuristic proposal ids, ranges, and headers
- nearby rows above, inside, and below each proposal
- merged-cell summaries
- multi-row header mappings when detected
- semantic row summaries from formatting analysis

## Output Shape

Gemma returns strict JSON:

```json
{
  "analysis_summary": {
    "sheet_complexity": "moderate",
    "reasoning_notes": "one logical report plus a side notes table",
    "confidence": 0.91
  },
  "decisions": [
    {
      "action": "merge",
      "proposal_ids": ["proposal_a", "proposal_b"],
      "range": "A1:F20",
      "suggested_name": "quarterly_sales_report",
      "confidence": 0.92,
      "reason": "These proposals are one continuous report."
    }
  ]
}
```

## Result Metadata

The final `TableInfo` objects include:

- `suggested_name`
- `header_info`
- `semantic_structure`
- `metadata["gemma_action"]`
- `metadata["reason"]`

Those table objects now feed the higher-level `schematize` surface, which emits workbook regions, entities, fields, record groups, relationships, provenance, and warnings.

## Debugging

If Gemma-first behavior looks wrong:

1. Run `gridgulp schematize ... --json`
2. Inspect workbook warnings plus sheet regions and entity metadata
3. Re-run `gridgulp detect ... --json` if you need raw table-boundary details
4. Re-run with `GRIDGULP_DETECTION_STRATEGY=heuristic_only`
5. Compare the deterministic proposals against the Gemma-final tables and schema
