# Constants Refactoring Summary

## Overview
Centralized all constants from across the codebase into a single, well-organized module structure.

## Changes Made

### 1. Created Core Module Structure
- **`src/gridporter/core/`** - New directory for shared functionality
  - `__init__.py` - Exports all constants for easy access
  - `constants.py` - Central location for all constants
  - `exceptions.py` - Custom exception classes
  - `types.py` - Shared type definitions
  - `configurable_constants.py` - Mechanism for config overrides

### 2. Centralized Constants
Moved constants from multiple files into `core/constants.py`:

#### Island Detection Constants
- From `detectors/island_detector.py`
- Cell count thresholds, density values, aspect ratios, confidence scores

#### Format Analysis Constants
- From `detectors/format_analyzer.py`
- Blank row threshold, formatting thresholds, pattern detection parameters

#### Complex Table Constants
- From `agents/complex_table_agent.py`
- Confidence thresholds, preview counts, vision estimates

#### Cost Optimization Constants
- From `utils/cost_optimizer.py`
- Cost limits, cache TTL, batch sizes, token estimates

#### Excel Limits
- From `vision/bitmap_generator.py`
- Maximum rows/columns for XLS and XLSX formats

#### Keywords (Multi-language)
- From `format_analyzer.py` and `hierarchical_detector.py`
- Subtotal, total, and section keywords in multiple languages

### 3. Updated All References
Updated imports in 6 files to use centralized constants:
- `detectors/island_detector.py`
- `detectors/format_analyzer.py`
- `utils/cost_optimizer.py`
- `agents/complex_table_agent.py`
- `vision/bitmap_generator.py`
- `vision/hierarchical_detector.py`

### 4. Added Configuration Support
- Extended `config.py` with configurable threshold fields
- Created mechanism to override constants via configuration
- Maintains backward compatibility with default values

## Benefits

1. **Maintainability**: All constants in one place, easy to find and modify
2. **Consistency**: Shared keywords and thresholds across modules
3. **Configurability**: Key thresholds can be overridden via config
4. **Organization**: Clear categorization by functionality
5. **Type Safety**: Using dataclasses with Final annotations
6. **Documentation**: Each constant has clear comments

## Usage Examples

```python
# Import specific constant groups
from gridporter.core.constants import ISLAND_DETECTION, FORMAT_ANALYSIS

# Use constants
if cell_count >= ISLAND_DETECTION.MIN_CELLS_GOOD:
    confidence += ISLAND_DETECTION.CONFIDENCE_BOOST_LARGE

# Or import from core module
from gridporter.core import KEYWORDS
subtotal_keywords = list(KEYWORDS.SUBTOTAL_KEYWORDS)
```

## Future Improvements

1. Consider moving more magic numbers to constants
2. Add validation for constant values
3. Create constant presets for different use cases
4. Add more comprehensive configuration override support
