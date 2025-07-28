# GridPorter Agent-Tool Architecture

## Overview

This document defines the architectural principles for separating Agents (decision-makers) from Tools (executors) in GridPorter. This separation creates a cleaner, more maintainable, and more testable codebase.

## Core Principles

### Agents: Strategic Decision Makers

**Agents** are autonomous components that:
- Make strategic decisions about which approaches to use
- Maintain state and context throughout their operations
- Handle errors, retries, and fallback strategies
- Coordinate multiple tools to achieve complex goals
- Adapt their behavior based on results

**Key Characteristics:**
- Stateful (maintain context)
- Implement retry/fallback logic
- Make decisions based on heuristics or ML models
- Can invoke other agents or tools
- Handle complex workflows

### Tools: Tactical Executors

**Tools** are deterministic functions that:
- Perform specific, well-defined operations
- Are stateless and pure (same input → same output)
- Have no decision-making logic
- Are highly reusable across different agents
- Focus on doing one thing well

**Key Characteristics:**
- Stateless (no context between calls)
- Deterministic behavior
- Single responsibility
- Easy to test in isolation
- No external API calls (those go through agents)

## The 5 Core Agents

### 1. PipelineOrchestrator

**Purpose**: Orchestrate the entire table detection pipeline

**Responsibilities:**
- Determine optimal workflow based on file characteristics
- Manage cost/performance tradeoffs
- Coordinate other agents
- Handle global error recovery
- Track overall progress and metrics

**Key Decisions:**
- Which detection strategy to use (vision, heuristic, hybrid)
- Whether to use progressive refinement
- When to stop retrying
- How to prioritize tables for extraction

**State Management:**
- Current pipeline phase
- Accumulated costs
- Tables detected so far
- Error history

### 2. DetectionAgent

**Purpose**: Detect table boundaries using various strategies

**Responsibilities:**
- Choose detection strategy based on file type and size
- Try fast paths first (named ranges, ListObjects)
- Fall back to vision detection if needed
- Manage progressive refinement for large sheets

**Key Decisions:**
- Use named ranges vs. vision detection
- Compression level for large sheets
- Which regions need detailed analysis
- When to stop refining boundaries

**Tools Used:**
- `named_range_detector`
- `list_object_extractor`
- `connected_components_detector`
- `data_region_preprocessor`
- `bitmap_generator`

### 3. VisionAgent

**Purpose**: Manage all vision model interactions

**Responsibilities:**
- Build appropriate prompts based on context
- Handle multi-image requests
- Implement retry logic for low confidence
- Provide feedback for refinement
- Manage vision API costs

**Key Decisions:**
- Single vs. multi-scale image analysis
- Prompt strategy based on table complexity
- When to request human-in-the-loop
- How to interpret ambiguous responses

**Tools Used:**
- `multi_scale_bitmap_generator`
- `vision_prompt_builder`
- `compression_selector`
- `image_encoder`
- `region_highlighter`
- `response_parser`

### 4. ExtractionAgent

**Purpose**: Extract data from detected tables

**Responsibilities:**
- Determine extraction complexity
- Handle special structures (merged cells, formulas)
- Preserve formatting when needed
- Manage hierarchical data extraction

**Key Decisions:**
- Simple vs. complex extraction path
- Which formatting to preserve
- How to handle merged cells
- Whether to evaluate formulas

**Tools Used:**
- `cell_data_extractor`
- `header_parser`
- `merge_cell_handler`
- `formula_evaluator`
- `format_extractor`
- `hierarchy_analyzer`

### 5. AnalysisAgent

**Purpose**: Understand table semantics and generate metadata

**Responsibilities:**
- Analyze table purpose and structure
- Generate meaningful names
- Create field descriptions
- Build pandas import configurations
- Provide business context

**Key Decisions:**
- Table categorization approach
- Whether to use LLM for naming
- Level of semantic analysis needed
- Metadata completeness vs. cost

**Tools Used:**
- `semantic_classifier`
- `field_analyzer`
- `name_generator`
- `relationship_detector`
- `metadata_builder`
- `pandas_config_generator`

## Tool Categories and Examples

### Detection Tools

Located in `tools/detection/`:

```python
def detect_named_ranges(workbook: Workbook) -> List[NamedRange]:
    """Extract all named ranges from an Excel workbook."""

def extract_list_objects(worksheet: Worksheet) -> List[TableObject]:
    """Extract Excel ListObjects (formal tables) from a worksheet."""

def find_connected_components(
    sheet_data: SheetData,
    gap_threshold: int = 3
) -> List[DataRegion]:
    """Find connected regions of data using flood fill."""

def preprocess_data_regions(
    sheet_data: SheetData,
    min_cells: int = 4
) -> DataRegionSummary:
    """Fast scan to identify regions containing data."""
```

### Vision Tools

Located in `tools/vision/`:

```python
def generate_multi_scale_bitmaps(
    sheet_data: SheetData,
    regions: List[DataRegion],
    compression_levels: List[int]
) -> MultiScaleBitmaps:
    """Generate bitmaps at multiple compression levels."""

def build_vision_prompt(
    template: str,
    images: List[VisionImage],
    context: Dict[str, Any]
) -> str:
    """Build a prompt for vision model with template and context."""

def calculate_optimal_compression(
    data_bounds: DataBounds,
    target_size_mb: float = 20.0
) -> CompressionStrategy:
    """Calculate optimal compression for size constraints."""

def highlight_region_with_context(
    image: Image,
    region: Rectangle,
    context_cells: int = 10
) -> Image:
    """Add visual indicators for region and context."""
```

### Verification Tools

Located in `tools/verification/`:

```python
def calculate_rectangularness(
    region: DataRegion,
    sheet_data: SheetData
) -> float:
    """Calculate how rectangular a region is (0.0 to 1.0)."""

def check_column_consistency(
    data: List[List[Any]],
    threshold: float = 0.8
) -> ConsistencyReport:
    """Check if columns have consistent data types."""

def validate_table_bounds(
    bounds: TableBounds,
    sheet_data: SheetData
) -> ValidationResult:
    """Validate proposed table boundaries."""
```

### Extraction Tools

Located in `tools/extraction/`:

```python
def extract_cell_data(
    sheet_data: SheetData,
    bounds: TableBounds
) -> List[List[Any]]:
    """Extract raw cell values from a region."""

def parse_headers(
    data: List[List[Any]],
    max_header_rows: int = 5
) -> HeaderStructure:
    """Analyze and parse complex headers."""

def resolve_merged_cells(
    sheet_data: SheetData,
    bounds: TableBounds
) -> MergeMap:
    """Create a map of merged cell relationships."""

def extract_cell_formats(
    sheet_data: SheetData,
    bounds: TableBounds
) -> FormatMatrix:
    """Extract formatting information for cells."""
```

### Analysis Tools

Located in `tools/analysis/`:

```python
def classify_table_type(
    data: List[List[Any]],
    headers: List[str]
) -> TableType:
    """Classify table type (financial, inventory, etc.)."""

def analyze_field_semantics(
    column_data: List[Any],
    header: str
) -> FieldSemantics:
    """Analyze what a field represents."""

def generate_table_name(
    headers: List[str],
    table_type: TableType,
    context: Dict[str, Any]
) -> str:
    """Generate a meaningful table name."""

def detect_relationships(
    tables: List[TableInfo]
) -> List[TableRelationship]:
    """Detect relationships between tables."""
```

## Directory Structure

```
src/gridporter/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py              # Abstract base class
│   ├── pipeline_orchestrator.py   # Main orchestrator
│   ├── detection_agent.py         # Detection strategies
│   ├── vision_agent.py            # Vision model interface
│   ├── extraction_agent.py        # Data extraction
│   └── analysis_agent.py          # Semantic analysis
│
├── tools/
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── named_range_detector.py
│   │   ├── list_object_extractor.py
│   │   ├── connected_components.py
│   │   ├── data_region_preprocessor.py
│   │   └── pattern_matcher.py
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── bitmap_generator.py
│   │   ├── compression_calculator.py
│   │   ├── prompt_builder.py
│   │   ├── image_encoder.py
│   │   ├── region_highlighter.py
│   │   └── response_parser.py
│   │
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── geometry_validator.py
│   │   ├── consistency_checker.py
│   │   ├── boundary_validator.py
│   │   └── metric_calculator.py
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── cell_data_extractor.py
│   │   ├── header_parser.py
│   │   ├── merge_cell_handler.py
│   │   ├── formula_processor.py
│   │   ├── format_extractor.py
│   │   └── hierarchy_handler.py
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── semantic_classifier.py
│       ├── field_analyzer.py
│       ├── name_generator.py
│       ├── relationship_detector.py
│       ├── metadata_builder.py
│       └── pandas_config_generator.py
│
└── prompts/
    ├── __init__.py
    ├── vision/
    │   ├── detection_prompts.json
    │   ├── refinement_prompts.json
    │   └── context_prompts.json
    └── analysis/
        ├── naming_prompts.json
        └── semantic_prompts.json
```

## Implementation Example

### Agent Example: DetectionAgent

```python
class DetectionAgent(BaseAgent):
    """Agent responsible for detecting tables in spreadsheets."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.strategies = ['named_range', 'list_object', 'vision', 'heuristic']

    async def detect_tables(
        self,
        sheet_data: SheetData,
        options: DetectionOptions
    ) -> List[TableProposal]:
        """Detect all tables in a sheet using appropriate strategies."""

        # Try fast path first - Excel's built-in table definitions
        if sheet_data.has_list_objects:
            self.logger.info("Trying ListObject detection")
            tables = extract_list_objects(sheet_data.worksheet)
            if tables:
                return self._convert_to_proposals(tables)

        # Check named ranges
        if sheet_data.has_named_ranges:
            self.logger.info("Checking named ranges")
            ranges = detect_named_ranges(sheet_data.workbook)
            table_ranges = self._filter_table_ranges(ranges)
            if table_ranges:
                return self._convert_to_proposals(table_ranges)

        # Fall back to vision detection for complex sheets
        if self._should_use_vision(sheet_data, options):
            self.logger.info("Using vision detection")
            return await self._vision_detection(sheet_data, options)

        # Final fallback: heuristic detection
        self.logger.info("Using heuristic detection")
        return self._heuristic_detection(sheet_data)

    def _should_use_vision(
        self,
        sheet_data: SheetData,
        options: DetectionOptions
    ) -> bool:
        """Decide whether to use vision detection."""
        # Decision logic based on sheet characteristics
        if not options.enable_vision:
            return False

        data_summary = preprocess_data_regions(sheet_data)

        # Use vision for complex layouts
        if data_summary.disconnected_regions > 3:
            return True

        # Use vision for sparse sheets
        if data_summary.data_density < 0.1:
            return True

        return False
```

### Tool Example: Named Range Detector

```python
def detect_named_ranges(workbook: Workbook) -> List[NamedRange]:
    """
    Extract all named ranges from an Excel workbook.

    This is a pure function that extracts defined names from the workbook
    without making any decisions about whether they represent tables.

    Args:
        workbook: The Excel workbook object

    Returns:
        List of NamedRange objects with name and cell references
    """
    named_ranges = []

    for defined_name in workbook.defined_names.definedName:
        # Skip print areas and other special ranges
        if defined_name.name.startswith('_'):
            continue

        # Parse the range reference
        try:
            sheet_name, cell_range = defined_name.value.split('!')
            sheet_name = sheet_name.strip("'")

            named_ranges.append(NamedRange(
                name=defined_name.name,
                sheet=sheet_name,
                range=cell_range,
                scope=defined_name.scope
            ))
        except ValueError:
            # Skip malformed references
            continue

    return named_ranges
```

## Testing Strategy

### Agent Testing

Agents require integration tests with mocked tools:

```python
class TestDetectionAgent:
    @pytest.fixture
    def mock_tools(self, monkeypatch):
        """Mock all tools used by DetectionAgent."""
        monkeypatch.setattr(
            'gridporter.tools.detection.extract_list_objects',
            lambda ws: [MockListObject()]
        )
        # ... mock other tools

    async def test_detection_strategy_selection(self, mock_tools):
        """Test that agent selects appropriate strategy."""
        agent = DetectionAgent(Config())

        # Sheet with ListObjects should use fast path
        sheet_with_tables = create_sheet_with_list_objects()
        proposals = await agent.detect_tables(sheet_with_tables)
        assert proposals[0].detection_method == 'list_object'
```

### Tool Testing

Tools can be tested in isolation with simple unit tests:

```python
class TestNamedRangeDetector:
    def test_extract_named_ranges(self):
        """Test extraction of named ranges."""
        workbook = create_test_workbook()
        workbook.add_named_range('SalesData', 'Sheet1!A1:D10')
        workbook.add_named_range('_xlnm.Print_Area', 'Sheet1!A1:Z100')

        ranges = detect_named_ranges(workbook)

        # Should extract user-defined ranges, not system ranges
        assert len(ranges) == 1
        assert ranges[0].name == 'SalesData'
        assert ranges[0].range == 'A1:D10'
```

## Migration Path

### Phase 1: Create Tool Infrastructure
1. Create `tools/` directory structure
2. Implement base tool utilities
3. Add tool registration system

### Phase 2: Extract Tools from Existing Code
1. Identify pure functions in current agents
2. Move them to appropriate tool modules
3. Update imports

### Phase 3: Refactor Agents
1. Simplify agents to use tools
2. Remove duplicate decision logic
3. Consolidate to 5 core agents

### Phase 4: Testing and Documentation
1. Write comprehensive tool tests
2. Update agent tests
3. Document tool APIs

## Benefits of This Architecture

1. **Clarity**: Clear separation of concerns between "what" and "how"
2. **Testability**: Tools are pure functions, easy to test
3. **Reusability**: Tools can be used by multiple agents
4. **Maintainability**: Fewer agents means less coordination complexity
5. **Performance**: Tools can be optimized independently
6. **Debuggability**: Easier to trace execution flow
7. **Extensibility**: New tools can be added without changing agents

## Conclusion

This architecture creates a clean, maintainable system where:
- **5 agents** make all strategic decisions
- **30-40 tools** perform all tactical operations
- Clear boundaries exist between decision-making and execution
- Testing and debugging are straightforward
- New functionality can be added easily

The result is a more robust, efficient, and understandable system that scales well with complexity.
