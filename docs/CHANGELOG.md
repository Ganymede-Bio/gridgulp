# Changelog

All notable changes to GridGulp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-07-29

### Added
- **DataFrame Extraction**: Extract detected tables as pandas DataFrames
  - `DataFrameExtractor`: Intelligent header detection with multi-row support
  - Quality scoring for extracted tables (0-1 scale)
  - Automatic header detection including merged cells
  - Type consistency analysis across columns (checks up to 100 rows)
  - Plate map format detection (6, 24, 96, 384, 1536 wells)
  - Export to CSV, JSON, and summary formats

- **Enhanced TSV/Text File Detection**: Specialized detection for instrument output
  - `StructuredTextDetector`: Handles TSV files and instrument output
  - Column consistency analysis for better table separation
  - Wide table detection for plate data (50+ columns)
  - Structural analysis using row patterns and empty row detection
  - Better handling of multiple table formats in single file

- **Improved Island Detection**: Smarter heuristics for table separation
  - Column consistency threshold (80% similarity for grouping)
  - Configurable gap settings per file type (gap=0 for TSV, gap=1 for Excel)
  - Structural analysis mode for text files
  - Empty row detection for natural table boundaries
  - Aspect ratio and size-based confidence scoring

- **Documentation and Packaging**:
  - Streamlined README.md for clarity
  - NOTICE file with all dependency licenses
  - GitHub Action for PyPI releases with OIDC
  - Updated all documentation to reflect new features

### Changed
- Island detector now uses structural analysis for text files by default
- Constants centralized with file type-specific defaults
- Detection pipeline selects strategies based on file type
- DataFrame extraction integrated into main workflow

### Fixed
- TSV file detection now properly identifies multiple tables
- Wide plate data detection for instrument output files
- Header detection handles merged cells correctly
- Empty cells in headers properly handled

## [0.2.2] - 2025-07-28

### Added
- **Excel Metadata Extraction**: Leverage Excel's built-in table definitions
  - `ExcelMetadataExtractor`: Extract ListObjects (Excel tables) with high confidence
  - Named ranges detection for better table boundaries
  - Zero API cost for Excel native table detection
  - Integration with hybrid detection pipeline

- **Traditional Detection Methods**: Fast, zero-cost table detection
  - `SimpleCaseDetector`: Optimized for single tables starting near A1
  - `IslandDetector`: Connected component analysis for multi-table sheets
  - Flood-fill algorithm for finding disconnected data regions
  - Confidence scoring based on density, shape, and header patterns
  - Both methods completely free (no API calls)

- **Cost Optimization Framework**: Intelligent routing and budget management
  - `CostOptimizer`: Track API usage and costs per session/file
  - Budget limits with automatic fallback to free methods
  - Real-time cost tracking and reporting
  - Intelligent method selection based on complexity and budget
  - `OpenAIPricing`: Real OpenAI cost calculation with current pricing
  - `OpenAICostsAPI`: Integration with OpenAI's costs endpoint (admin key required)

- **Code Organization Improvements**: Major refactoring for maintainability
  - **Centralized Constants**: All constants moved to `src/gridgulp/core/constants.py`
  - **Core Module**: New organizational structure with:
    - `constants.py` - Categorized constants (Island Detection, Format Analysis, etc.)
    - `exceptions.py` - Custom exception classes for better error handling
    - `types.py` - Shared type definitions and aliases
    - `configurable_constants.py` - Mechanism for config overrides
  - **Contextual Logging**: New logging framework with file/sheet/table context
  - Better separation of concerns and cleaner imports

- **Enhanced Detection Pipeline**: Hybrid approach for optimal cost/quality
  - Try free methods first (Simple Case, Island Detection, Excel Metadata)
  - Fall back to vision processing only when needed
  - Confidence-based routing and early termination
  - Methods used tracking in detection results
  - Cost reporting in metadata

### Changed
- **Configuration System**: New options for cost control and traditional methods
  - `max_cost_per_session`: Budget limit for entire session (default: $1.00)
  - `max_cost_per_file`: Budget limit per file (default: $0.10)
  - `enable_simple_case_detection`: Enable fast single-table detection (default: True)
  - `enable_island_detection`: Enable multi-table detection (default: True)
  - `use_excel_metadata`: Use Excel ListObjects/named ranges (default: True)
  - `openai_admin_key`: Optional admin key for cost API access
  - Configurable detection thresholds for fine-tuning

- **GridGulp Core**: Enhanced with hybrid detection and cost tracking
  - Integrated traditional methods into main detection pipeline
  - Added cost tracking to all detection results
  - Enhanced metadata with methods used and performance metrics
  - Better handling of budget constraints

- **ComplexTableAgent**: Updated with cost-aware routing
  - Intelligent fallback from expensive to free methods
  - Enhanced confidence scoring combining multiple detection strategies
  - Better integration with Excel metadata extraction
  - Contextual logging throughout detection process

### Fixed
- Improved constants organization reduces code duplication
- Better error handling with custom exception classes
- Enhanced logging provides better debugging information
- More robust detection with multiple fallback strategies
- Fixed magic number scattered throughout codebase

### Examples
- `week6_excel_metadata_example.py`: Demonstrates Excel ListObjects extraction
- `week6_hybrid_detection_example.py`: Shows hybrid detection approach with cost tracking

### Documentation
- `docs/WEEK6_SUMMARY.md`: Comprehensive overview of Week 6 features
- `docs/testing/WEEK6_TESTING_GUIDE.md`: Manual testing procedures for new features
- `CONSTANTS_REFACTORING.md`: Details of code organization improvements

### Developer Experience
- Constants centralization improves maintainability
- Better type safety with custom types module
- Enhanced debugging with contextual logging
- Cleaner code organization with core module structure
- All code continues to pass strict linting standards

## [0.2.1] - 2025-07-27

### Added
- **Complex Table Detection**: New agent-based system for detecting complex spreadsheet structures
  - `ComplexTableAgent`: Orchestrates multi-row header detection, semantic analysis, and format preservation
  - Handles financial reports, pivot tables, and hierarchical data structures
  - Confidence scoring based on multiple detection strategies
  - Format-aware detection preserving semantic meaning

- **Multi-Row Header Detection**: Advanced header analysis with merged cell support
  - `MultiHeaderDetector`: Identifies hierarchical headers spanning multiple rows
  - Column hierarchy mapping for nested headers
  - Merged cell analysis with span detection
  - Support for complex pivot table structures
  - Header confidence scoring based on formatting and content

- **Semantic Structure Analysis**: Understanding table meaning beyond layout
  - `SemanticFormatAnalyzer`: Detects sections, subtotals, and grand totals
  - Row type classification (header, data, total, separator, section)
  - Format pattern detection for consistent styling
  - Preserves semantic blank rows and formatting
  - Section boundary detection for grouped data

- **Merged Cell Analysis**: Comprehensive merged cell handling
  - `MergedCellAnalyzer`: Detects and maps merged cell regions
  - Column span calculation for proper data alignment
  - Header cell hierarchy construction
  - Support for both Excel native and custom merge formats

- **Feature Collection System**: Telemetry for continuous improvement
  - `FeatureCollector`: Records detailed detection metrics
  - SQLite-based feature storage with 40+ metrics
  - Geometric features (rectangularness, density, contiguity)
  - Pattern features (type, orientation, headers)
  - Format features (bold headers, totals, sections)
  - Export to CSV for analysis in pandas/Excel
  - Configurable retention and privacy-preserving

- **Comprehensive Test Suite**: 100% test coverage for semantic features
  - 20 test scenarios covering all detection strategies
  - Integration tests for complex real-world patterns
  - Performance benchmarks for large files
  - Feature collection validation tests

### Changed
- **Configuration System**: Enhanced with new options
  - `use_vision`: Toggle vision-based detection (default: True)
  - `enable_feature_collection`: Enable telemetry (default: False)
  - `feature_db_path`: SQLite database location
  - `feature_retention_days`: Data retention period (default: 30)
  - All options configurable via environment variables

- **GridGulp Core**: Enhanced with semantic understanding
  - Integrated ComplexTableAgent into main detection pipeline
  - Added metadata fields for tracking LLM usage
  - Improved confidence scoring with multi-factor analysis
  - Better handling of sparse and complex spreadsheets

### Fixed
- Improved handling of sparse spreadsheets with many empty cells
- Better detection of table boundaries in complex layouts
- More accurate confidence scoring for multi-table sheets
- Fixed edge cases in merged cell detection
- Resolved issues with format preservation in detection results

### Examples
- `week5_complex_tables_with_features.py`: Demonstrates complex financial report detection
- `week5_feature_collection_example.py`: Shows feature collection and analysis workflow
- `feature_collection_example.py`: Basic feature collection usage

### Developer Experience
- All code now passes ruff linting standards
- Improved type hints throughout the codebase
- Better error messages for debugging
- Comprehensive docstrings for all new components

## [0.2.0] - 2025-07-25

### Added
- **Region Verification System**: AI proposal validation using geometry analysis
  - RegionVerifier class with configurable thresholds
  - Geometry metrics: rectangularness, filledness, density, contiguity
  - Pattern-specific verification for header-data, matrix, and hierarchical patterns
  - Feedback generation for invalid regions
  - Integration with vision pipeline for automatic filtering
- **Verification Configuration**: New config options for region verification
  - enable_region_verification, verification_strict_mode
  - Configurable thresholds for filledness, rectangularness, contiguity
  - Feedback loop settings for iterative refinement

### Changed
- **Vision Infrastructure**: Complete bitmap-based vision pipeline for table detection
  - Bitmap generation with adaptive compression (2-bit, 4-bit, sampled modes)
  - Multi-scale visualization with quadtree optimization
  - Pattern detection for sparse spreadsheets (header-data, matrix, form, time series)
  - Hierarchical detector for financial statements with indentation
  - Integrated 4-phase detection pipeline
- **Vision Model Integration**:
  - OpenAI GPT-4 Vision support
  - Ollama local vision model support (qwen2-vl)
  - Region proposal parsing with confidence scoring
  - Batch processing and caching for performance
- **Reader Implementations**:
  - ExcelReader with full formatting support (openpyxl/xlrd)
  - CSVReader with encoding detection and delimiter inference
  - CalamineReader (Rust-based) for 10-100x faster Excel processing
  - Factory pattern for automatic reader selection
  - Async/sync adapters for flexible usage
- **Telemetry System**:
  - OpenTelemetry integration for LLM usage tracking
  - Token usage metrics and cost tracking
  - Performance monitoring
- **Large File Support**:
  - Handle full Excel limits (1M×16K cells for .xlsx)
  - Memory-efficient processing with streaming
  - Adaptive sampling for oversized sheets

### Planned
- Region verification algorithms
- Geometry analysis tools
- Excel ListObjects detection integration
- LLM-powered range naming suggestions
- CLI tool with progress indicators
- Full agent implementation with openai-agents-python

## [0.2.0] - 2025-07-25

### Added
- **Vision Module Implementation**: Complete vision-based table detection system
  - 9 specialized modules for different aspects of vision processing
  - Support for sparse, hierarchical, and complex table patterns
  - Integration with both cloud and local vision models
- **High-Performance Readers**: CalamineReader for fast Excel processing
- **Pattern Detection**: Automatic detection of common spreadsheet patterns
- **Telemetry**: OpenTelemetry-based monitoring and cost tracking

### Changed
- Default Excel reader changed to CalamineReader for performance
- Enhanced file type detection with magic byte verification

### Technical Details
- Implemented adaptive bitmap compression for large spreadsheets
- Added quadtree spatial indexing for efficient processing
- Created hierarchical pattern detector for financial statements
- Built integrated pipeline coordinating multiple detection strategies

## [0.1.0] - 2025-07-23

### Added
- **Project Foundation**: Complete project structure and build system
- **Pydantic 2 Models**: Type-safe data models for FileInfo, TableInfo, and DetectionResult
- **Configuration System**: Comprehensive config with environment variable support
- **Main API Structure**: GridGulp class with cost-efficient architecture design
- **File Type Detection**: Basic file magic detection utilities (placeholder implementation)
- **Development Tooling**:
  - Pre-commit hooks with Black, Ruff, mypy
  - GitHub Actions CI/CD pipeline
  - uv-based build system with hatchling
  - EditorConfig and development guidelines
- **Documentation**:
  - Comprehensive README with usage examples
  - AGENT_ARCHITECTURE.md with cost-optimization strategy
  - PROJECT_PLAN.md with weekly breakdown
  - CONTRIBUTING.md with development guidelines
- **Test Infrastructure**: Complete test structure with pytest configuration
- **Examples**: Basic usage examples showing different configuration patterns
- **Cost-Efficient Design**:
  - Local-first processing architecture
  - Optional LLM integration with local model support
  - Token usage tracking and optimization

### Changed
- Build system updated to use uv instead of standard setuptools
- Project timeline restructured from 2-week phases to weekly milestones

### Security
- Input validation framework for file operations
- File size limits and processing timeouts
- Safe file type detection before processing
- No execution of Excel macros (by design)
