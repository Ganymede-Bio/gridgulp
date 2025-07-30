# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] - 2025-07-29

### Added
- **Enhanced Table Detection Algorithms**: Major improvements to table detection accuracy
  - **Column Gap Detection**: Prevents merging of side-by-side tables separated by empty columns
  - **Empty Row Tolerance**: Configurable tolerance (0-5 rows) prevents splitting tables with section breaks
  - **Border-Based Detection**: Uses Excel cell borders for precise table boundary detection
  - **Weighted Scoring System**: Comprehensive confidence scoring using 7 weighted factors

- **New Configuration Options**: Fine-tune detection behavior
  - `empty_row_tolerance`: Number of empty rows to tolerate within tables (default: 1)
  - `column_gap_prevents_merge`: Prevent merging across empty columns (default: True)
  - `use_border_detection`: Enable border-based boundaries (default: True)
  - `min_column_overlap_for_merge`: Required column overlap ratio (default: 0.5)

### Improved
- **Detection Accuracy**: Addresses specific issues with complex layouts
  - Side-by-side tables (like dashboards) now correctly detected as separate
  - Tables with subtotals or section breaks stay intact
  - Border patterns provide precise table edges
  - Better confidence scores reflect actual table quality

- **Scoring Components**: Enhanced confidence calculation
  - Size Score (20%): Relative and absolute table size
  - Density Score (15%): Data density within region
  - Shape Score (10%): Preference for rectangular tables
  - Header Score (15%): Detection of header rows
  - Border Score (15%): Clean border patterns
  - Formatting Score (15%): Consistency analysis
  - Isolation Score (10%): Independence from other tables

## [0.3.3] - 2025-07-29

### Added
- **Smart Table Boundary Detection**: Enhanced formatting-based table separation
  - Headers now correctly included with their data sections
  - Well-separated table detection preserves natural boundaries
  - Adaptive merge distance based on formatting patterns and empty row separation
  - Formatting boundary detection using cell styling (bold, background colors)

- **Comprehensive Test Coverage**: Added extensive test coverage for key components
  - Enhanced test coverage for island detection, file reading, and core functionality
  - Better test reliability and API compliance

### Changed
- **Test Suite Cleanup**: Removed all skipped tests and updated implementations
  - All tests now pass with proper API implementations
  - Cleaner, more maintainable test suite

### Improved
- **Island Detection Algorithm**: Smarter merging logic
  - Detects well-separated tables and skips aggressive merging
  - Preserves formatting-based table boundaries
  - Improved confidence calculation with proper order of operations
  - Better empty cell detection using `is_empty` property

- **Table Detection Accuracy**: Reduced false merging of separate tables
  - Tables separated by empty rows are no longer incorrectly merged
  - Headers stay with their respective data sections
  - Better handling of multi-table sheets with different formatting

### Removed
- **XLSB File Support**: Excel Binary format is no longer supported
  - XLSB files are detected but will return a clear error message
  - Users must save XLSB files as XLSX format in Excel before processing

### Fixed
- Fixed header exclusion issue where headers were separated from data
- Fixed confidence calculation bug with order of operations
- Fixed empty cell detection treating empty strings as data
- Fixed file format detection to properly identify XLSB files and provide clear error messages
- Fixed GitHub release action configuration with proper tag handling
- Updated all tests to match actual API implementations
- Resolved test failures and inconsistencies

## [0.3.2] - 2025-07-29

### Added
- **MkDocs Documentation Site**: Professional documentation with Material theme
  - Comprehensive API documentation
  - Usage examples and guides
  - GitHub Pages integration

- **MyPy Type Checking**: Full static type checking support
  - Complete type annotations throughout codebase
  - MyPy configuration for strict type checking
  - Improved code quality and developer experience

### Improved
- **Enhanced README**: Better examples and documentation
  - Jupyter notebook examples
  - DataFrame output demonstrations
  - Clearer usage instructions

### Fixed
- Resolved all mypy type annotation errors
- Fixed documentation build and rendering issues
- Improved MkDocs configuration and theme setup
- Fixed various CI/CD pipeline issues

## [0.3.1] - 2025-07-29

### Changed
- CI improvements:
  - Added Python version matrix testing (3.10, 3.11, 3.12, 3.13)
  - Updated ruff target version to py310 (minimum supported)

### Fixed
- Fixed build configuration to match new project name
- Fixed all linting issues identified by ruff
- Added appropriate lint rule exceptions for tests, examples, and scripts
- Fixed CellRange/TableRange instantiation to use keyword arguments
- Fixed StructuredTextDetector dimension calculations
- Fixed header extraction in StructuredTextDetector
- Fixed test compatibility issues in DataFrameExtractor tests

## [0.3.0] - 2025-07-28

### Added
- Text file support with automatic CSV/TSV detection
- Sophisticated multi-layer encoding detection:
  - BOM (Byte Order Mark) detection for UTF-8, UTF-16, UTF-32
  - Advanced chardet analysis with language detection
  - Pattern-based detection for scientific data
  - Intelligent fallback chains
- TextReader class for processing text files as tabular data
- Enhanced delimiter detection for scientific instrument data

### Changed
- **BREAKING**: Simplified architecture - removed all agent dependencies
- Reduced codebase substantially while maintaining functionality
- Replaced complex agent orchestration with direct detection approach
- SimpleCaseDetector and IslandDetector now handle most use cases
- Improved file type detection to handle UTF-16 files correctly
- capture_detection_outputs.py now processes ALL files in examples directory

### Removed
- All agent classes and dependencies
- Vision orchestrator complexity
- Complex multi-agent coordination
- Unnecessary abstraction layers

### Fixed
- UTF-16 file detection (previously misidentified as "autohotkey")
- Scientific data files with varying column counts now properly detected
- Text files with complex encodings now handled correctly

## [0.2.2] - 2025-07-15

### Added
- Zero-Cost Detection Methods:
  - SimpleCaseDetector for fast single-table detection
  - IslandDetector for multi-table detection using connected components
  - Excel metadata extraction from ListObjects and named ranges
- Cost Optimization Framework:
  - Budget management with session and per-file limits
  - Real-time cost tracking and reporting
  - Intelligent routing between detection methods
  - Automatic fallback to free methods when budget exceeded
- Hybrid Detection Pipeline:
  - Try free methods first, use vision only when needed
  - Confidence-based routing for optimal cost/quality balance
  - Early termination when high-confidence results achieved
- Code Organization Improvements:
  - Centralized constants in core/constants.py
  - Custom exception classes and type definitions
  - Enhanced contextual logging throughout

### Changed
- Default detection strategy now prioritizes free methods
- Vision-based detection only used when necessary
- Improved performance with fast-path algorithms

## [0.2.1] - 2025-07-01

### Added
- Multi-row header detection with column hierarchy
- Merged cell analysis and mapping
- Section and subtotal detection
- Format pattern recognition
- Preservation of semantic blank rows
- Feature collection system for continuous improvement
- Complex table detection agent

### Changed
- Enhanced semantic understanding of spreadsheet structure
- Improved confidence scoring with multi-factor analysis
- Better handling of financial and scientific data formats

## [0.2.0] - 2025-06-15

### Added
- Vision-based table detection using bitmap analysis
- AI-powered table naming suggestions
- Async processing for better performance
- CalamineReader for high-speed Excel processing
- Comprehensive error handling

### Changed
- Refactored detection pipeline for extensibility
- Improved memory efficiency for large files
- Enhanced type safety with Pydantic 2

## [0.1.0] - 2025-06-01

### Added
- Initial release
- Basic Excel and CSV file support
- Simple table detection
- File type validation using magic bytes
