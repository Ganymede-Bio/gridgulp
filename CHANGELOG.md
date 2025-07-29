# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-07-29

### Changed
- **Project Rename**: Renamed from GridPorter to GridGulp
  - Updated all package references throughout codebase
  - Renamed source directory from `src/gridporter/` to `src/gridgulp/`
  - Updated project metadata and documentation
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
