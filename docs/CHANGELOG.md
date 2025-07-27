# Changelog

All notable changes to GridPorter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Main API Structure**: GridPorter class with cost-efficient architecture design
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

## Roadmap

### [0.2.0] - Planned
- Google Sheets API integration
- Improved merged cell handling
- Performance optimizations for large files
- WebSocket support for real-time updates

### [0.3.0] - Planned
- ML-based table detection using table-transformer
- Automatic data type inference
- Table relationship detection
- Plugin system for custom detectors

### [1.0.0] - Planned
- Production-ready release
- Stable API
- Full documentation
- Performance guarantees
- Enterprise features

[Unreleased]: https://github.com/yourusername/gridporter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/gridporter/releases/tag/v0.1.0
