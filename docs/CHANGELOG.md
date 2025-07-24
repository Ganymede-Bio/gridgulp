# Changelog

All notable changes to GridPorter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Basic file reading capabilities for Excel and CSV
- Single table detection algorithm
- Excel ListObjects detection
- Mask-based island detection for multiple tables
- Format and header heuristics
- LLM-powered range naming suggestions
- CLI tool with progress indicators
- Full agent implementation with openai-agents-python

## [0.1.0] - 2025-01-23

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
