"""Tests for enhanced file format detection."""

import tempfile
import zipfile
from pathlib import Path

import pytest
from gridgulp.models.file_info import FileType, UnsupportedFormatError
from gridgulp.utils.file_magic import (
    DetectionResult,
    FileFormatDetector,
    detect_file_info,
    detect_file_info_safe,
    detect_file_type,
)


class TestFileFormatDetector:
    """Test the enhanced file format detector."""

    def setup_method(self):
        """Setup for each test."""
        self.detector = FileFormatDetector()

    def test_detector_initialization(self):
        """Test detector initializes with library availability checks."""
        assert hasattr(self.detector, "magic_available")
        assert hasattr(self.detector, "filetype_available")

    def test_csv_detection_by_content(self):
        """Test CSV detection by analyzing content structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
            f.write("Name,Age,City\n")
            f.write("Alice,25,NYC\n")
            f.write("Bob,30,LA\n")
            f.flush()

            file_path = Path(f.name)

        try:
            result = self.detector.detect(file_path)

            assert result.detected_type == FileType.CSV
            assert result.confidence > 0.5
            # With Magika available, it should be the primary detection method
            if self.detector.magika_available:
                assert result.method == "magika"
            else:
                assert result.method in ["content", "magic_mime"]

        finally:
            file_path.unlink()

    def test_tsv_detection_by_content(self):
        """Test TSV detection by analyzing tab delimiters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
            f.write("Name\tAge\tCity\n")
            f.write("Alice\t25\tNYC\n")
            f.write("Bob\t30\tLA\n")
            f.flush()

            file_path = Path(f.name)

        try:
            result = self.detector.detect(file_path)

            assert result.detected_type == FileType.TSV
            assert (
                result.confidence > 0.2
            )  # Lower threshold for TSV as it might be detected as text/plain

        finally:
            file_path.unlink()

    def test_format_mismatch_detection(self):
        """Test detection of format mismatches."""
        # Create CSV content with .xls extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xls", delete=False) as f:
            f.write("Name,Age,Department\n")
            f.write("Alice,25,Engineering\n")
            f.write("Bob,30,Marketing\n")
            f.flush()

            file_path = Path(f.name)

        try:
            result = self.detector.detect(file_path)

            assert result.detected_type == FileType.CSV
            assert result.extension_type == FileType.XLS
            assert result.format_mismatch is True

        finally:
            file_path.unlink()

    def test_xlsx_zip_structure_detection(self):
        """Test XLSX detection by analyzing ZIP content."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            # Create a ZIP file with Excel-like structure
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("[Content_Types].xml", '<?xml version="1.0"?>')
                zf.writestr("_rels/.rels", '<?xml version="1.0"?>')
                zf.writestr("xl/workbook.xml", '<?xml version="1.0"?>')
                zf.writestr("xl/sharedStrings.xml", '<?xml version="1.0"?>')
                zf.writestr("xl/styles.xml", '<?xml version="1.0"?>')

            result = self.detector.detect(zip_path)

            assert result.detected_type == FileType.XLSX
            assert result.confidence >= 0.9

        finally:
            zip_path.unlink()

    def test_xlsm_detection_with_vba(self):
        """Test XLSM detection when VBA project is present."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            # Create a ZIP file with Excel + VBA structure
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("[Content_Types].xml", '<?xml version="1.0"?>')
                zf.writestr("_rels/.rels", '<?xml version="1.0"?>')
                zf.writestr("xl/workbook.xml", '<?xml version="1.0"?>')
                zf.writestr("xl/vbaProject.bin", "binary content")  # VBA indicator

            result = self.detector.detect(zip_path)

            # Should detect as XLSM or at least XLSX (depends on how many indicators match)
            assert result.detected_type in [FileType.XLSM, FileType.XLSX]
            assert result.confidence >= 0.7

        finally:
            zip_path.unlink()

    def test_magic_bytes_detection(self):
        """Test detection using magic byte signatures."""
        # Create file with XLS magic bytes (OLE2)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write OLE2 signature
            f.write(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
            f.write(b"some more content to make it substantial")
            f.flush()

            file_path = Path(f.name)

        try:
            result = self.detector.detect(file_path)

            # Should detect as XLS based on magic bytes
            assert result.detected_type == FileType.XLS
            assert result.confidence >= 0.8

        finally:
            file_path.unlink()

    def test_binary_vs_text_classification(self):
        """Test classification between binary and text files."""
        # Test with binary content
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")
            f.flush()
            binary_path = Path(f.name)

        # Test with text content
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("This is plain text content\nwith multiple lines\n")
            f.flush()
            text_path = Path(f.name)

        try:
            # Binary file should not be detected as text format
            binary_result = self.detector.detect(binary_path)
            assert binary_result.detected_type != FileType.CSV
            assert binary_result.detected_type != FileType.TSV

            # Text file should potentially be detected as delimited format
            text_result = self.detector.detect(text_path)
            # May or may not be CSV/TSV depending on content, but should not crash
            assert text_result.detected_type in [
                FileType.CSV,
                FileType.TSV,
                FileType.TXT,
                FileType.UNKNOWN,
            ]

        finally:
            binary_path.unlink()
            text_path.unlink()

    def test_encoding_detection(self):
        """Test encoding detection for text files."""
        # Create file with UTF-8 content
        content = "Name,Ñame,Naïve\nAlice,José,François\n"

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            f.flush()
            file_path = Path(f.name)

        try:
            result = self.detector.detect(file_path)

            assert result.detected_type == FileType.CSV
            assert result.encoding is not None
            # Should detect UTF-8 or similar
            assert result.encoding.lower() in ["utf-8", "ascii"]

        finally:
            file_path.unlink()

    def test_inconsistent_delimiters(self):
        """Test handling of files with inconsistent delimiters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Name,Age,City\n")  # CSV style
            f.write("Alice\t25\tNYC\n")  # TSV style
            f.write("Bob,30,LA\n")  # CSV style again
            f.flush()

            file_path = Path(f.name)

        try:
            result = self.detector.detect(file_path)

            # Should still detect some kind of delimited format
            assert result.detected_type in [FileType.CSV, FileType.TSV]
            # With Magika, it will have high confidence; without Magika, lower confidence due to inconsistency
            if self.detector.magika_available:
                # Magika sees it as CSV and gives high confidence
                assert result.confidence > 0.9
            else:
                # Content analysis should give lower confidence due to inconsistency
                assert result.confidence < 0.9

        finally:
            file_path.unlink()

    def test_empty_file_handling(self):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            file_path = Path(f.name)

        try:
            result = self.detector.detect(file_path)

            # With Magika available, empty file might be detected differently
            if self.detector.magika_available:
                # Magika might detect as 'unknown' for empty files
                assert result.detected_type in [FileType.CSV, FileType.UNKNOWN]
                if result.detected_type == FileType.UNKNOWN:
                    assert result.method in ["magika", "extension_fallback"]
                    # Magika can give high confidence for detecting empty files
                    assert result.confidence >= 0.1
                else:
                    assert result.method == "extension"  # Falls back after Magika fails
                    assert result.confidence <= 0.3
            else:
                # Should fall back to extension-based detection
                assert result.detected_type == FileType.CSV
                assert result.method in ["extension", "extension_fallback"]
                assert result.confidence <= 0.3

        finally:
            file_path.unlink()

    def test_fallback_detection(self):
        """Test fallback detection when all methods fail."""
        # Create file with unknown extension and unrecognizable content
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
            f.write(b"some random binary content that does not match any pattern")
            f.flush()
            file_path = Path(f.name)

        try:
            result = self.detector.detect(file_path)

            # TXT is now a valid detected type for text files
            assert result.detected_type in [FileType.TXT, FileType.UNKNOWN]
            # With Magika, it might detect the file type or fall back to extension
            if self.detector.magika_available:
                assert result.method in ["magika", "extension", "extension_fallback"]
                # Magika can detect TXT files with high confidence
                if result.detected_type == FileType.TXT:
                    assert result.confidence > 0.5
                else:
                    assert result.confidence <= 0.3
            else:
                assert result.method == "extension_fallback"
                assert result.confidence <= 0.3

        finally:
            file_path.unlink()

    def test_magic_library_unavailable(self):
        """Test behavior when python-magic is unavailable."""
        # Create a detector and manually set magic as unavailable
        detector = FileFormatDetector()
        detector.magic_available = False

        # Should still work with other methods
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Name,Age\nAlice,25\nBob,30\n")
            f.flush()
            file_path = Path(f.name)

        try:
            result = detector.detect(file_path)
            assert result.detected_type == FileType.CSV

        finally:
            file_path.unlink()


class TestConvenienceFunctions:
    """Test the convenience functions for file detection."""

    def test_detect_file_type_function(self):
        """Test the detect_file_type convenience function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Name,Age\nAlice,25\n")
            f.flush()
            file_path = Path(f.name)

        try:
            file_type = detect_file_type(file_path)
            assert file_type == FileType.CSV

        finally:
            file_path.unlink()

    def test_detect_file_info_function(self):
        """Test the detect_file_info convenience function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Name,Age\nAlice,25\n")
            f.flush()
            file_path = Path(f.name)

        try:
            result = detect_file_info(file_path)
            assert isinstance(result, DetectionResult)
            assert result.detected_type == FileType.CSV
            assert result.confidence > 0.0

        finally:
            file_path.unlink()

    def test_nonexistent_file_handling(self):
        """Test handling of non-existent files."""
        fake_path = Path("/path/that/does/not/exist.csv")

        with pytest.raises(FileNotFoundError):
            detect_file_type(fake_path)

        with pytest.raises(FileNotFoundError):
            detect_file_info(fake_path)


class TestDetectionResult:
    """Test the DetectionResult dataclass."""

    def test_detection_result_creation(self):
        """Test creating DetectionResult instances."""
        result = DetectionResult(
            detected_type=FileType.CSV,
            confidence=0.95,
            method="content",
            mime_type="text/csv",
            encoding="utf-8",
            magic_bytes="4e616d652c416765",
            format_mismatch=False,
            extension_type=FileType.CSV,
        )

        assert result.detected_type == FileType.CSV
        assert result.confidence == 0.95
        assert result.method == "content"
        assert result.mime_type == "text/csv"
        assert result.encoding == "utf-8"
        assert result.magic_bytes == "4e616d652c416765"
        assert result.format_mismatch is False
        assert result.extension_type == FileType.CSV


class TestMagikaIntegration:
    """Test Magika AI-powered detection functionality."""

    def test_magika_availability_check(self):
        """Test Magika availability detection."""
        detector = FileFormatDetector()
        # Should not crash even if Magika is not available
        assert hasattr(detector, "magika_available")
        assert isinstance(detector.magika_available, bool)


class TestUnsupportedFormatHandling:
    """Test unsupported format detection and error handling."""

    def test_detect_file_info_safe_with_supported_format(self):
        """Test detect_file_info_safe with supported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Name,Age\nAlice,25\n")
            f.flush()
            file_path = Path(f.name)

        try:
            # Should not raise error for supported format
            result = detect_file_info_safe(file_path)
            assert result.detected_type == FileType.CSV
            assert result.is_supported is True

        finally:
            file_path.unlink()

    def test_unsupported_format_error_creation(self):
        """Test UnsupportedFormatError creation and attributes."""
        file_path = Path("/fake/file.pdf")

        error = UnsupportedFormatError("pdf", file_path, "Custom reason")
        assert error.detected_format == "pdf"
        assert error.file_path == file_path
        assert error.reason == "Custom reason"
        assert str(error) == "Custom reason"

        # Test default reason
        error2 = UnsupportedFormatError("docx", file_path)
        assert error2.reason == "Format 'docx' is not supported for spreadsheet processing"
