"""Build vision requests with multi-scale bitmap strategy."""

import base64
import logging
import time

from ..models.multi_scale import (
    CompressionLevel,
    DataRegion,
    MultiScaleBitmaps,
    VisionImage,
    VisionRequest,
)
from ..models.sheet_data import SheetData
from ..utils.excel_utils import to_excel_range
from .bitmap_generator import BitmapGenerator
from .data_region_preprocessor import DataRegionPreprocessor
from .progressive_refiner import ProgressiveRefiner

logger = logging.getLogger(__name__)


class VisionRequestBuilder:
    """Build optimized vision requests for table detection.

    This builder creates multi-scale bitmap representations with explicit
    compression descriptions for the vision model.
    """

    # Vision API constraints
    MAX_IMAGE_SIZE_MB = 20.0
    MAX_TOTAL_SIZE_MB = 20.0
    MAX_IMAGES_PER_REQUEST = 10

    # Size thresholds for compression strategy
    SINGLE_IMAGE_THRESHOLD = 100_000  # cells
    MULTI_SCALE_THRESHOLD = 1_000_000  # cells

    def __init__(self):
        """Initialize the vision request builder."""
        self.preprocessor = DataRegionPreprocessor()
        self.bitmap_gen = BitmapGenerator(auto_compress=True)
        self.progressive_refiner = ProgressiveRefiner()

    def build_request(self, sheet_data: SheetData, sheet_name: str) -> VisionRequest:
        """Build a vision request for the given sheet.

        Args:
            sheet_data: The sheet data to analyze
            sheet_name: Name of the sheet

        Returns:
            VisionRequest ready for the vision model
        """
        start_time = time.time()

        # Detect data regions
        data_regions = self.preprocessor.detect_data_regions(sheet_data)

        if not data_regions:
            logger.info(f"No data regions found in sheet '{sheet_name}'")
            return self._build_empty_request()

        # Calculate total cells
        total_cells = sum(region.total_cells for region in data_regions)

        # Choose strategy based on size
        if total_cells <= self.SINGLE_IMAGE_THRESHOLD:
            strategy = "single_image"
            images = self._build_single_image_strategy(sheet_data, data_regions)
        elif total_cells <= self.MULTI_SCALE_THRESHOLD:
            strategy = "multi_scale"
            images = self._build_multi_scale_strategy(sheet_data, data_regions)
        else:
            strategy = "progressive"
            images = self._build_progressive_strategy(sheet_data, data_regions)

        # Build multi-scale bitmap collection
        bitmaps = MultiScaleBitmaps(
            sheet_name=sheet_name,
            sheet_dimensions={"rows": sheet_data.max_row + 1, "cols": sheet_data.max_column + 1},
            data_bounds=self._calculate_data_bounds(data_regions),
            compression_strategy=strategy,
            total_size_mb=sum(img.size_mb for img in images),
            generation_time_ms=int((time.time() - start_time) * 1000),
        )

        # Add images to collection
        for image in images:
            bitmaps.add_image(image)

        # Select prompt template
        if strategy == "single_image":
            prompt_template = "SINGLE_IMAGE"
        elif strategy == "progressive":
            prompt_template = "PROGRESSIVE"
        else:
            prompt_template = "EXPLICIT_MULTI_SCALE"

        # Create vision request
        request = VisionRequest(
            images=images,
            prompt_template=prompt_template,
            total_images=len(images),
            total_size_mb=bitmaps.total_size_mb,
        )

        # Validate size limits
        if not request.validate_size_limit(self.MAX_TOTAL_SIZE_MB):
            logger.warning(
                f"Request exceeds size limit ({request.total_size_mb:.1f}MB > "
                f"{self.MAX_TOTAL_SIZE_MB}MB), applying aggressive compression"
            )
            # Regenerate with more aggressive compression
            images = self._build_aggressive_compression(sheet_data, data_regions)
            request.images = images
            request.total_images = len(images)
            request.total_size_mb = sum(img.size_mb for img in images)

        logger.info(
            f"Built {strategy} vision request with {len(images)} images "
            f"({request.total_size_mb:.1f}MB) for sheet '{sheet_name}'"
        )

        return request

    def _build_empty_request(self) -> VisionRequest:
        """Build request for empty sheet."""
        return VisionRequest(
            images=[], prompt_template="SINGLE_IMAGE", total_images=0, total_size_mb=0.0
        )

    def _build_single_image_strategy(
        self, sheet_data: SheetData, regions: list[DataRegion]
    ) -> list[VisionImage]:
        """Build single image for small sheets."""
        # Use the overall data bounds
        bounds = self._calculate_data_bounds(regions)

        # Generate full resolution bitmap
        img_data, metadata = self.bitmap_gen.generate(sheet_data)

        # Create vision image
        image = VisionImage(
            image_id="full_sheet",
            image_data=base64.b64encode(img_data).decode("utf-8"),
            compression_level=metadata.compression_level.value if metadata.compression_level else 0,
            block_size=metadata.block_size or [1, 1],
            description=(
                f"Full sheet view with {metadata.total_rows}×{metadata.total_cols} cells. "
                f"No compression applied - each pixel represents exactly one cell."
            ),
            purpose="Identify all table boundaries in this sheet",
            covers_cells=to_excel_range(0, 0, bounds["bottom"], bounds["right"]),
            size_bytes=metadata.file_size_bytes,
        )

        return [image]

    def _build_multi_scale_strategy(
        self, sheet_data: SheetData, regions: list[DataRegion]
    ) -> list[VisionImage]:
        """Build multi-scale images for medium sheets."""
        images = []

        # 1. Overview with appropriate compression
        overview_level = self._select_overview_compression(sheet_data)
        overview_data, overview_meta = self.bitmap_gen.generate(sheet_data)

        overview = VisionImage(
            image_id="overview",
            image_data=base64.b64encode(overview_data).decode("utf-8"),
            compression_level=overview_level.value,
            block_size=[overview_level.row_block, overview_level.col_block],
            description=(
                f"Overview of entire sheet compressed at {overview_level.description}. "
                f"Each pixel represents {overview_level.row_block}×{overview_level.col_block} cells. "
                f"Use this to identify general table locations."
            ),
            purpose="Identify approximate table regions",
            covers_cells=to_excel_range(0, 0, sheet_data.max_row, sheet_data.max_column),
            size_bytes=overview_meta.file_size_bytes,
        )
        images.append(overview)

        # 2. Detail views for each data region
        for idx, region in enumerate(regions):
            if region.skip:
                continue

            # Generate detail view at full resolution
            from ..vision.quadtree import QuadTreeBounds

            bounds = QuadTreeBounds(
                min_row=region.bounds["top"],
                min_col=region.bounds["left"],
                max_row=region.bounds["bottom"],
                max_col=region.bounds["right"],
            )

            detail_data, detail_meta = self.bitmap_gen.generate(sheet_data, bounds)

            detail = VisionImage(
                image_id=f"detail_{idx + 1}",
                image_data=base64.b64encode(detail_data).decode("utf-8"),
                compression_level=detail_meta.compression_level.value
                if detail_meta.compression_level
                else 0,
                block_size=detail_meta.block_size or [1, 1],
                description=(
                    f"Detail view of region {idx + 1} "
                    f"({region.rows}×{region.cols} cells). "
                    f"{'No compression - full detail visible.' if detail_meta.compression_level == CompressionLevel.NONE else f'Compressed at {detail_meta.compression_level.description}.'}"
                ),
                purpose=f"Identify exact table boundaries in region {idx + 1}",
                covers_cells=to_excel_range(
                    region.bounds["top"],
                    region.bounds["left"],
                    region.bounds["bottom"],
                    region.bounds["right"],
                ),
                size_bytes=detail_meta.file_size_bytes,
            )
            images.append(detail)

            # Limit number of detail views
            if len(images) >= self.MAX_IMAGES_PER_REQUEST:
                logger.warning(
                    f"Reached image limit ({self.MAX_IMAGES_PER_REQUEST}), "
                    f"skipping {len(regions) - idx - 1} remaining regions"
                )
                break

        return images

    def _build_progressive_strategy(
        self, sheet_data: SheetData, regions: list[DataRegion]
    ) -> list[VisionImage]:
        """Build progressive refinement for very large sheets."""
        # Use ProgressiveRefiner for iterative refinement
        images, phases = self.progressive_refiner.refine_sheet(sheet_data, regions)

        # Convert hex-encoded images to base64
        vision_images = []
        for img in images:
            # Convert hex back to bytes then to base64
            img_bytes = bytes.fromhex(img.image_data)
            img.image_data = base64.b64encode(img_bytes).decode("utf-8")
            vision_images.append(img)

        logger.info(
            f"Progressive refinement completed {len(phases)} phases with {len(vision_images)} images"
        )

        return vision_images

    def _build_aggressive_compression(
        self, sheet_data: SheetData, regions: list[DataRegion]
    ) -> list[VisionImage]:
        """Build images with aggressive compression to fit size limits."""
        images = []

        # Use maximum compression for overview
        self.bitmap_gen.auto_compress = False
        self.bitmap_gen.compression_level = 9

        # Generate highly compressed overview
        overview_data, overview_meta = self.bitmap_gen.generate(sheet_data)

        # Force maximum compression level in metadata
        overview_meta.compression_level = CompressionLevel.MAXIMUM
        overview_meta.block_size = [
            CompressionLevel.MAXIMUM.row_block,
            CompressionLevel.MAXIMUM.col_block,
        ]

        overview = VisionImage(
            image_id="overview_compressed",
            image_data=base64.b64encode(overview_data).decode("utf-8"),
            compression_level=CompressionLevel.MAXIMUM.value,
            block_size=overview_meta.block_size,
            description=(
                f"Highly compressed overview using {CompressionLevel.MAXIMUM.description}. "
                f"Each pixel represents {overview_meta.block_size[0]}×{overview_meta.block_size[1]} cells. "
                f"Focus on identifying major table regions only."
            ),
            purpose="Identify major table regions despite high compression",
            covers_cells=to_excel_range(0, 0, sheet_data.max_row, sheet_data.max_column),
            size_bytes=overview_meta.file_size_bytes,
        )
        images.append(overview)

        # Reset bitmap generator
        self.bitmap_gen.auto_compress = True
        self.bitmap_gen.compression_level = 6

        return images

    def _select_overview_compression(self, sheet_data: SheetData) -> CompressionLevel:
        """Select appropriate compression level for overview."""
        total_cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)

        # Use compression level thresholds
        for level in CompressionLevel:
            if total_cells <= level.max_cells:
                return level

        return CompressionLevel.MAXIMUM

    def _calculate_data_bounds(self, regions: list[DataRegion]) -> dict[str, int]:
        """Calculate overall bounds of all data regions."""
        if not regions:
            return {"top": 0, "left": 0, "bottom": 0, "right": 0}

        return {
            "top": min(r.bounds["top"] for r in regions),
            "left": min(r.bounds["left"] for r in regions),
            "bottom": max(r.bounds["bottom"] for r in regions),
            "right": max(r.bounds["right"] for r in regions),
        }

    def create_explicit_prompt(self, request: VisionRequest) -> str:
        """Create explicit prompt for vision model based on request type.

        Args:
            request: The vision request

        Returns:
            Detailed prompt for the vision model
        """
        if request.prompt_template == "SINGLE_IMAGE":
            return self._create_single_image_prompt(request)
        elif request.prompt_template == "EXPLICIT_MULTI_SCALE":
            return self._create_multi_scale_prompt(request)
        else:
            return self._create_progressive_prompt(request)

    def _create_single_image_prompt(self, request: VisionRequest) -> str:
        """Create prompt for single image analysis."""
        image = request.images[0] if request.images else None
        if not image:
            return "No image provided."

        return f"""Analyze this spreadsheet image to identify all table regions.

Image details:
- Coverage: {image.covers_cells}
- Compression: {image.description}

Please identify ALL distinct table regions by:
1. Looking for rectangular blocks of filled cells
2. Identifying natural boundaries (empty rows/columns)
3. Detecting header patterns (bold/formatted cells at top)
4. Considering logical data groupings

For each table found, provide:
- Exact cell range (e.g., "A1:E20")
- Confidence score (0-1)
- Table characteristics (has headers, data types, etc.)
- Suggested table name based on content

Be precise with boundaries - include all data cells but exclude unnecessary empty cells."""

    def _create_multi_scale_prompt(self, request: VisionRequest) -> str:
        """Create prompt for multi-scale analysis."""
        overview = next((img for img in request.images if img.image_id == "overview"), None)
        detail_count = len([img for img in request.images if img.image_id.startswith("detail_")])

        prompt = f"""Analyze these {len(request.images)} images showing a spreadsheet at multiple scales.

Images provided:
1. Overview image: {overview.description if overview else 'Not provided'}
   - Purpose: {overview.purpose if overview else 'N/A'}
   - Coverage: {overview.covers_cells if overview else 'N/A'}

"""

        # Add detail image descriptions
        for idx, img in enumerate(request.images):
            if img.image_id.startswith("detail_"):
                prompt += f"""{idx + 1}. Detail image {img.image_id}: {img.description}
   - Purpose: {img.purpose}
   - Coverage: {img.covers_cells}

"""

        prompt += """Analysis approach:
1. First examine the overview to understand overall structure
2. Use detail images to refine exact table boundaries
3. Ensure no tables are missed between detail views
4. Account for any compression artifacts

For each table found, provide:
- Exact cell range in original sheet coordinates
- Which image(s) showed this table
- Confidence score (0-1)
- Table characteristics
- Suggested name

Note: Coordinates should be in the original sheet space, not image pixels."""

        return prompt

    def _create_progressive_prompt(self, request: VisionRequest) -> str:
        """Create prompt for progressive refinement."""
        return f"""Analyze this highly compressed spreadsheet overview.

Due to the sheet's large size ({request.images[0].covers_cells if request.images else 'Unknown'}),
extreme compression was applied: {request.images[0].description if request.images else 'No description'}

Despite compression, identify:
1. Major table regions (approximate boundaries are acceptable)
2. Areas that likely contain structured data
3. Regions that would benefit from detailed analysis

For each identified region:
- Approximate cell range
- Confidence (will be lower due to compression)
- Why this appears to be a table
- Priority for detailed analysis (high/medium/low)

Focus on finding all potential tables rather than precise boundaries."""
