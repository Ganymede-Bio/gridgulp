"""Vision model implementations for table detection."""

import base64
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..config import GridPorterConfig
from ..utils.openai_pricing import get_pricing_instance

logger = logging.getLogger(__name__)


class VisionModelError(Exception):
    """Base exception for vision model errors."""

    pass


class VisionModelResponse:
    """Standardized response from vision models."""

    def __init__(self, content: str, model: str, usage: dict[str, Any] | None = None):
        """Initialize vision model response.

        Args:
            content: The text response from the model
            model: Model name/identifier
            usage: Token/cost usage information
        """
        self.content = content
        self.model = model
        self.usage = usage or {}


class VisionModel(ABC):
    """Abstract base class for vision models."""

    def __init__(self, config: GridPorterConfig):
        """Initialize vision model.

        Args:
            config: GridPorter configuration
        """
        self.config = config

    @abstractmethod
    async def analyze_image(self, image_bytes: bytes, prompt: str) -> VisionModelResponse:
        """Analyze an image with the given prompt.

        Args:
            image_bytes: PNG image data
            prompt: Text prompt describing what to analyze

        Returns:
            Model response

        Raises:
            VisionModelError: If analysis fails
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def supports_batch(self) -> bool:
        """Check if model supports batch processing."""
        pass


class OpenAIVisionModel(VisionModel):
    """OpenAI GPT-4 Vision model implementation."""

    def __init__(self, config: GridPorterConfig):
        """Initialize OpenAI vision model."""
        super().__init__(config)
        if not config.openai_api_key:
            raise VisionModelError("OpenAI API key is required for OpenAI vision model")

        # Import openai here to avoid dependency if not using OpenAI
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=config.openai_api_key)
        except ImportError as e:
            raise VisionModelError("openai package is required for OpenAI vision model") from e

    @property
    def name(self) -> str:
        """Get model name."""
        return self.config.openai_model

    @property
    def supports_batch(self) -> bool:
        """OpenAI supports analyzing multiple images in one request."""
        return True

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> VisionModelResponse:
        """Analyze image using OpenAI GPT-4 Vision.

        Args:
            image_bytes: PNG image data
            prompt: Analysis prompt

        Returns:
            Model response
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ]

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.name,
                messages=messages,
                max_tokens=self.config.max_tokens_per_table,
                temperature=0.1,
            )

            # Extract response
            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # Calculate cost using pricing module
            pricing = get_pricing_instance()
            cost = pricing.calculate_cost(
                model_id=self.name,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
            )
            usage["cost_usd"] = cost

            logger.info(
                f"OpenAI vision analysis completed. "
                f"Tokens: {usage['total_tokens']} "
                f"Cost: ${cost:.6f}"
            )

            return VisionModelResponse(content=content, model=self.name, usage=usage)

        except Exception as e:
            logger.error(f"OpenAI vision analysis failed: {e}")
            raise VisionModelError(f"OpenAI analysis failed: {str(e)}") from e


class OllamaVisionModel(VisionModel):
    """Ollama local vision model implementation."""

    def __init__(self, config: GridPorterConfig):
        """Initialize Ollama vision model."""
        super().__init__(config)
        self.base_url = config.ollama_url.rstrip("/")
        self.model_name = config.ollama_vision_model

    @property
    def name(self) -> str:
        """Get model name."""
        return self.model_name

    @property
    def supports_batch(self) -> bool:
        """Ollama processes one image at a time."""
        return False

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> VisionModelResponse:
        """Analyze image using Ollama vision model.

        Args:
            image_bytes: PNG image data
            prompt: Analysis prompt

        Returns:
            Model response
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Prepare request
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": self.config.max_tokens_per_table},
            }

            # Call Ollama API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()

            result = response.json()
            content = result.get("response", "")

            # Ollama doesn't provide detailed token usage
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "eval_duration": result.get("eval_duration", 0),
                "total_duration": result.get("total_duration", 0),
            }

            logger.info(f"Ollama vision analysis completed in {usage['total_duration'] / 1e9:.2f}s")

            return VisionModelResponse(content=content, model=self.name, usage=usage)

        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise VisionModelError(f"Ollama connection failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Ollama vision analysis failed: {e}")
            raise VisionModelError(f"Ollama analysis failed: {str(e)}") from e

    async def check_model_available(self) -> bool:
        """Check if the specified model is available in Ollama.

        Returns:
            True if model is available
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()

            models = response.json().get("models", [])
            available_names = [m.get("name", "").split(":")[0] for m in models]
            model_base = self.model_name.split(":")[0]

            return model_base in available_names

        except Exception as e:
            logger.warning(f"Failed to check Ollama model availability: {e}")
            return False


def create_vision_model(config: GridPorterConfig) -> VisionModel:
    """Factory function to create appropriate vision model.

    Args:
        config: GridPorter configuration

    Returns:
        Vision model instance

    Raises:
        VisionModelError: If no suitable model can be created
    """
    if config.use_local_llm:
        return OllamaVisionModel(config)
    elif config.openai_api_key:
        return OpenAIVisionModel(config)
    else:
        raise VisionModelError(
            "No vision model available. Either set use_local_llm=True for Ollama "
            "or provide an OpenAI API key."
        )
