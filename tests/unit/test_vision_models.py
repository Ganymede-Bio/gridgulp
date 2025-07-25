"""Unit tests for vision model implementations."""

import base64
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gridporter.config import Config
from gridporter.vision.vision_models import (
    OllamaVisionModel,
    OpenAIVisionModel,
    VisionModel,
    VisionModelError,
    VisionModelResponse,
    create_vision_model,
)


class TestVisionModelResponse:
    """Test VisionModelResponse class."""

    def test_init_basic(self):
        """Test basic VisionModelResponse initialization."""
        response = VisionModelResponse(content="test content", model="test-model")

        assert response.content == "test content"
        assert response.model == "test-model"
        assert response.usage == {}

    def test_init_with_usage(self):
        """Test VisionModelResponse initialization with usage."""
        usage = {"prompt_tokens": 50, "completion_tokens": 25}
        response = VisionModelResponse(content="test content", model="test-model", usage=usage)

        assert response.usage == usage


class TestOpenAIVisionModel:
    """Test OpenAI vision model implementation."""

    def test_init_success(self):
        """Test successful OpenAI model initialization."""
        config = Config(openai_api_key="test-key", openai_model="gpt-4o")

        mock_client = MagicMock()
        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = OpenAIVisionModel(config)

        assert model.config == config
        assert model.client == mock_client
        mock_openai_module.AsyncOpenAI.assert_called_once_with(api_key="test-key")

    def test_init_no_api_key(self):
        """Test OpenAI model initialization without API key."""
        config = Config(openai_api_key=None)

        with pytest.raises(VisionModelError, match="OpenAI API key is required"):
            OpenAIVisionModel(config)

    def test_init_missing_openai_package(self):
        """Test OpenAI model initialization with missing openai package."""
        config = Config(openai_api_key="test-key")

        # Mock the import to raise ImportError
        with patch.dict("sys.modules", {"openai": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'openai'")):
                with pytest.raises(VisionModelError, match="openai package is required"):
                    OpenAIVisionModel(config)

    def test_name_property(self):
        """Test OpenAI model name property."""
        config = Config(openai_api_key="test-key", openai_model="gpt-4o")

        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = OpenAIVisionModel(config)

        assert model.name == "gpt-4o"

    def test_supports_batch_property(self):
        """Test OpenAI model supports_batch property."""
        config = Config(openai_api_key="test-key")

        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = OpenAIVisionModel(config)

        assert model.supports_batch is True

    @pytest.mark.asyncio
    async def test_analyze_image_success(self):
        """Test successful image analysis."""
        config = Config(openai_api_key="test-key", max_tokens_per_table=100)

        # Mock OpenAI client and response
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Analysis result"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 75
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = OpenAIVisionModel(config)

            image_bytes = b"fake_image_data"
            prompt = "Analyze this image"

            result = await model.analyze_image(image_bytes, prompt)

        assert isinstance(result, VisionModelResponse)
        assert result.content == "Analysis result"
        assert result.model == "gpt-4o-mini"  # Default model
        assert result.usage["total_tokens"] == 75

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"
        assert call_args[1]["max_tokens"] == 100
        assert call_args[1]["temperature"] == 0.1

        # Verify message structure
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_analyze_image_no_content(self):
        """Test image analysis with no content in response."""
        config = Config(openai_api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = OpenAIVisionModel(config)

            result = await model.analyze_image(b"image", "prompt")

        assert result.content == ""
        assert result.usage["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_analyze_image_api_error(self):
        """Test image analysis with API error."""
        config = Config(openai_api_key="test-key")

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API rate limit")

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = OpenAIVisionModel(config)

            with pytest.raises(VisionModelError, match="OpenAI analysis failed: API rate limit"):
                await model.analyze_image(b"image", "prompt")

    @pytest.mark.asyncio
    async def test_analyze_image_base64_encoding(self):
        """Test that image is properly base64 encoded."""
        config = Config(openai_api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = OpenAIVisionModel(config)

            image_bytes = b"test_image_data"
            await model.analyze_image(image_bytes, "prompt")

        # Verify base64 encoding
        call_args = mock_client.chat.completions.create.call_args
        image_url = call_args[1]["messages"][0]["content"][1]["image_url"]["url"]
        expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
        assert image_url == f"data:image/png;base64,{expected_b64}"


class TestOllamaVisionModel:
    """Test Ollama vision model implementation."""

    def test_init(self):
        """Test Ollama model initialization."""
        config = Config(ollama_url="http://localhost:11434", ollama_vision_model="qwen2-vl:7b")

        model = OllamaVisionModel(config)

        assert model.config == config
        assert model.base_url == "http://localhost:11434"
        assert model.model_name == "qwen2-vl:7b"

    def test_init_strips_trailing_slash(self):
        """Test Ollama model initialization strips trailing slash from URL."""
        config = Config(ollama_url="http://localhost:11434/")

        model = OllamaVisionModel(config)

        assert model.base_url == "http://localhost:11434"

    def test_name_property(self):
        """Test Ollama model name property."""
        config = Config(ollama_vision_model="custom-model:latest")

        model = OllamaVisionModel(config)

        assert model.name == "custom-model:latest"

    def test_supports_batch_property(self):
        """Test Ollama model supports_batch property."""
        config = Config()

        model = OllamaVisionModel(config)

        assert model.supports_batch is False

    @pytest.mark.asyncio
    async def test_analyze_image_success(self):
        """Test successful Ollama image analysis."""
        config = Config(ollama_vision_model="qwen2-vl:7b", max_tokens_per_table=200)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Analysis result",
            "eval_duration": 1500000000,  # 1.5 seconds in nanoseconds
            "total_duration": 2000000000,  # 2 seconds in nanoseconds
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            model = OllamaVisionModel(config)

            image_bytes = b"fake_image_data"
            prompt = "Analyze this image"

            result = await model.analyze_image(image_bytes, prompt)

        assert isinstance(result, VisionModelResponse)
        assert result.content == "Analysis result"
        assert result.model == "qwen2-vl:7b"
        assert result.usage["eval_duration"] == 1500000000
        assert result.usage["total_duration"] == 2000000000

        # Verify API call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"

        # Verify payload
        payload = call_args[1]["json"]
        assert payload["model"] == "qwen2-vl:7b"
        assert payload["prompt"] == prompt
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.1
        assert payload["options"]["num_predict"] == 200
        assert len(payload["images"]) == 1

    @pytest.mark.asyncio
    async def test_analyze_image_http_error(self):
        """Test Ollama image analysis with HTTP error."""
        config = Config()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client

            model = OllamaVisionModel(config)

            with pytest.raises(VisionModelError, match="Ollama analysis failed: Connection failed"):
                await model.analyze_image(b"image", "prompt")

    @pytest.mark.asyncio
    async def test_analyze_image_no_response_content(self):
        """Test Ollama analysis with no response content."""
        config = Config()

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            model = OllamaVisionModel(config)

            result = await model.analyze_image(b"image", "prompt")

        assert result.content == ""

    @pytest.mark.asyncio
    async def test_check_model_available_success(self):
        """Test checking model availability successfully."""
        config = Config(ollama_vision_model="qwen2-vl:7b")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "qwen2-vl:7b"}, {"name": "llama2:latest"}]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            model = OllamaVisionModel(config)

            available = await model.check_model_available()

        assert available is True
        mock_client.get.assert_called_once_with("http://localhost:11434/api/tags")

    @pytest.mark.asyncio
    async def test_check_model_available_not_found(self):
        """Test checking model availability when model not found."""
        config = Config(ollama_vision_model="nonexistent:latest")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "qwen2-vl:7b"}, {"name": "llama2:latest"}]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            model = OllamaVisionModel(config)

            available = await model.check_model_available()

        assert available is False

    @pytest.mark.asyncio
    async def test_check_model_available_with_tag(self):
        """Test checking model availability with version tag."""
        config = Config(ollama_vision_model="qwen2-vl:latest")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "qwen2-vl:7b"}, {"name": "qwen2-vl:latest"}]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            model = OllamaVisionModel(config)

            available = await model.check_model_available()

        assert available is True

    @pytest.mark.asyncio
    async def test_check_model_available_connection_error(self):
        """Test checking model availability with connection error."""
        config = Config()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value = mock_client

            model = OllamaVisionModel(config)

            available = await model.check_model_available()

        assert available is False


class TestCreateVisionModel:
    """Test vision model factory function."""

    def test_create_openai_model(self):
        """Test creating OpenAI model via factory."""
        config = Config(use_local_llm=False, openai_api_key="test-key")

        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = create_vision_model(config)

        assert isinstance(model, OpenAIVisionModel)

    def test_create_ollama_model(self):
        """Test creating Ollama model via factory."""
        config = Config(use_local_llm=True)

        model = create_vision_model(config)

        assert isinstance(model, OllamaVisionModel)

    def test_create_model_no_provider(self):
        """Test creating model with no available provider."""
        config = Config(use_local_llm=False, openai_api_key=None)

        with pytest.raises(VisionModelError, match="No vision model available"):
            create_vision_model(config)

    def test_create_model_prefers_openai_when_available(self):
        """Test that factory prefers OpenAI when API key is available."""
        config = Config(use_local_llm=False, openai_api_key="test-key")

        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            model = create_vision_model(config)

        assert isinstance(model, OpenAIVisionModel)

    def test_create_model_uses_ollama_when_local_preferred(self):
        """Test that factory uses Ollama when local LLM is preferred."""
        config = Config(
            use_local_llm=True,
            openai_api_key="test-key",  # Has key but prefers local
        )

        model = create_vision_model(config)

        assert isinstance(model, OllamaVisionModel)
