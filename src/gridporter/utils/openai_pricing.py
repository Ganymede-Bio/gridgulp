"""OpenAI pricing module with hardcoded pricing table.

This module provides pricing information for OpenAI models and calculates
costs based on token usage. Pricing is based on OpenAI's official rates
as of January 2025.

Note: Pricing should be manually updated when OpenAI changes their rates.
Last updated: January 2025
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OpenAIModelPricing:
    """Pricing information for an OpenAI model."""

    model_id: str
    input_cost_per_million: float  # USD per 1M input tokens
    output_cost_per_million: float  # USD per 1M output tokens
    image_cost_per_million: float = 0.0  # USD per 1M image tokens (vision models)
    context_length: int = 0
    supports_vision: bool = False

    @property
    def input_cost_per_token(self) -> float:
        """Cost per single input token."""
        return self.input_cost_per_million / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        """Cost per single output token."""
        return self.output_cost_per_million / 1_000_000

    @property
    def image_cost_per_token(self) -> float:
        """Cost per single image token."""
        return self.image_cost_per_million / 1_000_000


# OpenAI Model Pricing Table (as of January 2025)
# Source: https://openai.com/api/pricing/
OPENAI_PRICING = {
    # GPT-4o models
    "gpt-4o": OpenAIModelPricing(
        model_id="gpt-4o",
        input_cost_per_million=5.00,
        output_cost_per_million=15.00,
        context_length=128_000,
        supports_vision=True,
    ),
    "gpt-4o-2024-11-20": OpenAIModelPricing(
        model_id="gpt-4o-2024-11-20",
        input_cost_per_million=5.00,
        output_cost_per_million=15.00,
        context_length=128_000,
        supports_vision=True,
    ),
    "gpt-4o-2024-08-06": OpenAIModelPricing(
        model_id="gpt-4o-2024-08-06",
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
        context_length=128_000,
        supports_vision=True,
    ),
    "gpt-4o-2024-05-13": OpenAIModelPricing(
        model_id="gpt-4o-2024-05-13",
        input_cost_per_million=5.00,
        output_cost_per_million=15.00,
        context_length=128_000,
        supports_vision=True,
    ),
    # GPT-4o-mini models
    "gpt-4o-mini": OpenAIModelPricing(
        model_id="gpt-4o-mini",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        context_length=128_000,
        supports_vision=True,
    ),
    "gpt-4o-mini-2024-07-18": OpenAIModelPricing(
        model_id="gpt-4o-mini-2024-07-18",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        context_length=128_000,
        supports_vision=True,
    ),
    # GPT-4 Turbo models
    "gpt-4-turbo": OpenAIModelPricing(
        model_id="gpt-4-turbo",
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
        context_length=128_000,
        supports_vision=True,
    ),
    "gpt-4-turbo-2024-04-09": OpenAIModelPricing(
        model_id="gpt-4-turbo-2024-04-09",
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
        context_length=128_000,
        supports_vision=True,
    ),
    "gpt-4-turbo-preview": OpenAIModelPricing(
        model_id="gpt-4-turbo-preview",
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
        context_length=128_000,
        supports_vision=False,
    ),
    "gpt-4-0125-preview": OpenAIModelPricing(
        model_id="gpt-4-0125-preview",
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
        context_length=128_000,
        supports_vision=False,
    ),
    "gpt-4-1106-preview": OpenAIModelPricing(
        model_id="gpt-4-1106-preview",
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
        context_length=128_000,
        supports_vision=False,
    ),
    # GPT-4 models
    "gpt-4": OpenAIModelPricing(
        model_id="gpt-4",
        input_cost_per_million=30.00,
        output_cost_per_million=60.00,
        context_length=8_192,
        supports_vision=False,
    ),
    "gpt-4-0613": OpenAIModelPricing(
        model_id="gpt-4-0613",
        input_cost_per_million=30.00,
        output_cost_per_million=60.00,
        context_length=8_192,
        supports_vision=False,
    ),
    "gpt-4-32k": OpenAIModelPricing(
        model_id="gpt-4-32k",
        input_cost_per_million=60.00,
        output_cost_per_million=120.00,
        context_length=32_768,
        supports_vision=False,
    ),
    "gpt-4-32k-0613": OpenAIModelPricing(
        model_id="gpt-4-32k-0613",
        input_cost_per_million=60.00,
        output_cost_per_million=120.00,
        context_length=32_768,
        supports_vision=False,
    ),
    # GPT-3.5 Turbo models
    "gpt-3.5-turbo": OpenAIModelPricing(
        model_id="gpt-3.5-turbo",
        input_cost_per_million=0.50,
        output_cost_per_million=1.50,
        context_length=16_385,
        supports_vision=False,
    ),
    "gpt-3.5-turbo-0125": OpenAIModelPricing(
        model_id="gpt-3.5-turbo-0125",
        input_cost_per_million=0.50,
        output_cost_per_million=1.50,
        context_length=16_385,
        supports_vision=False,
    ),
    "gpt-3.5-turbo-1106": OpenAIModelPricing(
        model_id="gpt-3.5-turbo-1106",
        input_cost_per_million=1.00,
        output_cost_per_million=2.00,
        context_length=16_385,
        supports_vision=False,
    ),
    # o1 models (reasoning models)
    "o1": OpenAIModelPricing(
        model_id="o1",
        input_cost_per_million=15.00,
        output_cost_per_million=60.00,
        context_length=200_000,
        supports_vision=True,
    ),
    "o1-2024-12-17": OpenAIModelPricing(
        model_id="o1-2024-12-17",
        input_cost_per_million=15.00,
        output_cost_per_million=60.00,
        context_length=200_000,
        supports_vision=True,
    ),
    "o1-preview": OpenAIModelPricing(
        model_id="o1-preview",
        input_cost_per_million=15.00,
        output_cost_per_million=60.00,
        context_length=128_000,
        supports_vision=False,
    ),
    "o1-preview-2024-09-12": OpenAIModelPricing(
        model_id="o1-preview-2024-09-12",
        input_cost_per_million=15.00,
        output_cost_per_million=60.00,
        context_length=128_000,
        supports_vision=False,
    ),
    "o1-mini": OpenAIModelPricing(
        model_id="o1-mini",
        input_cost_per_million=3.00,
        output_cost_per_million=12.00,
        context_length=128_000,
        supports_vision=False,
    ),
    "o1-mini-2024-09-12": OpenAIModelPricing(
        model_id="o1-mini-2024-09-12",
        input_cost_per_million=3.00,
        output_cost_per_million=12.00,
        context_length=128_000,
        supports_vision=False,
    ),
}


class OpenAIPricing:
    """Manages OpenAI model pricing and cost calculations."""

    def __init__(self):
        """Initialize the pricing manager."""
        self.pricing = OPENAI_PRICING.copy()
        self.logger = logger
        self.last_pricing_update = datetime(2025, 1, 1)  # Last manual update

    def get_model_pricing(self, model_id: str) -> OpenAIModelPricing | None:
        """Get pricing for a specific model.

        Args:
            model_id: OpenAI model ID

        Returns:
            Model pricing or None if not found
        """
        return self.pricing.get(model_id)

    def calculate_cost(
        self,
        model_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> float:
        """Calculate cost for token usage.

        Args:
            model_id: OpenAI model ID
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            total_tokens: Total tokens (optional, for backwards compatibility)

        Returns:
            Total cost in USD
        """
        pricing = self.get_model_pricing(model_id)
        if not pricing:
            self.logger.warning(
                f"No pricing found for model {model_id}, using GPT-4o-mini as fallback"
            )
            pricing = self.pricing.get("gpt-4o-mini")
            if not pricing:
                return 0.0

        # Handle backwards compatibility
        if total_tokens is not None and prompt_tokens == 0 and completion_tokens == 0:
            # Estimate 60/40 split for prompt/completion
            prompt_tokens = int(total_tokens * 0.6)
            completion_tokens = total_tokens - prompt_tokens

        input_cost = pricing.input_cost_per_token * prompt_tokens
        output_cost = pricing.output_cost_per_token * completion_tokens

        return input_cost + output_cost

    def get_cheapest_vision_model(self) -> tuple[str, OpenAIModelPricing] | None:
        """Get the cheapest model that supports vision.

        Returns:
            Tuple of (model_id, pricing) or None
        """
        vision_models = [
            (model_id, pricing)
            for model_id, pricing in self.pricing.items()
            if pricing.supports_vision
        ]

        if not vision_models:
            return None

        # Sort by average cost (input + output) / 2
        vision_models.sort(
            key=lambda x: (x[1].input_cost_per_million + x[1].output_cost_per_million) / 2
        )

        return vision_models[0]

    def format_cost_breakdown(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> str:
        """Format a detailed cost breakdown.

        Args:
            model_id: OpenAI model ID
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Formatted cost breakdown string
        """
        pricing = self.get_model_pricing(model_id)
        if not pricing:
            return f"No pricing available for {model_id}"

        input_cost = pricing.input_cost_per_token * prompt_tokens
        output_cost = pricing.output_cost_per_token * completion_tokens
        total_cost = input_cost + output_cost

        breakdown = [
            f"Model: {model_id}",
            f"Input: {prompt_tokens:,} tokens × ${pricing.input_cost_per_million:.2f}/M = ${input_cost:.6f}",
            f"Output: {completion_tokens:,} tokens × ${pricing.output_cost_per_million:.2f}/M = ${output_cost:.6f}",
            f"Total: ${total_cost:.6f}",
        ]

        return "\n".join(breakdown)

    def check_pricing_age(self) -> dict[str, Any]:
        """Check how old the pricing data is.

        Returns:
            Dictionary with pricing age information
        """
        age = datetime.now() - self.last_pricing_update
        return {
            "last_updated": self.last_pricing_update.isoformat(),
            "days_old": age.days,
            "needs_update": age.days > 30,
            "message": (
                f"Pricing data is {age.days} days old. "
                + (
                    "Consider checking OpenAI's pricing page for updates."
                    if age.days > 30
                    else "Pricing is up to date."
                )
            ),
        }


class OpenAICostsAPI:
    """Client for OpenAI Costs API (requires admin key)."""

    COSTS_ENDPOINT = "https://api.openai.com/v1/organization/costs"

    def __init__(self, admin_api_key: str | None = None):
        """Initialize the Costs API client.

        Args:
            admin_api_key: OpenAI admin API key (required for costs endpoint)
        """
        self.admin_api_key = admin_api_key
        self.logger = logger

    async def get_costs(
        self,
        start_time: int,
        end_time: int | None = None,
        bucket_width: str = "1d",
        limit: int = 30,
    ) -> dict[str, Any] | None:
        """Get cost data from OpenAI Costs API.

        Args:
            start_time: Start time in Unix seconds
            end_time: End time in Unix seconds (optional)
            bucket_width: Time bucket width (currently only "1d" supported)
            limit: Number of buckets to return

        Returns:
            Cost data or None if request fails
        """
        if not self.admin_api_key:
            self.logger.warning("No admin API key provided for Costs API")
            return None

        headers = {
            "Authorization": f"Bearer {self.admin_api_key}",
            "Content-Type": "application/json",
        }

        params = {
            "start_time": start_time,
            "bucket_width": bucket_width,
            "limit": limit,
        }

        if end_time:
            params["end_time"] = end_time

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self.COSTS_ENDPOINT,
                    headers=headers,
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            self.logger.error(f"Costs API error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get costs data: {e}")
            return None


# Global instance for convenience
_pricing_instance: OpenAIPricing | None = None


def get_pricing_instance() -> OpenAIPricing:
    """Get or create the global pricing instance."""
    global _pricing_instance
    if _pricing_instance is None:
        _pricing_instance = OpenAIPricing()
    return _pricing_instance
