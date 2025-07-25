"""LLM usage tracking using OpenTelemetry."""

import time
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from opentelemetry.trace import Status, StatusCode


class LLMTracker:
    """Track LLM calls using OpenTelemetry spans."""

    def __init__(self, service_name: str = "gridporter", exporter: SpanExporter | None = None):
        """Initialize LLM tracker.

        Args:
            service_name: Name of the service for telemetry
            exporter: Optional custom span exporter (defaults to console)
        """
        # Create resource
        resource = Resource.create({"service.name": service_name})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add span processor with exporter
        if exporter is None:
            exporter = ConsoleSpanExporter()

        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        # Instrument OpenAI
        OpenAIInstrumentor().instrument()

        # Track totals
        self._total_calls = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    @contextmanager
    def track_call(self, model: str, operation: str = "chat_completion", **kwargs: Any):
        """Track an LLM call as a span.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3")
            operation: Type of operation
            **kwargs: Additional attributes to add to span

        Yields:
            Span object for adding attributes
        """
        with self.tracer.start_as_current_span(f"llm.{operation}") as span:
            # Set standard attributes
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.operation", operation)

            # Add any additional attributes
            for key, value in kwargs.items():
                span.set_attribute(f"llm.{key}", value)

            # Track call count
            self._total_calls += 1
            span.set_attribute("llm.total_calls", self._total_calls)

            start_time = time.time()

            try:
                yield span

                # Mark as successful
                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                # Mark as failed
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            finally:
                # Record duration
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)

    def record_tokens(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int | None = None
    ):
        """Record token usage on the current span.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (computed if not provided)
        """
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        # Update totals
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_tokens += total_tokens

        # Add to current span if active
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("llm.prompt_tokens", prompt_tokens)
            span.set_attribute("llm.completion_tokens", completion_tokens)
            span.set_attribute("llm.total_tokens", total_tokens)

            # Add running totals
            span.set_attribute("llm.total_prompt_tokens", self._total_prompt_tokens)
            span.set_attribute("llm.total_completion_tokens", self._total_completion_tokens)
            span.set_attribute("llm.total_tokens_all", self._total_tokens)

    def get_totals(self) -> dict[str, int]:
        """Get total usage statistics.

        Returns:
            Dictionary with total calls and tokens
        """
        return {
            "calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
        }

    def reset_totals(self):
        """Reset total counters."""
        self._total_calls = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0


# Global tracker instance
_tracker: LLMTracker | None = None


def get_tracker() -> LLMTracker:
    """Get or create the global LLM tracker."""
    global _tracker
    if _tracker is None:
        _tracker = LLMTracker()
    return _tracker


@contextmanager
def track_llm_call(model: str, operation: str = "chat_completion", **kwargs):
    """Convenience function to track LLM calls.

    Example:
        with track_llm_call("gpt-4", "vision_analysis") as span:
            # Make LLM call
            response = await model.analyze(image)

            # Record tokens
            tracker = get_tracker()
            tracker.record_tokens(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )
    """
    tracker = get_tracker()
    with tracker.track_call(model, operation, **kwargs) as span:
        yield span
