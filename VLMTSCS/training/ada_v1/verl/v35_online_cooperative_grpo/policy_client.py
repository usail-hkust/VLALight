"""Policy adapters for online Stage 2 rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OpenAICompatiblePolicy:
    """Callable policy for an OpenAI-compatible vLLM endpoint.

    The endpoint is intended for smoke tests or deployments where the serving
    process is refreshed with the latest checkpoint between GRPO updates.
    ``assistant_prefix`` is sent as a prefill so forced fast/slow assignments
    remain part of the sampled response contract.
    """

    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 300.0
    api_key: str = "EMPTY"

    def __call__(self, intersection_id: str, prompt: str, assistant_prefix: str = "") -> str:
        del intersection_id
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ImportError("requests is required for OpenAICompatiblePolicy") from exc
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": float(self.temperature),
                "max_tokens": int(self.max_tokens),
                "enable_thinking": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: Any = response.json()
        generated = str(payload["choices"][0]["message"].get("content", ""))
        # vLLM may return only the continuation after an assistant prefill.
        # Restore the prefix so the parser and training response see the mode.
        if assistant_prefix:
            prefix = assistant_prefix.strip()
            if not generated.lstrip().lower().startswith(prefix.lower()):
                generated = assistant_prefix + generated
        return generated


def make_openai_policy(**kwargs: Any) -> OpenAICompatiblePolicy:
    return OpenAICompatiblePolicy(**kwargs)


def make_policy_factory(*, base_url: str, model: str, temperature: float = 0.7,
                        max_tokens: int = 512, timeout: float = 300.0,
                        api_key: str = "EMPTY"):
    """Return the zero-argument factory expected by Ray rollout actors."""
    def factory() -> OpenAICompatiblePolicy:
        return OpenAICompatiblePolicy(base_url, model, temperature, max_tokens, timeout, api_key)
    return factory


__all__ = ["OpenAICompatiblePolicy", "make_openai_policy", "make_policy_factory"]
