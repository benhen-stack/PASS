"""Language-model backends for the contextual score, with an on-disk prompt cache.

The method itself takes the model as a plain callable ``score(prompt) -> str``,
so these classes are a convenience rather than part of PASS. Three are provided:

  * ``OllamaClient``  the default. A local model over the OpenAI-compatible
    endpoint, so the contextual score is reproducible with no API key and no
    quota.
  * ``GeminiClient``  the hosted configuration used for the runs reported in
    the paper, decoding deterministically at temperature 0 with top_p 1.0 and
    top_k 1.
  * ``ConstantClient``  returns a fixed severity for every prompt. With
    ``value=1.0`` the potential reduces to the geometric score, which is the
    geometry-only ablation, and it needs no inference at all.

Any other provider plugs in by passing its own callable to ``pass_augment``;
nothing in the method has to change.

``CachedLLM`` keys responses on the SHA-256 of the prompt. With the cache warm,
re-running a grid over the synthesis parameters costs nothing, because those
parameters do not enter the prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"


class LLMError(RuntimeError):
    pass


class GeminiClient:
    """google-generativeai backend, the configuration reported in the paper.

    The API key is read from the ``GEMINI_API_KEY`` environment variable, or
    passed explicitly. It is never written to disk or into a prompt.

    The thinking budget is pinned, because the 2.5 reasoning models otherwise
    spend the whole output allowance on internal reasoning and return empty
    text with ``finish_reason=2``.
    """

    def __init__(
        self,
        api_key=None,
        model_name="gemini-2.5-pro",
        max_output_tokens=256,
        thinking_budget=128,
        min_interval=0.0,
    ):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMError(
                "The gemini backend needs google-generativeai. Install it with "
                "'pip install google-generativeai', or use --backend ollama."
            ) from exc

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError(
                "No Gemini API key. Set GEMINI_API_KEY in the environment, or "
                "pass api_key=... Do not hard-code it."
            )
        genai.configure(api_key=api_key)

        config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "candidate_count": 1,
            "max_output_tokens": max_output_tokens,
        }
        if thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": thinking_budget}

        try:
            self._model = genai.GenerativeModel(model_name, generation_config=config)
        except TypeError:
            # An SDK without the thinking field: drop it and retry.
            config.pop("thinking_config", None)
            self._model = genai.GenerativeModel(model_name, generation_config=config)

        self.model_name = model_name
        self.min_interval = min_interval
        self._last_call = 0.0
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "seconds": 0.0}

    def generate(self, prompt):
        if self.min_interval:
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        t0 = time.time()
        response = self._model.generate_content(prompt)
        self._last_call = time.time()
        meta = getattr(response, "usage_metadata", None)
        self.usage["calls"] += 1
        self.usage["seconds"] += time.time() - t0
        if meta is not None:
            self.usage["prompt_tokens"] += int(getattr(meta, "prompt_token_count", 0))
            self.usage["completion_tokens"] += int(
                getattr(meta, "candidates_token_count", 0)
            )
        text = getattr(response, "text", None)
        if not text:
            raise LLMError(
                "Empty response. A finish_reason of 2 means the output "
                "allowance was consumed, usually by an unpinned thinking budget."
            )
        return text


class OllamaClient:
    """Local Ollama model over the OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model_name="llama3",
        base_url="http://localhost:11434/v1",
        max_output_tokens=256,
        timeout=180,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        # Cumulative usage, read back through CachedLLM.stats().
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "seconds": 0.0}

    def generate(self, prompt):
        import requests

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_output_tokens,
            "seed": 42,
        }
        t0 = time.time()
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout
            )
            r.raise_for_status()
        except Exception as exc:
            raise LLMError(
                f"Ollama request failed: {exc}. Is 'ollama serve' running and "
                f"'{self.model_name}' pulled?"
            ) from exc
        data = r.json()
        use = data.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["seconds"] += time.time() - t0
        self.usage["prompt_tokens"] += int(use.get("prompt_tokens", 0))
        self.usage["completion_tokens"] += int(use.get("completion_tokens", 0))
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Ollama payload: {data}") from exc


class ConstantClient:
    """Ablation backend. Returns a fixed severity for every prompt."""

    def __init__(self, value=1.0):
        self.value = float(value)
        self.model_name = f"constant-{self.value}"
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "seconds": 0.0}

    def generate(self, prompt):
        self.usage["calls"] += 1
        return f"Severity: {self.value}"


class CachedLLM:
    """Disk-backed memo over any backend exposing ``generate(prompt) -> str``."""

    def __init__(self, backend, cache_dir=None, namespace=None):
        self.backend = backend
        self.namespace = namespace or getattr(backend, "model_name", "llm")
        self.path = Path(cache_dir or CACHE_DIR) / f"{self.namespace}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store = {}
        if self.path.exists():
            try:
                self._store = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._store = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(prompt):
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def generate(self, prompt):
        key = self._key(prompt)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        text = self.backend.generate(prompt)
        self._store[key] = text
        self.misses += 1
        if self.misses % 10 == 0:
            self.flush()
        return text

    def flush(self):
        self.path.write_text(json.dumps(self._store, indent=0))

    def stats(self):
        """Aggregate counters for the run profile."""
        u = dict(getattr(self.backend, "usage", {}))
        calls = u.get("calls", 0)
        out = {
            "model": self.namespace,
            "prompts_served": self.hits + self.misses,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "backend_calls": calls,
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("prompt_tokens", 0) + u.get("completion_tokens", 0),
            "backend_seconds": round(u.get("seconds", 0.0), 1),
            "cached_entries": len(self._store),
        }
        if calls:
            out["mean_seconds_per_call"] = round(u["seconds"] / calls, 3)
            out["mean_prompt_tokens"] = round(u.get("prompt_tokens", 0) / calls, 1)
        if out["prompts_served"]:
            out["cache_hit_rate_pct"] = round(
                100 * self.hits / out["prompts_served"], 1
            )
        return out


BACKENDS = {"ollama": OllamaClient, "gemini": GeminiClient,
            "constant": ConstantClient}


def make_llm(backend="ollama", cache_dir=None, **kwargs):
    """Factory. ``backend`` is ``ollama`` (the default), ``gemini`` or ``constant``."""
    try:
        cls = BACKENDS[backend.lower()]
    except KeyError:
        raise ValueError(
            f"backend must be one of {sorted(BACKENDS)}, got {backend!r}"
        ) from None
    return CachedLLM(cls(**kwargs), cache_dir=cache_dir)
