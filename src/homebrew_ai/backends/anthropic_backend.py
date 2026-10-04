"""Claude via the official Anthropic SDK.

Notes that shape this file:

* history is append-only and assistant turns are replayed verbatim, so
  thinking blocks stay valid (preserved thinking) and the prompt cache hits;
* thinking is adaptive and effort-controlled; on models that support it we ask
  for ``display: "updates"`` so the progress notes Claude writes between tool
  calls can be shown to the user;
* refusals (``stop_reason == "refusal"``) are checked before reading content,
  and server-side fallbacks are enabled where available;
* tool inputs stream eagerly and are validated by the agent loop before any
  tool runs.

Optional request features are switched off for the rest of the session if the
API rejects them, so older or newer models keep working.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from homebrew_ai.backends.base import Backend, BackendError, Message, ProgressCallback, TextCallback, ToolCall, ToolSpec, Turn

BETA_UPDATES = "thinking-display-updates-2026-08-18"
BETA_BINDING = "thinking-binding-controls-2026-08-01"
BETA_FALLBACK = "server-side-fallback-2026-07-01"

UPDATES_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5", "claude-mythos-5-1")
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")
NO_ADAPTIVE = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5", "claude-3")


def _matches(model: str, prefixes: tuple[str, ...]) -> bool:
    return any(model == p or model.startswith(p + "-") for p in prefixes)


class AnthropicBackend(Backend):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, *, effort: str | None = "medium", max_tokens: int | None = None, base_url: str | None = None, capability: str = "high"):
        super().__init__(model, capability)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise BackendError("the 'anthropic' package is not installed") from exc
        self._anthropic = anthropic
        kwargs: dict[str, Any] = {"max_retries": 3, "timeout": 600.0}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.effort = effort
        self.max_tokens = max_tokens or (32000 if "haiku" in model else 64000)
        adaptive = not _matches(model, NO_ADAPTIVE)
        self.features = {
            "adaptive": adaptive,
            "effort": adaptive and effort is not None,
            "updates": _matches(model, UPDATES_MODELS),
            "binding": adaptive,
            "fallback": _matches(model, FALLBACK_MODELS),
            "eager": True,
            "cache": True,
        }

    def supports_vision(self) -> bool:
        return True

    # -- request building ----------------------------------------------------

    def _tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            spec: dict[str, Any] = {"name": t.name, "description": t.description, "input_schema": t.parameters}
            if self.features["eager"]:
                spec["eager_input_streaming"] = True
            out.append(spec)
        return out

    def _messages(self, history: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results(extra_text: list[str] | None = None) -> None:
            nonlocal pending_results
            if pending_results or extra_text:
                content = list(pending_results)
                content += [{"type": "text", "text": t} for t in (extra_text or []) if t]
                out.append({"role": "user", "content": content})
            pending_results = []

        for m in history:
            if m.role == "tool":
                block: dict[str, Any] = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.parts[0] if m.parts else ""}
                if m.is_error:
                    block["is_error"] = True
                pending_results.append(block)
                if len(m.parts) > 1:  # reminder text appended after the results of this round
                    flush_results(m.parts[1:])
                continue
            if pending_results:
                flush_results()
            if m.role == "user":
                out.append({"role": "user", "content": [{"type": "text", "text": p} for p in m.parts if p]})
            else:
                if m.raw_provider == self.provider and isinstance(m.raw, list):
                    out.append({"role": "assistant", "content": m.raw})
                else:
                    content: list[dict[str, Any]] = []
                    if m.text:
                        content.append({"type": "text", "text": m.text})
                    for c in m.tool_calls:
                        content.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
                    out.append({"role": "assistant", "content": content or [{"type": "text", "text": "(no reply)"}]})
        flush_results()
        return out

    def _params(self, system: str, history: list[Message], tools: list[ToolSpec]) -> tuple[dict[str, Any], list[str]]:
        f = self.features
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system}],
            "messages": self._messages(history),
        }
        if tools:
            params["tools"] = self._tools(tools)
        if f["cache"]:
            params["cache_control"] = {"type": "ephemeral"}
        betas: list[str] = []
        if f["adaptive"]:
            thinking: dict[str, Any] = {"type": "adaptive"}
            if f["updates"]:
                thinking["display"] = "updates"
                betas.append(BETA_UPDATES)
            if f["binding"]:
                thinking["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
                betas.append(BETA_BINDING)
            params["thinking"] = thinking
        if f["effort"] and self.effort:
            params["output_config"] = {"effort": self.effort}
        if f["fallback"]:
            params["fallbacks"] = "default"
            betas.append(BETA_FALLBACK)
        return params, betas

    def _degrade(self, message: str) -> bool:
        """Turn off the optional feature an error message complains about. True if something changed."""
        msg = message.lower()
        rules = [
            ("display", "updates"),
            ("updates", "updates"),
            ("block_binding", "binding"),
            ("prefix_mismatch", "binding"),
            ("fallback", "fallback"),
            ("effort", "effort"),
            ("output_config", "effort"),
            ("eager_input_streaming", "eager"),
            ("thinking", "adaptive"),
            ("cache_control", "cache"),
        ]
        for needle, feature in rules:
            if needle in msg and self.features.get(feature):
                self.features[feature] = False
                if feature == "adaptive":
                    self.features.update(updates=False, binding=False, effort=False)
                return True
        return False

    # -- calls ---------------------------------------------------------------

    def _stream(self, params: dict[str, Any], betas: list[str], on_text: TextCallback | None, on_progress: TextCallback | None):
        if betas:
            ctx = self.client.beta.messages.stream(betas=betas, **params)
        else:
            params = {k: v for k, v in params.items() if k != "fallbacks"}
            ctx = self.client.messages.stream(**params)
        with ctx as stream:
            for event in stream:
                etype = getattr(event, "type", "")
                if etype == "text" and on_text:
                    on_text(event.text)
                elif etype == "thinking" and on_progress:
                    # with display="updates" these are Claude's short progress notes, meant for the user
                    on_progress(getattr(event, "thinking", "") or "")
                elif etype == "content_block_stop" and on_progress and getattr(getattr(event, "content_block", None), "type", "") == "thinking":
                    on_progress("\n")
            return stream.get_final_message()

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec], *, on_text: TextCallback | None = None, on_progress: TextCallback | None = None) -> Turn:
        anthropic = self._anthropic
        json_retries = 0
        degrade_retries = 0
        while True:
            params, betas = self._params(system, messages, tools)
            try:
                final = self._stream(params, betas, on_text, on_progress)
                break
            except anthropic.BadRequestError as exc:
                if degrade_retries < 4 and self._degrade(str(exc)):
                    degrade_retries += 1
                    continue
                raise BackendError(f"Claude rejected the request: {_err(exc)}") from exc
            except anthropic.AuthenticationError as exc:
                raise BackendError("the Anthropic API key was rejected; run `homebrew setup` to fix it") from exc
            except anthropic.PermissionDeniedError as exc:
                raise BackendError(f"this API key cannot use {self.model}: {_err(exc)}") from exc
            except anthropic.NotFoundError as exc:
                raise BackendError(f"model {self.model!r} was not found; run `homebrew setup` to pick another") from exc
            except anthropic.RateLimitError as exc:
                raise BackendError("Anthropic rate limit reached; wait a minute and try again") from exc
            except anthropic.APIStatusError as exc:
                raise BackendError(f"Anthropic API error {exc.status_code}: {_err(exc)}") from exc
            except anthropic.APIConnectionError as exc:
                raise BackendError("could not reach the Anthropic API; check your internet connection") from exc
            except anthropic.APIError as exc:  # e.g. an error event in the middle of the stream
                raise BackendError(f"Anthropic API error: {_err(exc)}") from exc
            except httpx.HTTPError as exc:
                raise BackendError(f"the connection to the Anthropic API broke off: {exc}") from exc
            except ValueError:
                # tool input JSON the SDK could not parse at all (eager streaming): re-issue the turn
                json_retries += 1
                if json_retries > 2:
                    raise BackendError("Claude produced unreadable tool input three times in a row")
                continue
        return self._to_turn(final)

    def _to_turn(self, final: Any) -> Turn:
        usage = getattr(final, "usage", None)
        usage_d = {}
        if usage is not None:
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                v = getattr(usage, k, None)
                if isinstance(v, int):
                    usage_d[k] = v
        stop = final.stop_reason or "end_turn"
        if stop == "refusal":
            # discard any partial output (including half-made tool calls); a short note keeps the history valid
            details = getattr(final, "stop_details", None)
            reason = getattr(details, "explanation", None) or getattr(details, "category", None) or "declined by safety filters"
            note = "(My reply was stopped by Claude's safety filters and discarded.)"
            message = Message(role="assistant", parts=[note], tool_calls=[], raw=[{"type": "text", "text": note}], raw_provider=self.provider)
            return Turn(message, "refusal", usage_d, refusal=str(reason))
        content = _served_content(list(final.content))
        raw = [_block_to_param(b) for b in content]
        texts = [b.text for b in content if getattr(b, "type", "") == "text"]
        calls = [ToolCall(b.id, b.name, b.input if isinstance(b.input, dict) else {}, None if isinstance(b.input, dict) else "input is not an object") for b in content if getattr(b, "type", "") == "tool_use"]
        message = Message(role="assistant", parts=["".join(texts)] if texts else [], tool_calls=calls, raw=raw, raw_provider=self.provider)
        if stop == "max_tokens":
            return Turn(message, "max_tokens", usage_d)
        return Turn(message, "tool_use" if calls else "end", usage_d)

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 8000,
        json_schema: dict[str, Any] | None = None,
        images: list[tuple[bytes, str]] | None = None,
        on_progress: ProgressCallback | None = None,
        effort: str | None = None,
    ) -> str:
        import base64

        anthropic = self._anthropic
        content: list[dict[str, Any]] = []
        for data, media_type in images or []:
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(data).decode()}})
        content.append({"type": "text", "text": prompt})
        params: dict[str, Any] = {"model": self.model, "max_tokens": max(max_tokens, 2000), "system": system, "messages": [{"role": "user", "content": content}]}
        output_config: dict[str, Any] = {}
        if self.features["effort"]:
            output_config["effort"] = "low" if effort in (None, "off") else effort
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            params["output_config"] = output_config
        for _ in range(3):
            try:
                with self.client.messages.stream(**params) as stream:
                    written = thought = 0
                    for event in stream:
                        etype = getattr(event, "type", "")
                        if etype == "text" and on_progress:
                            written += len(event.text or "")
                            on_progress("writing", written)
                        elif etype == "thinking" and on_progress:
                            thought += len(getattr(event, "thinking", "") or "")
                            on_progress("thinking", thought)
                    final = stream.get_final_message()
                break
            except anthropic.BadRequestError as exc:
                text = str(exc).lower()
                if "format" in text and "output_config" in params and "format" in params["output_config"]:
                    params["output_config"].pop("format")
                    continue
                if "effort" in text and "output_config" in params:
                    params["output_config"].pop("effort", None)
                    continue
                raise BackendError(f"Claude rejected the request: {_err(exc)}") from exc
            except anthropic.APIError as exc:
                raise BackendError(f"Anthropic API error: {_err(exc)}") from exc
            except httpx.HTTPError as exc:
                raise BackendError(f"the connection to the Anthropic API broke off: {exc}") from exc
        if final.stop_reason == "refusal":
            raise BackendError("the request was declined by Claude's safety filters")
        return "".join(b.text for b in final.content if getattr(b, "type", "") == "text")


def _served_content(content: list[Any]) -> list[Any]:
    """Content to keep after a server-side fallback.

    Before the last ``fallback`` block only text survives: the declined model's thinking and tool calls
    must neither be executed nor echoed back. The ``fallback`` marker itself is only an audit note.
    """
    last = max((i for i, b in enumerate(content) if getattr(b, "type", "") == "fallback"), default=-1)
    return [b for i, b in enumerate(content) if getattr(b, "type", "") != "fallback" and (i > last or getattr(b, "type", "") == "text")]


def _block_to_param(block: Any) -> dict[str, Any]:
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if btype == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if btype == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # by_alias: wire names such as "from" are Python attributes like ``from_`` in the SDK
    dump = block.model_dump(mode="json", by_alias=True, exclude_none=True) if hasattr(block, "model_dump") else dict(block)
    return dump


def _err(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:400]
    return str(exc)[:400]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
