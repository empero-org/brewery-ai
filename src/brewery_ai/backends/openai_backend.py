"""Any OpenAI-compatible chat endpoint: OpenAI, OpenRouter, Ollama, LM Studio, vLLM, llama.cpp, ...

Two tool modes:

* ``native`` — standard function calling (``tools`` / ``tool_calls``);
* ``text`` — for servers or models without function calling: tools are
  described in the system prompt and the model answers with
  ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks.

Even in native mode, a reply that contains such blocks in plain text (common
with local servers whose tool parser is off) is understood as tool calls.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx

from brewery_ai.backends.base import Backend, BackendError, Message, ProgressCallback, TextCallback, ToolCall, ToolSpec, Turn, parse_json_loose

TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

TEXT_TOOLS_PROMPT = """
# Tools
You can use the tools listed below. To call a tool, reply with one block per call and nothing after it:
<tool_call>{{"name": "<tool name>", "arguments": {{...}}}}</tool_call>
You will receive the result in a <tool_result> block. Only call tools listed here.

{tools}
"""


class OpenAICompatBackend(Backend):
    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        capability: str = "medium",
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_mode: str = "native",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(model, capability)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise BackendError("the 'openai' package is not installed") from exc
        self._openai = openai
        self.client = openai.OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=600.0, max_retries=2, default_headers=extra_headers or None)
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.options = {"stream_options": True, "parallel_tool_calls": True}

    def supports_vision(self) -> bool:
        return True  # depends on the model; callers handle errors

    # -- request building ----------------------------------------------------

    def _system(self, system: str, tools: list[ToolSpec]) -> str:
        if self.tool_mode != "text" or not tools:
            return system
        listing = "\n".join(json.dumps({"name": t.name, "description": t.description, "parameters": t.parameters}, ensure_ascii=False) for t in tools)
        return system + "\n" + TEXT_TOOLS_PROMPT.format(tools=listing)

    def _messages(self, system: str, history: list[Message], tools: list[ToolSpec]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": self._system(system, tools)}]
        text_mode = self.tool_mode == "text"
        for m in history:
            if m.role == "user":
                out.append({"role": "user", "content": m.text})
            elif m.role == "tool":
                result = m.parts[0] if m.parts else ""
                extra = "\n\n".join(m.parts[1:])
                if text_mode:
                    content = f'<tool_result name="{m.name}" id="{m.tool_call_id}">\n{result}\n</tool_result>' + (f"\n\n{extra}" if extra else "")
                    if out and out[-1]["role"] == "user":
                        out[-1]["content"] += "\n\n" + content
                    else:
                        out.append({"role": "user", "content": content})
                else:
                    out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": result})
                    if extra:
                        out.append({"role": "user", "content": extra})
            else:
                if text_mode:
                    text = m.text
                    for c in m.tool_calls:
                        text += "\n<tool_call>" + json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False) + "</tool_call>"
                    out.append({"role": "assistant", "content": text.strip() or "(no reply)"})
                elif m.raw_provider == self.provider and isinstance(m.raw, dict):
                    out.append(m.raw)
                else:
                    msg: dict[str, Any] = {"role": "assistant", "content": m.text or None}
                    if m.tool_calls:
                        msg["tool_calls"] = [
                            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                            for c in m.tool_calls
                        ]
                    out.append(msg)
        return out

    def _request(self, system: str, history: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        req: dict[str, Any] = {"model": self.model, "messages": self._messages(system, history, tools), "stream": True}
        if tools and self.tool_mode == "native":
            req["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]
        if self.options["stream_options"]:
            req["stream_options"] = {"include_usage": True}
        if self.temperature is not None:
            req["temperature"] = self.temperature
        if self.max_tokens:
            req["max_tokens"] = self.max_tokens
        return req

    # -- calls ---------------------------------------------------------------

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec], *, on_text: TextCallback | None = None, on_progress: TextCallback | None = None) -> Turn:
        openai = self._openai
        for attempt in range(3):
            req = self._request(system, messages, tools)
            try:
                return self._run_stream(req, on_text, on_progress)
            except openai.BadRequestError as exc:
                text = str(exc).lower()
                if "stream_options" in text and self.options["stream_options"]:
                    self.options["stream_options"] = False
                    continue
                if self.tool_mode == "native" and tools and _no_tool_support(text):
                    self.tool_mode = "text"  # server cannot do function calling: fall back to the text protocol
                    continue
                raise BackendError(f"the model server rejected the request: {_err(exc)}") from exc
            except openai.AuthenticationError as exc:
                raise BackendError("the API key was rejected; run `brewery setup` to fix it") from exc
            except openai.NotFoundError as exc:
                raise BackendError(f"model {self.model!r} was not found on {self.base_url or 'the server'}") from exc
            except openai.RateLimitError as exc:
                raise BackendError("rate limit reached; wait a minute and try again") from exc
            except openai.APIConnectionError as exc:
                where = self.base_url or "the API"
                raise BackendError(f"could not reach {where}; is the server running?") from exc
            except openai.APIStatusError as exc:
                raise BackendError(f"API error {exc.status_code}: {_err(exc)}") from exc
            except openai.APIError as exc:  # e.g. an error event in the middle of the stream (common on OpenRouter)
                raise BackendError(f"the model provider reported an error: {_err(exc)}") from exc
            except httpx.HTTPError as exc:
                raise BackendError(f"the connection to {self.base_url or 'the API'} broke off: {exc}") from exc
        raise BackendError("the model server kept rejecting the request")

    def _run_stream(self, req: dict[str, Any], on_text: TextCallback | None, on_progress: TextCallback | None) -> Turn:
        stream = self.client.chat.completions.create(**req)
        text_parts: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish = None
        usage: dict[str, int] = {}
        hide = _ToolBlockFilter(on_text) if on_text else None
        for chunk in stream:
            if getattr(chunk, "usage", None):
                u = chunk.usage
                usage = {"input_tokens": getattr(u, "prompt_tokens", 0) or 0, "output_tokens": getattr(u, "completion_tokens", 0) or 0}
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                piece = getattr(delta, "content", None)
                if piece:
                    text_parts.append(piece)
                    if hide:
                        hide.feed(piece)
                r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if isinstance(r, str) and r:
                    reasoning.append(r)
                    if on_progress:
                        on_progress(r)  # shown as a separate, dim "thinking" block
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = calls.setdefault(tc.index if tc.index is not None else len(calls), {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments
            if choice.finish_reason:
                finish = choice.finish_reason
        if hide:
            hide.flush()
        text = "".join(text_parts)
        tool_calls: list[ToolCall] = []
        for _, slot in sorted(calls.items()):
            args, err = _parse_args(slot["arguments"])
            tool_calls.append(ToolCall(slot["id"] or f"call_{uuid.uuid4().hex[:8]}", slot["name"], args, err))
        if not tool_calls and "<tool_call>" in text:
            tool_calls, text = _calls_from_text(text)
        raw = None
        if self.tool_mode == "native":
            raw = {"role": "assistant", "content": text or None}
            if tool_calls:
                raw["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}} for c in tool_calls]
        message = Message("assistant", [text.strip()] if text.strip() else [], tool_calls, raw=raw, raw_provider=self.provider if raw else None)
        if finish == "length" and not tool_calls:
            return Turn(message, "max_tokens", usage)
        if finish == "content_filter":
            # drop the partial reply and any tool calls: a dangling call would break every later request
            note = "(My reply was stopped by the provider's content filter and discarded.)"
            raw_note = {"role": "assistant", "content": note} if self.tool_mode == "native" else None
            message = Message("assistant", [note], [], raw=raw_note, raw_provider=self.provider if raw_note else None)
            return Turn(message, "refusal", usage, refusal="the provider's content filter stopped the reply")
        return Turn(message, "tool_use" if tool_calls else "end", usage)

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

        openai = self._openai
        if images:
            content: Any = [{"type": "image_url", "image_url": {"url": f"data:{mt};base64,{base64.b64encode(data).decode()}"}} for data, mt in images]
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt
        req: dict[str, Any] = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}], "stream": True}
        if json_schema is not None:
            req["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            req["temperature"] = self.temperature
        if effort:
            host = (self.base_url or "https://api.openai.com").lower()
            if "openrouter.ai" in host:  # OpenRouter's unified reasoning control
                req["extra_body"] = {"reasoning": {"enabled": False} if effort == "off" else {"effort": effort}}
            elif "api.openai.com" in host:
                req["reasoning_effort"] = "minimal" if effort == "off" else effort
        for _ in range(4):
            try:
                return self._stream_text(req, on_progress)
            except openai.BadRequestError as exc:
                if ("extra_body" in req or "reasoning_effort" in req) and "reason" in str(exc).lower():
                    req.pop("extra_body", None)
                    req.pop("reasoning_effort", None)  # the model has no adjustable reasoning
                    continue
                if "response_format" in req:
                    req.pop("response_format")
                    continue
                raise BackendError(f"the model server rejected the request: {_err(exc)}") from exc
            except openai.APIError as exc:
                raise BackendError(f"API error: {_err(exc)}") from exc
            except httpx.HTTPError as exc:
                raise BackendError(f"the connection to {self.base_url or 'the API'} broke off: {exc}") from exc
        raise BackendError("the model server kept rejecting the request")

    def _stream_text(self, req: dict[str, Any], on_progress: ProgressCallback | None) -> str:
        """Streamed one-shot answer; streaming keeps long generations visibly alive."""
        parts: list[str] = []
        written = thought = 0
        for chunk in self.client.chat.completions.create(**req):
            if not chunk.choices or chunk.choices[0].delta is None:
                continue
            delta = chunk.choices[0].delta
            r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if isinstance(r, str) and r:
                thought += len(r)
                if on_progress:
                    on_progress("thinking", thought)
            piece = getattr(delta, "content", None)
            if piece:
                parts.append(piece)
                written += len(piece)
                if on_progress:
                    on_progress("writing", written)
        return "".join(parts)

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self.client.models.list())
        except Exception:
            return []


_NO_TOOLS_RE = re.compile(
    r"(tools?|function[ _]?call(ing|s)?|tool[ _]?(use|choice|calls?))\b[^.]{0,60}\b(not|un)\s*(supported|available|enabled|allowed)"
    r"|(does not|doesn't|do not|cannot|can't)\s+support\s+(tools?|function|tool[ _]?use)"
    r"|unrecognized\s+(request\s+)?(argument|field|parameter)[^.]{0,40}\btools?\b"
    r"|extra (inputs|fields)[^.]{0,40}\btools?\b"
    r"|no endpoints found that support tool",
    re.I,
)


def _no_tool_support(error_text: str) -> bool:
    """Does this 400 say the server cannot do function calling (and not, say, that the prompt is too long)?"""
    return bool(_NO_TOOLS_RE.search(error_text))


class _ToolBlockFilter:
    """Streams text to the UI but hides ``<tool_call>`` blocks."""

    def __init__(self, sink: TextCallback):
        self.sink = sink
        self.buf = ""
        self.inside = False

    def feed(self, piece: str) -> None:
        self.buf += piece
        while self.buf:
            if self.inside:
                end = self.buf.find("</tool_call>")
                if end < 0:
                    return
                self.buf = self.buf[end + len("</tool_call>"):]
                self.inside = False
            else:
                start = self.buf.find("<tool_call>")
                if start < 0:
                    keep = len(self.buf) - len("<tool_call>") + 1
                    if keep > 0:
                        self.sink(self.buf[:keep])
                        self.buf = self.buf[keep:]
                    return
                if start:
                    self.sink(self.buf[:start])
                self.buf = self.buf[start + len("<tool_call>"):]
                self.inside = True

    def flush(self) -> None:
        if self.buf and not self.inside:
            self.sink(self.buf)
        self.buf = ""


def _parse_args(raw: str) -> tuple[dict[str, Any], str | None]:
    if not raw.strip():
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = parse_json_loose(raw)
        except ValueError:
            return {}, f"arguments are not valid JSON: {raw[:200]}"
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, None


def _calls_from_text(text: str) -> tuple[list[ToolCall], str]:
    calls = []
    for m in TOOL_BLOCK_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("name"):
            args = data.get("arguments") or data.get("parameters") or {}
            if isinstance(args, str):
                args, _ = _parse_args(args)
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", str(data["name"]), args if isinstance(args, dict) else {}))
    return calls, TOOL_BLOCK_RE.sub("", text).strip()


def _err(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:400]
    return str(exc)[:400]
