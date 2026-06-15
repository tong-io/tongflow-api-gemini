from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tongflow.node_slots import NodeSlots
from tongflow.slots import node_slot
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.combine_text import CombineTextInput, CombineTextOutput
from tongflow.models.split_text import SplitTextInput, SplitTextOutput
from tongflow.models.drop_video import DropVideoInput, DropVideoOutput
from tongflow.models.arrange_group import ArrangeGroupInput, ArrangeGroupOutput
from tongflow.llm_batch_handlers import arrange_group_output, drop_video_output

# Plugin logs go to stderr — stdout is reserved for the ABI JSON response.
# Level can be tuned via `TONGFLOW_PLUGIN_LOG_LEVEL` (default INFO).
logging.basicConfig(
    level=os.environ.get("TONGFLOW_PLUGIN_LOG_LEVEL", "INFO").upper(),
    stream=sys.stderr,
    format="[gemini] %(levelname)s %(message)s",
)
log = logging.getLogger("tongflow.plugins.gemini")


DEFAULT_MODEL = "gemini-2.0-flash"


def _chat_gemini(
    *,
    api_key: str,
    model: str,
    user_message: str,
    response_schema: Dict[str, Any] | None = None,
) -> str:
    qs = urlencode({"key": api_key})
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent?"
        + qs
    )
    # `url` includes the api key in the query string — log a redacted form.
    redacted_url = url.split("?", 1)[0] + "?key=***"
    generation_config: Dict[str, Any] = {"temperature": 1.0}
    if response_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": generation_config,
    }
    headers = {"Content-Type": "application/json"}

    log.info("POST %s model=%s", redacted_url, model)
    log.debug("request payload: %s", json.dumps(payload, ensure_ascii=False))

    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=180)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        log.error(
            "HTTP %s on %s\nresponse body: %s", e.code, redacted_url, err_body
        )
        raise RuntimeError(
            f"HTTP {e.code} from Gemini: {err_body or e.reason}"
        ) from e
    except URLError as e:
        log.error("network error contacting %s: %s", redacted_url, e.reason)
        raise RuntimeError(f"Network error: {e.reason}") from e

    body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    candidates = obj.get("candidates") or []
    if not candidates:
        log.error("response missing 'candidates': %s", body[:500])
        raise RuntimeError("Gemini response missing candidates")
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    pieces: list[str] = []
    for part in parts:
        piece = part.get("text") if isinstance(part, dict) else None
        if isinstance(piece, str) and piece:
            pieces.append(piece)
    if not pieces:
        log.error("response missing text parts: %s", body[:500])
        raise RuntimeError("Gemini response missing text parts")
    return "".join(pieces).strip()


def _resolve_model() -> str:
    return (os.environ.get("GEMINI_MODEL") or "").strip() or DEFAULT_MODEL


def _require_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return api_key


@node_slot(NodeSlots.GEN_TEXT)
def gen_text(input: GenTextInput) -> GenTextOutput:
    user_message = (
        f"{input.userPrompt or ''}\n\nUser input: {input.text}\n\n"
        "Note: output only the requested answer. Do not include any other content."
    )
    answer = _chat_gemini(
        api_key=_require_api_key(),
        model=_resolve_model(),
        user_message=user_message,
    )
    return GenTextOutput(success=True, text=answer)


def _parse_split_texts(raw: str) -> list[str]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            _, _, s = s.partition("\n")
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    obj = json.loads(s)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        raw_items = obj.get("texts")
        items = raw_items if isinstance(raw_items, list) else None
    else:
        items = None
    if items is None or not all(isinstance(x, str) for x in items):
        raise ValueError("LLM did not return a JSON array of strings")
    cleaned = [x.strip() for x in items if x.strip()]
    if not cleaned:
        raise ValueError("LLM returned an empty split")
    return cleaned


def _build_split_user_message(input: SplitTextInput) -> str:
    instruction = (input.userPrompt or "").strip() or "Split into natural, coherent segments."
    return (
        f"Split the following text into multiple segments according to this instruction:\n"
        f"{instruction}\n\n"
        f"Return ONLY a JSON array of strings — no prose, no markdown, no keys, no code fences. "
        f"Each array element is one segment. Preserve the original wording; do not summarize.\n\n"
        f"TEXT:\n{input.text}"
    )


@node_slot(NodeSlots.SPLIT_TEXT)
def split_text(input: SplitTextInput) -> SplitTextOutput:
    raw = _chat_gemini(
        api_key=_require_api_key(),
        model=_resolve_model(),
        user_message=_build_split_user_message(input),
        response_schema={"type": "array", "items": {"type": "string"}},
    )
    try:
        texts = _parse_split_texts(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return SplitTextOutput(success=False, error=str(e))
    return SplitTextOutput(success=True, texts=texts)


@node_slot(NodeSlots.COMBINE_TEXT)
def combine_text(input: CombineTextInput) -> CombineTextOutput:
    joined = "\n\n".join(input.texts)
    user_message = (
        f"{input.userPrompt or ''}\n\nUser input: {joined}\n\n"
        "Note: output only the requested answer. Do not include any other content."
    )
    answer = _chat_gemini(
        api_key=_require_api_key(),
        model=_resolve_model(),
        user_message=user_message,
    )
    return CombineTextOutput(success=True, text=answer)


@node_slot(NodeSlots.DROP_VIDEO)
def drop_video(input: DropVideoInput) -> DropVideoOutput:
    result = drop_video_output(input.model_dump())
    return DropVideoOutput.model_construct(**result)


@node_slot(NodeSlots.ARRANGE_GROUP)
def arrange_group(input: ArrangeGroupInput) -> ArrangeGroupOutput:
    result = arrange_group_output(input.model_dump())
    return ArrangeGroupOutput.model_construct(**result)


# Runtime dispatcher. The @node_slot wrapper accepts a raw dict here (it
# deep-constructs the typed BaseModel internally) and dumps the BaseModel
# return to a dict. `Any` reflects the I/O boundary, not the plugin contract.
_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.GEN_TEXT: gen_text,
    NodeSlots.COMBINE_TEXT: combine_text,
    NodeSlots.SPLIT_TEXT: split_text,
    NodeSlots.DROP_VIDEO: drop_video,
    NodeSlots.ARRANGE_GROUP: arrange_group,
}


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        prompt = req.get("prompt") if isinstance(req, dict) else {}
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "") if isinstance(req, dict) else ""

        handler = _SLOT_HANDLERS.get(slot)
        if handler is None:
            raise RuntimeError(f"unsupported nodeSlot: {slot!r}")
        out = handler(prompt)
    except Exception as e:  # noqa: BLE001 — surfaced as ABI failure
        _write({"success": False, "error": str(e)})
        return 1

    _write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
