from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
import wave
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tongflow.node_slots import NodeSlots
from tongflow.protocol import Asset, asset, prompt_media_to_bytes
from tongflow.slots import node_slot
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.audio_describe import (
    AudioDescribeInput,
    AudioDescribeOutput,
)
from tongflow.models.combine_text import CombineTextInput, CombineTextOutput
from tongflow.models.split_text import SplitTextInput, SplitTextOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_fusion import ImageFusionInput, ImageFusionOutput
from tongflow.models.image_gen_text import ImageGenTextInput, ImageGenTextOutput
from tongflow.models.image_describe import ImageDescribeInput, ImageDescribeOutput
from tongflow.models.video_gen_text import VideoGenTextInput, VideoGenTextOutput
from tongflow.models.video_describe import VideoDescribeInput, VideoDescribeOutput
from tongflow.models.parse_document import ParseDocumentInput, ParseDocumentOutput
from tongflow.models.transcribe import TranscribeInput, TranscribeOutput
from tongflow.models.transcribe_timestamp import (
    TranscribeTimestampInput,
    TranscribeTimestampOutput,
    TranscribeTimestampOutputRootTimeStampsItem,
)
from tongflow.models.text_gen_speech_preset import (
    TextGenSpeechPresetInput,
    TextGenSpeechPresetOutput,
)
from tongflow.models.text_gen_speech_instruct import (
    TextGenSpeechInstructInput,
    TextGenSpeechInstructOutput,
)
from tongflow.models.text_gen_video import TextGenVideoInput, TextGenVideoOutput
from tongflow.models.image_gen_video import ImageGenVideoInput, ImageGenVideoOutput
from tongflow.models.images_gen_video import ImagesGenVideoInput, ImagesGenVideoOutput
from tongflow.models.image_image_gen_video import (
    ImageImageGenVideoInput,
    ImageImageGenVideoOutput,
)
from tongflow.models.drop_video import DropVideoInput, DropVideoOutput
from tongflow.models.arrange_group import ArrangeGroupInput, ArrangeGroupOutput
from tongflow.llm_batch_handlers import arrange_group_output, drop_video_output


# ── Per-node model picker ───────────────────────────────────────────────────
# Pure dict literal read by the platform scanner via AST (module never runs at
# scan time). First entry per slot = default when no model is picked. Model ids
# verified 2026-07 — Gemini 3.x suffixes rotate; re-check the live model list
# (ai.google.dev/gemini-api/docs/models) before release.
TONGFLOW_SLOT_MODELS = {
    "gen-text": ["gemini-3.6-flash", "gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "split-text": ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "combine-text": ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    "image-gen": ["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image", "imagen-4.0-generate-001", "imagen-4.0-ultra-generate-001", "imagen-4.0-fast-generate-001"],
    "image-edit": ["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image"],
    "image-fusion": ["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image"],
    "image-gen-text": ["gemini-3.6-flash", "gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
    "image-describe": ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    "video-gen-text": ["gemini-2.5-pro", "gemini-3.6-flash", "gemini-2.5-flash"],
    "video-describe": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "audio-describe": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.6-flash"],
    "parse-document": ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-2.5-flash"],
    "transcribe": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "transcribe-timestamp": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "text-gen-speech-preset": ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"],
    "text-gen-speech-instruct": ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"],
    "text-gen-video": ["veo-3.1-generate-preview", "veo-3.1-fast-generate-001", "veo-3.1-lite-generate-preview"],
    "image-gen-video": ["veo-3.1-generate-preview", "veo-3.1-fast-generate-001"],
    "images-gen-video": ["veo-3.1-generate-preview", "veo-3.1-fast-generate-001"],
    "image-image-gen-video": ["veo-3.1-generate-preview"],
}

# Slots this plugin is the default implementation of: the node picker lists it
# first and a newly added node preselects it. Read statically by the scanner.
TONGFLOW_DEFAULT_SLOTS = [
    "gen-text",
    "split-text",
    "combine-text",
    "arrange-group",
    "drop-video",
    "audio-describe",
    "image-gen",
    "image-edit",
    "image-fusion",
]

# Set from the request envelope's top-level `model` field in main().
_REQUEST_MODEL: str = ""

# Plugin logs go to stderr — stdout is reserved for the ABI JSON response.
logging.basicConfig(
    level=os.environ.get("TONGFLOW_PLUGIN_LOG_LEVEL", "INFO").upper(),
    stream=sys.stderr,
    format="[gemini] %(levelname)s %(message)s",
)
log = logging.getLogger("tongflow.plugins.gemini")


BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image"
DEFAULT_IMAGE_SIZE = "1K"


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _active_model(slot: str, env_override: str = "") -> str:
    """Resolve the model for a slot: per-node pick > legacy env override >
    list default."""
    models = TONGFLOW_SLOT_MODELS[slot]
    if _REQUEST_MODEL:
        if _REQUEST_MODEL not in models:
            raise RuntimeError(
                f"unknown model {_REQUEST_MODEL!r} for {slot}; "
                f"available: {', '.join(models)}"
            )
        return _REQUEST_MODEL
    if env_override:
        return env_override
    return models[0]


def _require_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return api_key


def _post_json(url: str, payload: Dict[str, Any], *, timeout: int = 180) -> Dict[str, Any]:
    redacted = url.split("?", 1)[0] + "?key=***"
    log.info("POST %s", redacted)
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=timeout)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        log.error("HTTP %s on %s\nbody: %s", e.code, redacted, err_body)
        raise RuntimeError(f"HTTP {e.code} from Gemini: {err_body or e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e
    return json.loads(resp.read().decode("utf-8", errors="replace"))


def _get_json(url: str, *, timeout: int = 60) -> Dict[str, Any]:
    try:
        resp = urlopen(url, timeout=timeout)  # noqa: S310
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from Gemini") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e
    return json.loads(resp.read().decode("utf-8", errors="replace"))


def _inline_part(a: Any, *, default_mime: str) -> Dict[str, Any]:
    mime = (a.mime or default_mime).strip() or default_mime
    return {"inlineData": {"mimeType": mime, "data": a.bytesBase64}}


def _generate_content(
    *,
    model: str,
    user_message: str,
    extra_parts: List[Dict[str, Any]] | None = None,
    response_schema: Dict[str, Any] | None = None,
) -> str:
    """Call `generateContent` and return concatenated text parts."""
    api_key = _require_api_key()
    url = f"{BASE}/models/{model}:generateContent?" + urlencode({"key": api_key})
    generation_config: Dict[str, Any] = {"temperature": 1.0}
    if response_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema
    parts: List[Dict[str, Any]] = [{"text": user_message}]
    if extra_parts:
        parts.extend(extra_parts)
    obj = _post_json(
        url,
        {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        },
    )
    candidates = obj.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini response missing candidates")
    pieces: list[str] = []
    for part in (candidates[0] or {}).get("content", {}).get("parts") or []:
        piece = part.get("text") if isinstance(part, dict) else None
        if isinstance(piece, str) and piece:
            pieces.append(piece)
    if not pieces:
        raise RuntimeError("Gemini response missing text parts")
    return "".join(pieces).strip()


# ── Image generation (Nano Banana / Imagen) ────────────────────────────────

_IMAGE_RATIOS: Dict[str, float] = {
    "1:1": 1.0, "3:2": 3 / 2, "2:3": 2 / 3, "3:4": 3 / 4, "4:3": 4 / 3,
    "4:5": 4 / 5, "5:4": 5 / 4, "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9,
}


def _resolve_image_model() -> str:
    return _env("GEMINI_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL


def _resolve_image_size() -> str:
    return _env("GEMINI_IMAGE_SIZE") or DEFAULT_IMAGE_SIZE


def _resolve_aspect_ratio(width: int | None, height: int | None) -> str | None:
    override = _env("GEMINI_IMAGE_ASPECT_RATIO")
    if override:
        return override
    if width and height and height > 0:
        target = width / height
        return min(_IMAGE_RATIOS, key=lambda r: abs(_IMAGE_RATIOS[r] - target))
    return None


def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _extract_image_asset(obj: Dict[str, Any]) -> Asset:
    candidates = obj.get("candidates") or []
    if not candidates:
        feedback = obj.get("promptFeedback") or {}
        reason = feedback.get("blockReason")
        raise RuntimeError(
            f"Gemini image response blocked: {reason}"
            if reason
            else "Gemini image response missing candidates"
        )
    fallback_text: list[str] = []
    for part in (candidates[0] or {}).get("content", {}).get("parts") or []:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict):
            data = inline.get("data")
            if isinstance(data, str) and data:
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                return asset(base64.b64decode(data), mime=mime)
        piece = part.get("text")
        if isinstance(piece, str) and piece:
            fallback_text.append(piece)
    detail = (" ".join(fallback_text)).strip()
    raise RuntimeError(
        f"Gemini returned no image (model said: {detail})"
        if detail
        else "Gemini image response contained no image data"
    )


def _imagen_predict(
    *, model: str, prompt: str, width: int | None, height: int | None
) -> Asset:
    """Imagen 4 uses the :predict surface, not generateContent."""
    api_key = _require_api_key()
    url = f"{BASE}/models/{model}:predict?" + urlencode({"key": api_key})
    parameters: Dict[str, Any] = {"sampleCount": 1}
    ratio = _resolve_aspect_ratio(width, height)
    if ratio:
        parameters["aspectRatio"] = ratio
    obj = _post_json(
        url, {"instances": [{"prompt": prompt}], "parameters": parameters}, timeout=300
    )
    for pred in obj.get("predictions") or []:
        if isinstance(pred, dict):
            b64 = pred.get("bytesBase64Encoded") or pred.get("bytes_base64_encoded")
            if isinstance(b64, str) and b64:
                mime = pred.get("mimeType") or "image/png"
                return asset(base64.b64decode(b64), mime=mime)
    raise RuntimeError("Imagen response contained no image data")


def _generate_image(
    *, model: str, prompt: str, images: List[bytes], width: int | None, height: int | None
) -> Asset:
    if model.startswith("imagen"):
        if images:
            raise RuntimeError("Imagen models do not accept reference images; use a Gemini image model")
        return _imagen_predict(model=model, prompt=prompt, width=width, height=height)

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for blob in images:
        parts.append(
            {"inlineData": {"mimeType": _sniff_mime(blob), "data": base64.b64encode(blob).decode("ascii")}}
        )
    image_config: Dict[str, Any] = {"imageSize": _resolve_image_size()}
    ratio = _resolve_aspect_ratio(width, height)
    if ratio:
        image_config["aspectRatio"] = ratio
    api_key = _require_api_key()
    url = f"{BASE}/models/{model}:generateContent?" + urlencode({"key": api_key})
    obj = _post_json(
        url,
        {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": image_config},
        },
        timeout=300,
    )
    return _extract_image_asset(obj)


# ── TTS (generateContent AUDIO modality) ────────────────────────────────────

DEFAULT_TTS_VOICE = "Kore"


def _pcm_to_wav(pcm: bytes, *, rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _rate_from_mime(mime: str) -> int:
    for token in (mime or "").split(";"):
        token = token.strip()
        if token.startswith("rate="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                pass
    return 24000


def _synthesize_speech(*, model: str, text: str, voice: str, style: str | None) -> Asset:
    prompt = f"Say in a {style} voice: {text}" if style else text
    api_key = _require_api_key()
    url = f"{BASE}/models/{model}:generateContent?" + urlencode({"key": api_key})
    obj = _post_json(
        url,
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
        },
        timeout=300,
    )
    candidates = obj.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini TTS response missing candidates")
    for part in (candidates[0] or {}).get("content", {}).get("parts") or []:
        inline = part.get("inlineData") or part.get("inline_data") if isinstance(part, dict) else None
        if isinstance(inline, dict):
            data = inline.get("data")
            if isinstance(data, str) and data:
                mime = inline.get("mimeType") or inline.get("mime_type") or "audio/L16;rate=24000"
                pcm = base64.b64decode(data)
                # Gemini returns raw 16-bit PCM; wrap it in a WAV container.
                return asset(_pcm_to_wav(pcm, rate=_rate_from_mime(mime)), mime="audio/wav")
    raise RuntimeError("Gemini TTS response contained no audio data")


# ── Veo video (predictLongRunning + operation poll) ─────────────────────────

VEO_POLL_INTERVAL_S = 10
VEO_POLL_TIMEOUT_S = 600


def _veo_generate(
    *,
    model: str,
    prompt: str,
    first_image: Any | None = None,
    last_image: Any | None = None,
    width: int | None = None,
    height: int | None = None,
) -> Asset:
    api_key = _require_api_key()
    instance: Dict[str, Any] = {"prompt": prompt}
    if first_image is not None:
        instance["image"] = {
            "bytesBase64Encoded": first_image.bytesBase64,
            "mimeType": (first_image.mime or "image/png").strip() or "image/png",
        }
    if last_image is not None:
        instance["lastFrame"] = {
            "bytesBase64Encoded": last_image.bytesBase64,
            "mimeType": (last_image.mime or "image/png").strip() or "image/png",
        }
    parameters: Dict[str, Any] = {"sampleCount": 1}
    ratio = _resolve_aspect_ratio(width, height)
    if ratio:
        parameters["aspectRatio"] = ratio

    start_url = f"{BASE}/models/{model}:predictLongRunning?" + urlencode({"key": api_key})
    op = _post_json(start_url, {"instances": [instance], "parameters": parameters}, timeout=120)
    op_name = op.get("name")
    if not isinstance(op_name, str) or not op_name:
        raise RuntimeError("Veo did not return an operation name")

    poll_url = f"{BASE}/{op_name}?" + urlencode({"key": api_key})
    deadline = time.monotonic() + VEO_POLL_TIMEOUT_S
    while True:
        status = _get_json(poll_url)
        if status.get("done"):
            if "error" in status:
                raise RuntimeError(f"Veo generation failed: {status['error']}")
            return _extract_veo_video(status.get("response") or {}, api_key)
        if time.monotonic() > deadline:
            raise RuntimeError("Veo generation timed out")
        time.sleep(VEO_POLL_INTERVAL_S)


def _extract_veo_video(response: Dict[str, Any], api_key: str) -> Asset:
    # Response shape: response.generateVideoResponse.generatedSamples[].video.{uri|bytesBase64Encoded}
    gvr = response.get("generateVideoResponse") or response
    samples = gvr.get("generatedSamples") or gvr.get("generated_samples") or []
    for sample in samples:
        video = (sample or {}).get("video") or {}
        b64 = video.get("bytesBase64Encoded") or video.get("bytes_base64_encoded")
        if isinstance(b64, str) and b64:
            return asset(base64.b64decode(b64), mime="video/mp4")
        uri = video.get("uri")
        if isinstance(uri, str) and uri:
            dl = uri if "key=" in uri else uri + ("&" if "?" in uri else "?") + urlencode({"key": api_key})
            try:
                resp = urlopen(dl, timeout=300)  # noqa: S310
            except (HTTPError, URLError) as e:
                raise RuntimeError(f"Failed to download Veo video: {e}") from e
            return asset(resp.read(), mime="video/mp4")
    raise RuntimeError("Veo response contained no video sample")


# ── Text slots ──────────────────────────────────────────────────────────────


@node_slot(NodeSlots.GEN_TEXT)
def gen_text(input: GenTextInput) -> GenTextOutput:
    user_message = (
        f"{input.userPrompt or ''}\n\nUser input: {input.text}\n\n"
        "Note: output only the requested answer. Do not include any other content."
    )
    answer = _generate_content(
        model=_active_model("gen-text", _env("GEMINI_MODEL")), user_message=user_message
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
    raw = _generate_content(
        model=_active_model("split-text", _env("GEMINI_MODEL")),
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
    answer = _generate_content(
        model=_active_model("combine-text", _env("GEMINI_MODEL")), user_message=user_message
    )
    return CombineTextOutput(success=True, text=answer)


# ── Vision / audio / document → text slots ──────────────────────────────────


@node_slot(NodeSlots.IMAGE_GEN_TEXT)
def image_gen_text(input: ImageGenTextInput) -> ImageGenTextOutput:
    user_message = ((input.system or "") + "\n\n" + input.text).strip() if input.system else input.text
    extra = [_inline_part(input.image, default_mime="image/png")] if input.image is not None else None
    text = _generate_content(
        model=_active_model("image-gen-text", _env("GEMINI_MODEL")),
        user_message=user_message,
        extra_parts=extra,
    )
    return ImageGenTextOutput(success=True, text=text)


@node_slot(NodeSlots.IMAGE_DESCRIBE)
def image_describe(input: ImageDescribeInput) -> ImageDescribeOutput:
    instruction = (
        (input.userPrompt or "").strip() or (input.text or "").strip() or "Describe this image in detail."
    )
    text = _generate_content(
        model=_active_model("image-describe", _env("GEMINI_MODEL")),
        user_message=instruction,
        extra_parts=[_inline_part(input.image, default_mime="image/png")],
    )
    return ImageDescribeOutput(success=True, text=text)


@node_slot(NodeSlots.VIDEO_GEN_TEXT)
def video_gen_text(input: VideoGenTextInput) -> VideoGenTextOutput:
    text = _generate_content(
        model=_active_model("video-gen-text", _env("GEMINI_MODEL")),
        user_message=input.text,
        extra_parts=[_inline_part(input.video, default_mime="video/mp4")],
    )
    return VideoGenTextOutput(success=True, text=text)


@node_slot(NodeSlots.VIDEO_DESCRIBE)
def video_describe(input: VideoDescribeInput) -> VideoDescribeOutput:
    instruction = (input.text or "").strip() or "Describe this video in detail."
    text = _generate_content(
        model=_active_model("video-describe", _env("GEMINI_MODEL")),
        user_message=instruction,
        extra_parts=[_inline_part(input.video, default_mime="video/mp4")],
    )
    return VideoDescribeOutput(success=True, text=text)


@node_slot(NodeSlots.AUDIO_DESCRIBE)
def audio_describe(input: AudioDescribeInput) -> AudioDescribeOutput:
    instruction = (
        (input.userPrompt or "").strip()
        or (input.text or "").strip()
        or "Describe this audio in detail (genre, mood, instruments, vocals, notable events)."
    )
    text = _generate_content(
        model=_active_model("audio-describe", _env("GEMINI_MODEL")),
        user_message=instruction,
        extra_parts=[_inline_part(input.audio, default_mime="audio/wav")],
    )
    return AudioDescribeOutput(success=True, text=text)


@node_slot(NodeSlots.PARSE_DOCUMENT)
def parse_document(input: ParseDocumentInput) -> ParseDocumentOutput:
    text = _generate_content(
        model=_active_model("parse-document", _env("GEMINI_MODEL")),
        user_message="Extract all text from this document, preserving reading order. Output only the extracted text.",
        extra_parts=[_inline_part(input.document, default_mime="application/pdf")],
    )
    return ParseDocumentOutput(success=True, text=text)


@node_slot(NodeSlots.TRANSCRIBE)
def transcribe(input: TranscribeInput) -> TranscribeOutput:
    hint = f" The audio language is {input.language}." if input.language else ""
    text = _generate_content(
        model=_active_model("transcribe", _env("GEMINI_MODEL")),
        user_message=f"Transcribe this audio verbatim. Output only the transcription text.{hint}",
        extra_parts=[_inline_part(input.audio, default_mime="audio/wav")],
    )
    return TranscribeOutput(success=True, text=text)


@node_slot(NodeSlots.TRANSCRIBE_TIMESTAMP)
def transcribe_timestamp(input: TranscribeTimestampInput) -> TranscribeTimestampOutput:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "start": {"type": "number"},
                "end": {"type": "number"},
                "text": {"type": "string"},
            },
            "required": ["start", "end", "text"],
        },
    }
    raw = _generate_content(
        model=_active_model("transcribe-timestamp", _env("GEMINI_MODEL")),
        user_message=(
            "Transcribe this audio into timestamped segments. Return a JSON array of "
            '{"start": seconds, "end": seconds, "text": "..."} objects covering the whole clip.'
        ),
        extra_parts=[_inline_part(input.audio, default_mime="audio/wav")],
        response_schema=schema,
    )
    try:
        segments = json.loads(raw)
    except json.JSONDecodeError as e:
        return TranscribeTimestampOutput(success=False, error=f"bad timestamp JSON: {e}")
    stamps: List[TranscribeTimestampOutputRootTimeStampsItem] = []
    text_pieces: list[str] = []
    for seg in segments if isinstance(segments, list) else []:
        if not isinstance(seg, dict) or not isinstance(seg.get("text"), str):
            continue
        stamps.append(
            TranscribeTimestampOutputRootTimeStampsItem(
                start=float(seg.get("start") or 0.0),
                end=float(seg.get("end") or 0.0),
                text=seg["text"].strip(),
            )
        )
        text_pieces.append(seg["text"].strip())
    return TranscribeTimestampOutput(success=True, text=" ".join(text_pieces), time_stamps=stamps)


# ── Image slots ─────────────────────────────────────────────────────────────


@node_slot(NodeSlots.IMAGE_GEN)
def image_gen(input: ImageGenInput) -> ImageGenOutput:
    prompt = (input.text or "").strip()
    if not prompt:
        return ImageGenOutput(success=False, error="image-gen requires a text prompt")
    image = _generate_image(
        model=_active_model("image-gen", _env("GEMINI_IMAGE_MODEL")),
        prompt=prompt,
        images=[],
        width=input.width,
        height=input.height,
    )
    return ImageGenOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_EDIT)
def image_edit(input: ImageEditInput) -> ImageEditOutput:
    image = _generate_image(
        model=_active_model("image-edit", _env("GEMINI_IMAGE_MODEL")),
        prompt=input.text,
        images=[prompt_media_to_bytes(input.image)],
        width=input.width,
        height=input.height,
    )
    return ImageEditOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_FUSION)
def image_fusion(input: ImageFusionInput) -> ImageFusionOutput:
    images = [prompt_media_to_bytes(x) for x in (input.images or [])]
    if not images:
        return ImageFusionOutput(success=False, error="image-fusion requires at least one input image")
    image = _generate_image(
        model=_active_model("image-fusion", _env("GEMINI_IMAGE_MODEL")),
        prompt=input.text,
        images=images,
        width=input.width,
        height=input.height,
    )
    return ImageFusionOutput(success=True, image=image)


# ── TTS slots ───────────────────────────────────────────────────────────────


@node_slot(NodeSlots.TEXT_GEN_SPEECH_PRESET)
def text_gen_speech_preset(input: TextGenSpeechPresetInput) -> TextGenSpeechPresetOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenSpeechPresetOutput(success=False, error="Missing input text")
    voice = (input.speaker or "").strip() or _env("GEMINI_TTS_VOICE") or DEFAULT_TTS_VOICE
    audio = _synthesize_speech(
        model=_active_model("text-gen-speech-preset", _env("GEMINI_TTS_MODEL")),
        text=text,
        voice=voice,
        style=(input.instruct or "").strip() or None,
    )
    return TextGenSpeechPresetOutput(success=True, audio=audio)


@node_slot(NodeSlots.TEXT_GEN_SPEECH_INSTRUCT)
def text_gen_speech_instruct(input: TextGenSpeechInstructInput) -> TextGenSpeechInstructOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenSpeechInstructOutput(success=False, error="Missing input text")
    voice = _env("GEMINI_TTS_VOICE") or DEFAULT_TTS_VOICE
    audio = _synthesize_speech(
        model=_active_model("text-gen-speech-instruct", _env("GEMINI_TTS_MODEL")),
        text=text,
        voice=voice,
        style=(input.instruct or "").strip() or None,
    )
    return TextGenSpeechInstructOutput(success=True, audio=audio)


# ── Veo video slots ─────────────────────────────────────────────────────────


@node_slot(NodeSlots.TEXT_GEN_VIDEO)
def text_gen_video(input: TextGenVideoInput) -> TextGenVideoOutput:
    video = _veo_generate(
        model=_active_model("text-gen-video", _env("GEMINI_VIDEO_MODEL")),
        prompt=input.text,
        width=input.width,
        height=input.height,
    )
    return TextGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGE_GEN_VIDEO)
def image_gen_video(input: ImageGenVideoInput) -> ImageGenVideoOutput:
    video = _veo_generate(
        model=_active_model("image-gen-video", _env("GEMINI_VIDEO_MODEL")),
        prompt=input.text,
        first_image=input.image,
        width=input.width,
        height=input.height,
    )
    return ImageGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGES_GEN_VIDEO)
def images_gen_video(input: ImagesGenVideoInput) -> ImagesGenVideoOutput:
    first = (input.images or [None])[0]
    video = _veo_generate(
        model=_active_model("images-gen-video", _env("GEMINI_VIDEO_MODEL")),
        prompt=input.text,
        first_image=first,
        width=input.width,
        height=input.height,
    )
    return ImagesGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGE_IMAGE_GEN_VIDEO)
def image_image_gen_video(input: ImageImageGenVideoInput) -> ImageImageGenVideoOutput:
    video = _veo_generate(
        model=_active_model("image-image-gen-video", _env("GEMINI_VIDEO_MODEL")),
        prompt=input.text,
        first_image=input.image,
        last_image=input.end_image,
        width=input.width,
        height=input.height,
    )
    return ImageImageGenVideoOutput(success=True, video=video)


# ── Deterministic batch slots (no model) ────────────────────────────────────


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
    NodeSlots.SPLIT_TEXT: split_text,
    NodeSlots.COMBINE_TEXT: combine_text,
    NodeSlots.IMAGE_GEN: image_gen,
    NodeSlots.IMAGE_EDIT: image_edit,
    NodeSlots.IMAGE_FUSION: image_fusion,
    NodeSlots.IMAGE_GEN_TEXT: image_gen_text,
    NodeSlots.IMAGE_DESCRIBE: image_describe,
    NodeSlots.VIDEO_GEN_TEXT: video_gen_text,
    NodeSlots.VIDEO_DESCRIBE: video_describe,
    NodeSlots.AUDIO_DESCRIBE: audio_describe,
    NodeSlots.PARSE_DOCUMENT: parse_document,
    NodeSlots.TRANSCRIBE: transcribe,
    NodeSlots.TRANSCRIBE_TIMESTAMP: transcribe_timestamp,
    NodeSlots.TEXT_GEN_SPEECH_PRESET: text_gen_speech_preset,
    NodeSlots.TEXT_GEN_SPEECH_INSTRUCT: text_gen_speech_instruct,
    NodeSlots.TEXT_GEN_VIDEO: text_gen_video,
    NodeSlots.IMAGE_GEN_VIDEO: image_gen_video,
    NodeSlots.IMAGES_GEN_VIDEO: images_gen_video,
    NodeSlots.IMAGE_IMAGE_GEN_VIDEO: image_image_gen_video,
    NodeSlots.DROP_VIDEO: drop_video,
    NodeSlots.ARRANGE_GROUP: arrange_group,
}


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    global _REQUEST_MODEL
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        prompt = req.get("prompt") if isinstance(req, dict) else {}
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "") if isinstance(req, dict) else ""
        _REQUEST_MODEL = str(req.get("model") or "").strip() if isinstance(req, dict) else ""

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
