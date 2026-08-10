"""Gemma 4 scene description via LiteRT-LM.

The same runtime, the same version and the same weights file the Android app
uses, driven by the same manifest. That is the point: a description you can
read on a laptop is the description the phone will speak, so prompt changes can
be judged without a 2.5 GB push and a device in hand.

**Not on the safety path.** Hazards come from the detector and the geometric
estimator, which are bounded and measured. This is asked for explicitly, takes
seconds, and when it is wrong it is wrong in confident prose - the worst
possible failure mode for someone who cannot check it against what they see.
See `android/.../vlm/SceneDescriber.kt`, which enforces the same separation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...util.log import get_logger
from ..base import SceneDescriber, register_adapter
from ..manifest import ManifestError, ModelManifest, Task

log = get_logger(__name__)

#: Container section that must be present for image input to work.
VISION_SECTION = b"tf_lite_vision_encoder"

#: How much of the file to read looking for it. Section names live in the
#: header; 4 KB is generous.
_HEADER_BYTES = 4096


def has_vision_encoder(weights: Path) -> bool:
    """Whether this .litertlm build can actually see.

    A .litertlm is a container of named sections, and builds of the same model
    do not all carry the same ones. The `-gpu` build of Gemma 4 E2B holds
    exactly one section, ``tf_lite_artisan_text_decoder``, and no vision
    encoder - which its model card does not mention. Without this check the
    discovery happens eleven seconds into a request, as
    ``NOT_FOUND: TF_LITE_VISION_ENCODER``, after the whole model has loaded.

    Cheap enough to run before downloading, too::

        curl -sL -r 0-4096 <url> | strings | grep tf_lite
    """
    try:
        with weights.open("rb") as handle:
            return VISION_SECTION in handle.read(_HEADER_BYTES)
    except OSError:
        return False


class GemmaSceneDescriber(SceneDescriber):
    """On-demand scene description with a Gemma-family VLM."""

    #: Hard ceiling on generation. The system prompt asks for one sentence and
    #: a model may still write a paragraph; this bounds the wait as well as the
    #: length of speech that would sit in front of the next hazard warning.
    max_output_tokens = 64

    system_message = (
        "You describe scenes for a blind person who is standing where the "
        "camera is pointing. Answer in one short sentence, under 30 words. "
        "Lead with what is directly ahead and closest. Name things plainly. "
        "Give positions as left, ahead or right. Do not mention the image, "
        "the photo, or the camera. Do not guess at anything you cannot see. "
        "If the view is too dark or blurred to read, say exactly that."
    )

    def __init__(
        self,
        manifest: ModelManifest,
        weights: Path,
        labels: list[str] | None = None,
        *,
        backend: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(manifest, weights, labels)
        if not weights.exists():
            raise ManifestError(
                f"{manifest.id}: weights not found at {weights}. This model is "
                f"distribution={manifest.distribution.value}; fetch it with "
                f"`sarathi models fetch {manifest.id}`."
            )
        if not has_vision_encoder(weights):
            raise ManifestError(
                f"{manifest.id}: {weights.name} contains no {VISION_SECTION.decode()} "
                f"section, so it is a text-only build and cannot describe an image. "
                f"Check which build you have with: "
                f"head -c 4096 {weights.name} | strings | grep tf_lite"
            )

        self._weights = weights
        self._backend_name = backend
        self._cache_dir = cache_dir
        self._engine = None
        self.load_ms = 0.0
        self.last_describe_ms = 0.0

    # -- lifecycle ----------------------------------------------------------

    def _engine_or_load(self):
        if self._engine is not None:
            return self._engine

        import litert_lm

        backends = (
            [self._backend_name]
            if self._backend_name
            else ["GPU", "CPU"]  # GPU where it exists, and it often does not
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for name in backends:
            backend = getattr(litert_lm.Backend, name, None)
            if backend is None:
                continue
            try:
                self._engine = litert_lm.Engine(
                    model_path=str(self._weights),
                    backend=backend,
                    vision_backend=backend,
                    # One image, one question about right now. Declaring that
                    # sizes the KV cache and image buffers for what this app
                    # does rather than for a chat client's worst case.
                    max_num_images=1,
                    cache_dir=str(self._cache_dir) if self._cache_dir else None,
                )
                self.load_ms = (time.perf_counter() - started) * 1000.0
                log.info("%s ready on %s in %.0f ms", self.id, name, self.load_ms)
                return self._engine
            except Exception as exc:  # noqa: BLE001 - any backend failure is a fallback
                log.info("%s unavailable on %s: %s", self.id, name, exc)
                last_error = exc
        raise ManifestError(f"{self.id}: no usable backend ({last_error})")

    def warmup(self, runs: int = 1) -> None:
        """Load the engine without generating.

        Worth doing explicitly because it is the expensive half - eleven
        seconds on a Pixel 8a - and leaving it inside the first user-visible
        request makes that request look like the model is slow.
        """
        self._engine_or_load()

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.close()
            except Exception:  # noqa: BLE001 - closing must not raise
                pass

    # -- inference ----------------------------------------------------------

    def describe(self, image: np.ndarray, prompt: str | None = None) -> str:
        import cv2
        import litert_lm

        engine = self._engine_or_load()
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise RuntimeError("failed to encode frame as JPEG")

        started = time.perf_counter()
        conversation = engine.create_conversation(
            system_message=self.system_message,
            max_output_tokens=self.max_output_tokens,
        )
        try:
            reply = conversation.send_message(
                litert_lm.Contents.of(
                    litert_lm.Content.ImageBytes(encoded.tobytes()),
                    litert_lm.Content.Text(prompt or "What is in front of me right now?"),
                )
            )
        finally:
            conversation.close()
        self.last_describe_ms = (time.perf_counter() - started) * 1000.0
        return tidy(_text_of(reply))


def _text_of(reply: object) -> str:
    """Pull the text out of whatever shape the reply arrives in.

    The Python binding returns a mapping and the Kotlin one returns a Message;
    both wrap a list of content parts. Reaching through defensively keeps this
    adapter working across a runtime that is still moving quickly, rather than
    breaking on a key rename.
    """
    if isinstance(reply, str):
        return reply
    if isinstance(reply, dict):
        contents = reply.get("contents") or reply.get("content") or []
        if isinstance(contents, str):
            return contents
        parts = []
        for item in contents if isinstance(contents, list) else [contents]:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        joined = "".join(parts).strip()
        if joined:
            return joined
        return str(reply.get("text", "")).strip()
    return str(getattr(reply, "text", reply)).strip()


#: Openers an instruction-tuned model reaches for, which cost a blind listener
#: a second of speech to learn nothing.
_OPENERS = (
    "the image shows",
    "this image shows",
    "the picture shows",
    "in this image,",
    "in the image,",
    "the photo shows",
    "i see",
    "here is",
    "this is a picture of",
    "this appears to be",
)

#: Roughly twelve seconds of speech. Past that it stops being an answer, and it
#: occupies the audio channel long enough to displace a real hazard warning.
MAX_CHARS = 220


def tidy(raw: str) -> str:
    """Trim a model's answer into something worth speaking.

    Kept module-level and pure so it can be tested without a 2.5 GB model, and
    so the Kotlin implementation has something to be checked against.
    """
    text = raw.strip()
    for character in "*_#`":
        text = text.replace(character, "")
    lowered = text.lower()
    for opener in _OPENERS:
        if lowered.startswith(opener):
            text = text[len(opener) :].lstrip()
            text = text[:1].upper() + text[1:] if text else text
            break
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(first) > MAX_CHARS:
        first = first[:MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return first


register_adapter(Task.VLM, "litert-lm", GemmaSceneDescriber)
register_adapter(Task.VLM, "litertlm", GemmaSceneDescriber)
