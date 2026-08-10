"""MJPEG-over-HTTP source, written for the ESP32-CAM.

OpenCV *can* open an MJPEG URL, but it goes through FFmpeg and buffers, and it
gives no visibility into what the stream is actually doing. Since the ESP32 is
the weakest link in the whole chain - it browns out, it stalls, it silently
drops to a lower frame rate when it heats up - this parser is hand-rolled so
those failures are observable rather than showing up as a mysteriously frozen
picture.

The wire format Espressif's `camera_httpd` example produces:

    HTTP/1.1 200 OK
    Content-Type: multipart/x-mixed-replace;boundary=123456789000000000000987654321

    --123456789000000000000987654321
    Content-Type: image/jpeg
    Content-Length: 12345
    X-Timestamp: 1234.567890

    <12345 bytes of JPEG>
    --123456789000000000000987654321
    ...

Content-Length is present in stock firmware but not in every fork, so there is
a fallback that scans for the JPEG start/end markers.
"""

from __future__ import annotations

import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from ..types import Frame, SourceInfo
from ..util.log import get_logger
from .base import FrameSource, SourceError

log = get_logger(__name__)

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_BOUNDARY_RE = re.compile(rb"boundary=(?:\"([^\"]+)\"|([^\s;]+))", re.IGNORECASE)

# If a single frame's payload exceeds this, the stream is desynchronised and we
# are scanning through garbage. Bail out and let the reconnect logic recover
# rather than growing the buffer without bound.
_MAX_FRAME_BYTES = 8 * 1024 * 1024


class MjpegSource(FrameSource):
    """Reads `multipart/x-mixed-replace` JPEG frames straight off the socket."""

    kind = "mjpeg"

    def __init__(
        self,
        source_id: str,
        url: str,
        *,
        timeout_s: float = 5.0,
        chunk_size: int = 16384,
    ) -> None:
        super().__init__(source_id)
        self._url = url
        self._timeout_s = timeout_s
        self._chunk_size = chunk_size
        self._resp: Any = None
        self._buf = bytearray()
        self._boundary: bytes | None = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> SourceInfo:
        req = urllib.request.Request(self._url, headers={"User-Agent": "sarathi/0.1"})
        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout_s)
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            raise SourceError(f"cannot open MJPEG stream {self._url}: {exc}") from exc

        content_type = resp.headers.get("Content-Type", "")
        match = _BOUNDARY_RE.search(content_type.encode())
        if match:
            raw = match.group(1) or match.group(2)
            self._boundary = b"--" + raw
        else:
            # Some minimal firmwares omit the boundary parameter. Marker
            # scanning still works without it.
            self._boundary = None
            log.warning("no MJPEG boundary in Content-Type %r; falling back to marker scan",
                        content_type)

        self._resp = resp
        self._buf.clear()

        # Resolution is unknown until the first frame arrives, so it is filled
        # in lazily on the first successful grab.
        self._info = SourceInfo(
            source_id=self.source_id,
            kind=self.kind,
            width=0,
            height=0,
            nominal_fps=0.0,
            is_networked=True,
            detail={"url": self._url, "boundary": (self._boundary or b"").decode(errors="replace")},
        )
        log.info("opened MJPEG stream %s", self._url)
        return self._info

    def close(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:  # noqa: BLE001 - closing a dead socket is fine
                pass
            self._resp = None
        self._buf.clear()

    # -- reading -----------------------------------------------------------

    def grab(self) -> Frame | None:
        if self._resp is None:
            raise SourceError("grab() before open()")

        payload, headers = self._read_part()
        received = time.monotonic()

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            # A single corrupt JPEG is normal on a flaky WiFi link. Skip it
            # rather than tearing down the connection.
            log.debug("dropped undecodable JPEG (%d bytes)", len(payload))
            return self.grab()

        if self._info is not None and self._info.width == 0:
            self._info = SourceInfo(
                source_id=self._info.source_id,
                kind=self._info.kind,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                nominal_fps=self._info.nominal_fps,
                is_networked=True,
                detail=self._info.detail,
            )

        meta: dict[str, Any] = {"bytes": len(payload)}
        # The ESP32's own clock is not synchronised with ours, so this is
        # recorded for drift analysis only - it must not be used as the
        # capture time or every latency number becomes fiction.
        if "X-Timestamp" in headers:
            meta["device_timestamp"] = headers["X-Timestamp"]

        return Frame(
            image=image,
            seq=self._next_seq(),
            ts_capture=received,
            ts_received=received,
            source_id=self.source_id,
            meta=meta,
        )

    # -- multipart parsing -------------------------------------------------

    def _fill(self) -> None:
        """Pull one chunk from the socket into the buffer."""
        try:
            chunk = self._resp.read(self._chunk_size)
        except (socket.timeout, OSError) as exc:
            raise SourceError(f"MJPEG read failed: {exc}") from exc
        if not chunk:
            raise SourceError("MJPEG stream closed by peer")
        self._buf.extend(chunk)
        if len(self._buf) > _MAX_FRAME_BYTES:
            raise SourceError("MJPEG buffer overrun - stream desynchronised")

    def _read_until(self, needle: bytes, start: int = 0) -> int:
        """Return the index of `needle`, reading more data until it appears."""
        while True:
            idx = self._buf.find(needle, start)
            if idx >= 0:
                return idx
            # Everything before the last len(needle)-1 bytes has been searched.
            start = max(0, len(self._buf) - len(needle) + 1)
            self._fill()

    def _read_part(self) -> tuple[bytes, dict[str, str]]:
        """Read one complete multipart section and return (jpeg_bytes, headers)."""
        if self._boundary is not None:
            idx = self._read_until(self._boundary)
            del self._buf[: idx + len(self._boundary)]

            header_end = self._read_until(b"\r\n\r\n")
            header_blob = bytes(self._buf[:header_end])
            del self._buf[: header_end + 4]

            headers: dict[str, str] = {}
            for line in header_blob.split(b"\r\n"):
                if b":" in line:
                    key, _, value = line.partition(b":")
                    headers[key.strip().decode(errors="replace")] = value.strip().decode(
                        errors="replace"
                    )

            length = headers.get("Content-Length")
            if length and length.isdigit():
                need = int(length)
                if need > _MAX_FRAME_BYTES:
                    raise SourceError(f"implausible Content-Length {need}")
                while len(self._buf) < need:
                    self._fill()
                payload = bytes(self._buf[:need])
                del self._buf[:need]
                return payload, headers
            # No Content-Length - fall through to marker scanning.
            return self._scan_markers(), headers

        return self._scan_markers(), {}

    def _scan_markers(self) -> bytes:
        """Extract one JPEG by locating its SOI and EOI markers."""
        soi = self._read_until(_JPEG_SOI)
        if soi:
            del self._buf[:soi]
        eoi = self._read_until(_JPEG_EOI, start=2)
        end = eoi + len(_JPEG_EOI)
        payload = bytes(self._buf[:end])
        del self._buf[:end]
        return payload
