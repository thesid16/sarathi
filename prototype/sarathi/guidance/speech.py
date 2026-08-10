"""Getting words and tones out of the device.

The policy here mirrors the frame pipeline: **drop, don't queue.** If the
system is still speaking when the next thing becomes worth saying, the new
utterance is discarded rather than queued behind it. Queued speech describes a
world the user has already walked through, and a backlog only grows - by the
third queued sentence the guidance is narrating the past. Silence is a valid
output; stale speech is not.

The exception is urgency. An urgent alert stops whatever is playing, sounds an
earcon and speaks. A rising tone reaches the user a few hundred milliseconds
before a spoken word can, and for something they are about to walk into, that
gap is the whole reason earcons exist.

Speaker backends are swappable so the prototype can run silently under test,
record what it would have said for benchmarking, or actually talk on a Mac.
Android supplies its own via the platform TextToSpeech engine.
"""

from __future__ import annotations

import abc
import math
import shutil
import struct
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from ..types import Urgency, Utterance
from ..util.log import get_logger

log = get_logger(__name__)


class Speaker(abc.ABC):
    """A text-to-speech backend. `speak` must not block."""

    @abc.abstractmethod
    def speak(self, text: str, lang: str = "en") -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @property
    @abc.abstractmethod
    def busy(self) -> bool: ...

    def close(self) -> None:
        self.stop()


class NullSpeaker(Speaker):
    """Says nothing. The default under test."""

    def speak(self, text: str, lang: str = "en") -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def busy(self) -> bool:
        return False


@dataclass
class Spoken:
    text: str
    lang: str
    urgency: Urgency
    at: float


class RecordingSpeaker(Speaker):
    """Records what would have been said.

    The backbone of evaluating guidance quality: replay a recorded walk, then
    inspect the transcript for utterance precision - of the things it said, how
    many were worth saying - without anyone having to listen in real time.
    """

    def __init__(self, speech_rate_cps: float = 14.0) -> None:
        self.spoken: list[Spoken] = []
        #: Characters per second, used to simulate how long speech occupies the
        #: channel. Without it every utterance appears instantaneous and the
        #: drop behaviour never triggers in tests.
        self.speech_rate_cps = speech_rate_cps
        self._free_at = 0.0
        self._now = 0.0

    def set_time(self, now: float) -> None:
        """Drive the simulated clock. Tests control time explicitly."""
        self._now = now

    def speak(self, text: str, lang: str = "en") -> None:
        self.spoken.append(Spoken(text, lang, Urgency.NORMAL, self._now))
        self._free_at = self._now + len(text) / self.speech_rate_cps

    def stop(self) -> None:
        self._free_at = self._now

    @property
    def busy(self) -> bool:
        return self._now < self._free_at

    @property
    def transcript(self) -> list[str]:
        return [s.text for s in self.spoken]


class MacSpeaker(Speaker):
    """macOS `say`. Prototype only - Android uses its own TextToSpeech."""

    #: Preferred voices per language, best first, falling back down the list.
    #:
    #: English defaults to Indian-English voices rather than US ones. The
    #: audience is Indian, and place names, road names and transliterated words
    #: come out far more intelligibly from Rishi than from Samantha - which
    #: matters more here than it would for a general-purpose app, because the
    #: user cannot glance at a screen to check what they misheard.
    VOICES = {
        "en": ["Rishi", "Aman", "Tara", "Samantha", "Daniel"],
        "hi": ["Lekha"],
    }

    def __init__(self, rate_wpm: int = 210) -> None:
        if shutil.which("say") is None:
            raise RuntimeError("the `say` binary is not available; not on macOS?")
        self.rate_wpm = rate_wpm
        self._proc: subprocess.Popen | None = None
        self._available = self._installed_voices()

    @staticmethod
    def _installed_voices() -> set[str]:
        try:
            out = subprocess.run(
                ["say", "-v", "?"], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return set()
        return {line.split()[0] for line in out.splitlines() if line.strip()}

    def voice_for(self, lang: str) -> str | None:
        for candidate in self.VOICES.get(lang, []):
            if candidate in self._available:
                return candidate
        return None

    def speak(self, text: str, lang: str = "en") -> None:
        self.stop()
        cmd = ["say", "-r", str(self.rate_wpm)]
        voice = self.voice_for(lang)
        if voice:
            cmd += ["-v", voice]
        elif lang != "en":
            # Devanagari read by an English voice is not accented Hindi, it is
            # unintelligible noise. Worth warning about rather than failing
            # quietly into gibberish.
            log.warning(
                "no installed voice for language %r; falling back to the system "
                "default, which will not read this text correctly",
                lang,
            )
        self._proc = subprocess.Popen([*cmd, text])

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    @property
    def busy(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# -- earcons -----------------------------------------------------------------


@dataclass
class EarconBank:
    """Short non-speech tones, synthesised once and cached on disk.

    Two of them, deliberately. A vocabulary of earcons has to be learned; two
    can be recognised immediately - a sharp double rise for something dangerous
    directly ahead, a single softer tone for everything else urgent.
    """

    sample_rate: int = 22050
    directory: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "sarathi-earcons")

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    SPECS = {
        # (start Hz, end Hz, seconds, repeats)
        "alert": (900.0, 1500.0, 0.09, 2),
        "warn": (600.0, 780.0, 0.13, 1),
    }

    def path(self, name: str) -> Path | None:
        if name not in self.SPECS:
            return None
        out = self.directory / f"{name}.wav"
        if not out.exists():
            self._synthesise(name, out)
        return out

    def _synthesise(self, name: str, out: Path) -> None:
        start_hz, end_hz, seconds, repeats = self.SPECS[name]
        frames: list[int] = []
        gap = [0] * int(self.sample_rate * 0.045)
        n = int(self.sample_rate * seconds)
        for r in range(repeats):
            if r:
                frames.extend(gap)
            phase = 0.0
            for i in range(n):
                t = i / n
                freq = start_hz + (end_hz - start_hz) * t
                phase += 2 * math.pi * freq / self.sample_rate
                # Short raised-cosine envelope: a tone that starts and stops
                # abruptly clicks, and a click is unpleasant on repeat.
                env = 0.5 - 0.5 * math.cos(2 * math.pi * min(t, 1.0))
                frames.append(int(22000 * env * math.sin(phase)))
        with wave.open(str(out), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(self.sample_rate)
            fh.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, f))) for f in frames))


class EarconPlayer:
    """Plays earcons, if the platform can."""

    def __init__(self, bank: EarconBank | None = None, enabled: bool = True) -> None:
        self.bank = bank or EarconBank()
        self._player = shutil.which("afplay") or shutil.which("aplay")
        self.enabled = enabled and self._player is not None
        self.played: list[str] = []

    def play(self, name: str) -> bool:
        self.played.append(name)
        if not self.enabled:
            return False
        path = self.bank.path(name)
        if path is None:
            return False
        try:
            subprocess.Popen(
                [self._player, str(path)],  # type: ignore[list-item]
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            log.debug("earcon playback failed: %s", exc)
            return False
        return True


# -- output ------------------------------------------------------------------


class VoiceOutput:
    """Speech and earcon playback, with the drop-don't-queue policy."""

    def __init__(
        self,
        speaker: Speaker | None = None,
        earcons: EarconPlayer | None = None,
    ) -> None:
        self.speaker = speaker or NullSpeaker()
        self.earcons = earcons
        #: Counters the benchmark harness reads. `dropped` is a quality signal,
        #: not just a stat: a high drop rate means the system is trying to say
        #: more than the channel can carry, and the saliency thresholds are
        #: too loose.
        self.spoken_count = 0
        self.dropped_count = 0
        self.interrupted_count = 0

    def say(self, utterance: Utterance, now: float | None = None) -> bool:
        """Speak it, or drop it. Returns True if it was spoken."""
        now = time.monotonic() if now is None else now
        if isinstance(self.speaker, RecordingSpeaker):
            self.speaker.set_time(now)

        if utterance.urgency is Urgency.URGENT:
            if self.speaker.busy:
                self.speaker.stop()
                self.interrupted_count += 1
            if utterance.earcon and self.earcons is not None:
                self.earcons.play(utterance.earcon)
            self.speaker.speak(utterance.text, utterance.lang)
            self.spoken_count += 1
            return True

        if self.speaker.busy:
            # Not queued. By the time it could play it would describe a scene
            # the user has walked past.
            self.dropped_count += 1
            log.debug("dropped while busy: %s", utterance.text)
            return False

        self.speaker.speak(utterance.text, utterance.lang)
        self.spoken_count += 1
        return True

    def stop(self) -> None:
        self.speaker.stop()

    def close(self) -> None:
        self.speaker.close()
