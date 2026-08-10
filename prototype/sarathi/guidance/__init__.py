"""Guidance: turning what was perceived into what gets said - and what does not."""

from .phrasing import PhraseBook, PhraseError, Phraser
from .saliency import Ranked, SaliencyConfig, SaliencyEngine
from .speech import (
    EarconBank,
    EarconPlayer,
    MacSpeaker,
    NullSpeaker,
    RecordingSpeaker,
    Speaker,
    VoiceOutput,
)

__all__ = [
    "EarconBank",
    "EarconPlayer",
    "MacSpeaker",
    "NullSpeaker",
    "PhraseBook",
    "PhraseError",
    "Phraser",
    "Ranked",
    "RecordingSpeaker",
    "SaliencyConfig",
    "SaliencyEngine",
    "Speaker",
    "VoiceOutput",
]
