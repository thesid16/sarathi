"""Tests for phrasing and speech output.

The real phrase tables are loaded, not fixtures - they ship with the product
and are read by the Android app too, so a broken table is a broken release.

Most of these assert product behaviour rather than code behaviour: urgent
alerts must be short, distances must be rounded to something worth hearing,
and speech must be dropped rather than queued when the channel is busy.
"""

from __future__ import annotations

import wave

import pytest

from sarathi.guidance import (
    EarconBank,
    EarconPlayer,
    NullSpeaker,
    PhraseBook,
    PhraseError,
    Phraser,
    RecordingSpeaker,
    SaliencyEngine,
    VoiceOutput,
)
from sarathi.perception.tracking import Tracker
from sarathi.types import Detection, Hazard, Urgency, Utterance

ENGINE = SaliencyEngine()


def build(label, distance, bearing, hazard):
    tracker = Tracker(min_hits=1)
    for i in range(4):
        tracker.update(
            [Detection((0, 0, 20, 20), 0.9, 0, label, distance_m=distance,
                       bearing_deg=bearing, hazard=hazard)],
            i * 0.3,
        )
    return tracker.tracks[-1]


def ranked(label, distance, bearing, hazard):
    return ENGINE.score(build(label, distance, bearing, hazard))


# -- phrase tables -----------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_both_shipped_tables_load(lang):
    book = PhraseBook.load(lang)
    assert book.lang == lang
    assert book.templates and book.distance["steps"]


def test_missing_language_is_a_clear_error():
    with pytest.raises(PhraseError, match="no phrase table"):
        PhraseBook.load("kl")


def test_a_table_without_distance_steps_is_rejected(tmp_path):
    (tmp_path / "xx.yaml").write_text(
        "lang: xx\ntemplates: {full: '{object}'}\nbearing: {}\ndistance: {}\n"
    )
    with pytest.raises(PhraseError, match="distance.steps"):
        PhraseBook.load("xx", tmp_path)


def test_spoken_names_override_class_labels():
    """'stairs_down' is a label; 'step down' is what a person needs to hear."""
    book = PhraseBook.load("en")
    assert book.object_name("stairs_down") == "step down"
    assert book.object_name("open_manhole") == "open manhole"


def test_unknown_labels_fall_back_to_readable_text():
    assert PhraseBook.load("en").object_name("some_new_class") == "some new class"


# -- bearing -----------------------------------------------------------------


def test_straight_ahead_is_said_as_ahead_not_twelve_oclock():
    book = PhraseBook.load("en")
    assert book.bearing_phrase(0.0) == "ahead"
    assert book.bearing_phrase(5.0) == "ahead"


@pytest.mark.parametrize(("bearing", "expected"), [
    (30, "one o'clock"), (60, "two o'clock"), (-30, "eleven o'clock"), (-60, "ten o'clock"),
])
def test_clock_bearings(bearing, expected):
    assert PhraseBook.load("en").bearing_phrase(bearing) == expected


def test_hindi_uses_relative_directions_by_default():
    """Clock bearings in Hindi come out long and only work if you were taught them."""
    book = PhraseBook.load("hi")
    assert book.bearing_style == "relative"
    assert book.bearing_phrase(0.0) == "सामने"
    assert book.bearing_phrase(20.0) == "थोड़ा दाएँ"
    assert book.bearing_phrase(60.0) == "दाईं ओर"
    assert book.bearing_phrase(-60.0) == "बाईं ओर"


def test_bearing_style_is_switchable_not_hardcoded_per_language():
    book = PhraseBook.load("hi")
    book.bearing_style = "clock"
    assert book.bearing_phrase(30.0) == "एक बजे"


def test_missing_bearing_is_treated_as_ahead():
    assert PhraseBook.load("en").bearing_phrase(None) == "ahead"


# -- distance ----------------------------------------------------------------


@pytest.mark.parametrize(("metres", "expected"), [
    (0.4, "half a metre"), (1.0, "one metre"), (1.5, "one and a half metres"),
    (2.2, "two metres"), (3.0, "three metres"), (5.2, "five metres"), (40.0, "six metres"),
])
def test_distance_is_rounded_to_something_worth_hearing(metres, expected):
    assert PhraseBook.load("en").distance_phrase(metres) == expected


def test_hindi_uses_the_single_word_for_one_and_a_half():
    assert PhraseBook.load("hi").distance_phrase(1.5) == "डेढ़ मीटर"


def test_low_confidence_distances_are_hedged():
    book = PhraseBook.load("en")
    assert book.distance_phrase(3.0, uncertainty=0.05) == "three metres"
    assert book.distance_phrase(3.0, uncertainty=0.5) == "about three metres"


def test_no_distance_yields_no_words():
    assert PhraseBook.load("en").distance_phrase(None) == ""


# -- utterances --------------------------------------------------------------


def test_urgent_alerts_drop_the_distance():
    """At urgent range the user needs to stop, not to hear a measurement."""
    utt = Phraser(lang="en").utterance(ranked("open_manhole", 1.5, 0.0, Hazard.CRITICAL))
    assert utt.urgency is Urgency.URGENT
    assert utt.text == "open manhole ahead"
    assert "metre" not in utt.text


def test_urgent_alerts_are_shorter_than_normal_ones():
    phraser = Phraser(lang="en")
    urgent = phraser.utterance(ranked("stairs_down", 1.5, 0.0, Hazard.CRITICAL))
    normal = phraser.utterance(ranked("chair", 2.0, 40.0, Hazard.MEDIUM))
    assert len(urgent.text) < len(normal.text)


def test_a_normal_announcement_names_object_bearing_and_distance():
    utt = Phraser(lang="en").utterance(ranked("chair", 1.6, 35.0, Hazard.HIGH))
    assert utt.text == "chair, one o'clock, one and a half metres"


def test_something_straight_ahead_reads_naturally():
    utt = Phraser(lang="en").utterance(ranked("car", 5.0, 2.0, Hazard.CRITICAL))
    assert utt.text == "car ahead, five metres"


def test_hindi_utterance_is_devanagari_and_well_formed():
    utt = Phraser(lang="hi").utterance(ranked("chair", 1.6, 35.0, Hazard.HIGH))
    assert utt.text == "कुर्सी, दाईं ओर, डेढ़ मीटर"
    assert utt.lang == "hi"


def test_topic_is_keyed_on_the_object_not_the_sentence():
    """So a chair ticking from 'two metres' to 'one and a half' is one subject."""
    phraser = Phraser(lang="en")
    near = phraser.utterance(ranked("chair", 1.4, 0.0, Hazard.HIGH))
    nearer = phraser.utterance(ranked("chair", 1.1, 0.0, Hazard.HIGH))
    assert near.text != nearer.text
    assert near.topic.startswith("chair#") and nearer.topic.startswith("chair#")


def test_critical_hazards_get_the_sharper_earcon():
    phraser = Phraser(lang="en")
    assert phraser.utterance(ranked("open_manhole", 1.2, 0.0, Hazard.CRITICAL)).earcon == "alert"
    assert phraser.utterance(ranked("chair", 3.0, 40.0, Hazard.MEDIUM)).earcon is None


def test_no_dangling_punctuation_when_a_slot_is_empty():
    utt = Phraser(lang="en").utterance(ranked("chair", None, 40.0, Hazard.HIGH))
    assert ",," not in utt.text
    assert not utt.text.endswith(",")
    assert "  " not in utt.text


def test_system_phrases():
    assert Phraser(lang="en").system("camera_lost").text == "Camera disconnected"
    assert Phraser(lang="hi").system("camera_lost").text == "कैमरा बंद"


# -- output policy -----------------------------------------------------------


def utt(text, urgency=Urgency.NORMAL, earcon=None):
    return Utterance(text=text, urgency=urgency, topic=text, earcon=earcon)


def test_speech_is_dropped_not_queued_while_busy():
    """Queued speech describes a world the user has already walked through."""
    speaker = RecordingSpeaker(speech_rate_cps=10.0)
    voice = VoiceOutput(speaker)
    assert voice.say(utt("chair, one o'clock, two metres"), now=0.0) is True
    assert voice.say(utt("bin ahead, three metres"), now=1.0) is False
    assert voice.dropped_count == 1
    assert speaker.transcript == ["chair, one o'clock, two metres"]


def test_speech_resumes_once_the_channel_is_free():
    speaker = RecordingSpeaker(speech_rate_cps=10.0)
    voice = VoiceOutput(speaker)
    voice.say(utt("chair ahead"), now=0.0)
    assert voice.say(utt("bin ahead"), now=10.0) is True
    assert len(speaker.transcript) == 2


def test_an_urgent_alert_interrupts_speech_in_progress():
    speaker = RecordingSpeaker(speech_rate_cps=10.0)
    voice = VoiceOutput(speaker)
    voice.say(utt("a long sentence about some furniture"), now=0.0)
    assert voice.say(utt("step down ahead", Urgency.URGENT), now=0.5) is True
    assert voice.interrupted_count == 1
    assert speaker.transcript[-1] == "step down ahead"


def test_an_urgent_alert_plays_its_earcon_first():
    earcons = EarconPlayer(enabled=False)  # records without making noise
    voice = VoiceOutput(RecordingSpeaker(), earcons)
    voice.say(utt("open manhole ahead", Urgency.URGENT, earcon="alert"), now=0.0)
    assert earcons.played == ["alert"]


def test_normal_utterances_do_not_play_earcons():
    earcons = EarconPlayer(enabled=False)
    voice = VoiceOutput(RecordingSpeaker(), earcons)
    voice.say(utt("chair ahead", Urgency.NORMAL), now=0.0)
    assert earcons.played == []


def test_counters_track_what_the_channel_could_carry():
    speaker = RecordingSpeaker(speech_rate_cps=5.0)
    voice = VoiceOutput(speaker)
    for i in range(5):
        voice.say(utt("something reasonably long to say"), now=i * 0.5)
    assert voice.spoken_count == 1
    assert voice.dropped_count == 4  # a signal that saliency is too permissive


def test_the_null_speaker_is_never_busy():
    voice = VoiceOutput(NullSpeaker())
    assert voice.say(utt("one"), now=0.0)
    assert voice.say(utt("two"), now=0.01)


# -- earcons -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["alert", "warn"])
def test_earcons_synthesise_to_playable_wav(tmp_path, name):
    path = EarconBank(directory=tmp_path).path(name)
    assert path is not None and path.exists()
    with wave.open(str(path), "rb") as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        duration = fh.getnframes() / fh.getframerate()
    # Long enough to hear, short enough not to delay the words behind it.
    assert 0.05 < duration < 0.5


def test_an_unknown_earcon_is_ignored_rather_than_raising(tmp_path):
    assert EarconBank(directory=tmp_path).path("fanfare") is None


def test_earcons_are_synthesised_once_and_cached(tmp_path):
    bank = EarconBank(directory=tmp_path)
    first = bank.path("alert")
    stamp = first.stat().st_mtime_ns
    assert bank.path("alert").stat().st_mtime_ns == stamp
