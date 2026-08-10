# Text reading

> Hold volume-up and the phone reads any text in front of it.

A door number, a bus route, a platform sign, a packet of medicine. All things a
blind person needs routinely, none of which a bounding-box detector can supply,
and none of which is a hazard. So this sits beside scene description: off the
safety path, on request only, and unable to make the app less safe by failing.

## Measured

Pixel 8a, ML Kit Latin recogniser, on the bundled self-test sign:

| | |
|---|---:|
| Recognition | **720 ms** (first call), 417 ms warm |
| Blocks found | 4 of 4 |
| Model in APK | **none** — Play Services delivers it |
| APK cost of the dependency | ~2 MB of client stubs |

```
$ adb logcat -s SarathiOCR
read 4 blocks in 720ms (LATIN):
  PLATFORM 3. Exit via stairs. Lift out of order. Room 214
```

## Which recogniser, and the licence question

This is the least clean dependency in Sarathi, so it gets stated plainly rather
than buried.

Sarathi is AGPL-3.0 and has already turned down a model over licence terms.
ML Kit is proprietary. What makes it defensible here is that the artifacts used
are the **unbundled** Play Services variants: no proprietary model ships inside
the APK, and recognition happens in Google Play Services — the same place the
`TextToSpeech` engine that speaks every word of this app already lives. That is
the basis for treating it the way AGPL-3.0 treats a System Library.

It is a defensible reading, not an airtight one, so nothing depends on it:

- Text reading is **optional**. Without Play Services, `read()` returns null and
  the rest of the app is unaffected. A de-Googled build simply has no OCR.
- `rapidocr-mobile` remains in the model registry with Apache-2.0
  PaddleOCR-derived weights. It is the fully open path, it runs in the
  prototype today, and a fork can take it to Android by writing one adapter.

### The comparison that decided it

Same synthetic sign, both recognisers:

| | RapidOCR (Apache-2.0) | ML Kit (Play Services) |
|---|---|---|
| Time | **148 ms** (M3 laptop) | 720 ms (Pixel 8a) |
| `PLATFORM 3` | ❌ read as `PLATFORM 13` | ✅ |
| `Exit via stairs` | ✅ | ✅ |
| `Lift out of order` | ✅ | ✅ |
| `Room 214` | ❌ read as `Ro0m 214` | ✅ |

```bash
.venv/bin/python -m sarathi.cli read android/app/src/main/assets/models/ocr-selftest.jpg
```

**Caveat, and it is a large one:** this is one synthetic image on two different
machines, not a benchmark. It does not establish that ML Kit is generally more
accurate.

What it does establish is that the errors are not harmless. `PLATFORM 13` for
`PLATFORM 3` sends someone to the wrong platform, and they have no way to
notice. A digit hallucinated into a door number or a dosage is the same class of
failure. Against that, the open alternative is a port of DBNet detection plus a
CRNN/CTC recogniser and a character dictionary — several hundred lines of
numerical code whose failure mode is exactly this, shipped half-validated.

The trade was made knowingly in that direction, and it is recorded here so a
fork with different priorities can make it differently.

## Two scripts, chosen explicitly

Devanagari is a **separate model**, not an option on the Latin one. Point the
Latin recogniser at a Hindi sign and it returns nothing — which is
indistinguishable from "there is no text here". So the recogniser is selected
from the user's language rather than left to a default, and both artifacts are
declared.

The Devanagari path is **unverified**: no Hindi sign has been read on the
device. The code path is identical and the artifact is declared, but that is not
the same as evidence.

## Reading order

ML Kit returns blocks roughly in the order it found them, which for a
photograph of a sign is not the order a person reads. Blocks are sorted into
vertical bands first and by horizontal position second, so rows survive: two
labels side by side stay side by side, a caption underneath stays underneath.

The band tolerance is 60% of the mean block height. Sorting on raw `top`
interleaves text that is visually on one line but a few pixels out of
alignment, which is unintelligible read aloud.

Output is capped at 400 characters — about twenty seconds of speech. A dense
sign carries far more text than anyone wants read at them, and the audio
channel is shared with hazard warnings.

## The self-test

```
ocr self-test (LATIN): PLATFORM 3. Exit via stairs. Lift out of order. Room 214 in 720ms
```

"0 blocks" is ambiguous in exactly the way a detection count of zero is: the
view may hold no text, or Play Services may never have delivered the model, or
the wrong script may be selected. On a device operated with the screen off
there is no other way to tell those apart, and the difference decides whether
the user should point the camera elsewhere or stop trying.

So the app reads an image whose answer is known, at startup, and says so. Same
pattern as `LiteRtDetector.selfTest`, for the same reason.

## Controls

| | |
|---|---|
| volume-down | start / stop guidance |
| volume-up, tapped | describe the scene |
| volume-up, **held** | read any text |

Both on-demand features share one button because there are only two buttons,
and tap-versus-hold can be told apart by feel without looking. The tap fires on
key-*up*: committing on the way down would make every long press trigger a
description first.
