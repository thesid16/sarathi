# Edit guide

How to change things, and what will bite you.

Organised by *what you want to do*, because that is the question people arrive
with. Every entry says which file, what to run afterwards, and what has gone
wrong there before.

---

## Before anything else: the one rule

**This codebase fails silently.** Every serious bug in its history ran without
an error and produced plausible output:

| What it did | What it looked like |
|---|---|
| Detector fed sideways frames | `0 detections` — same as an empty room |
| A wall clock measured at 54.3 m | correct label, correct count, nonsense number |
| Quantized model with a dead class head | fastest variant available, detected nothing |
| GPU delegate computing garbage | twice as fast as the CPU |
| Validation split at 79% instead of 15% | a model that validated beautifully |
| An inverted tilt sign | a confident, plausible `-28°` |

So the test that matters is rarely "did it crash". It is "does the number make
sense". Two habits catch most of it:

1. **Look at a picture.** `python -m sarathi.desktop` draws boxes on a live
   feed. A shear, a rotation or a bad scale is obvious in one glance and
   invisible in a log.
2. **Check the self-tests.** The app scores a bundled image with a known answer
   at every launch. `adb logcat -s SarathiService | grep self-test`. If it
   says `maxScore=1.000, detections=50` all one class, something is broken
   regardless of how healthy everything else looks.

---

## Set up

```bash
# Python side - prototype, desktop app, training, evaluation
cd prototype
uv venv && uv pip install -e .
.venv/bin/python -m pytest tests -q          # expect 406 passing

# Android side
cd android
JAVA_HOME=/path/to/jdk-17-or-newer ./gradlew assembleDebug
```

`uv` is the package manager; `pip` works too. The Android build needs JDK 17+
and an Android SDK with API 36.

---

## "I want to change what it says"

**File:** `phrases/en.yaml`, `phrases/hi.yaml`

Everything spoken lives here — object names, distance wording, bearings, system
messages, and the VLM's system instruction. No spoken string is hardcoded.

```yaml
distance:
  steps:
    - {max: 0.6, text: "one and a half feet"}   # max is METRES, text is feet
```

**The `max` thresholds are metres** even though the text is feet. The geometry
computes metres; converting the threshold instead of the maths would give two
sources of truth. Rungs are deliberately coarse — the estimate carries tens of
percent of error, so "eleven point four feet" would be false precision.

- **Afterwards:** `pytest tests/test_speech.py` — several tests assert exact
  strings and will fail loudly, which is the point.
- **Android picks it up automatically:** a Gradle task copies `phrases/` into
  assets at build time. Both sides read the same file; that is why they agree.
- **Watch out:** changing the language of the VLM's answer means changing
  `describe_system`, not just `describe_prompt`. Translating the question alone
  leaves the model answering in English — that bug shipped once.

## "I want to change when it speaks"

**File:** `prototype/sarathi/guidance/saliency.py` and
`android/.../guidance/Saliency.kt` — **both**, identically.

```python
score_floor = 0.55        # below this, silence
announce_low_hazard = False
repeat_cooldown_ms = 8000
corridor_half_width_m = 0.9
```

Ranking is `proximity × path`, multiplied not averaged, plus hazard and closing
bonuses. Multiplication is deliberate: averaging let a cup on a table nearly
clear the floor while a chair in the walking line sat under it.

- **Afterwards:** `pytest tests/test_tracking_saliency.py`, then
  `python -m sarathi.cli speak` for a scripted walk you can listen to.
- **Watch out:** the two implementations must agree. There is no test that
  compares them automatically — that is a known gap.

## "I want to add or swap a model"

**Files:** a new `models/manifests/<id>.yaml`, weights in `models/weights/`

Nothing about a model is in code. The manifest declares input size, colour
order, normalisation, decoder, labels, licence and distribution policy.

```yaml
id: my-detector-320
task: detection
license: Apache-2.0
distribution: bundled          # bundled | user_download | excluded
runtime:
  android: litert
  prototype: onnxruntime
files:
  tflite: { path: my-detector-320.tflite, sha256: ..., size_bytes: ... }
input:  { width: 320, height: 320, layout: NHWC, color: RGB, resize: letterbox, ... }
output: { decoder: yolo11, labels: coco80, conf_threshold: 0.35 }
```

To bundle it on Android, drop the `.tflite` into
`android/app/src/main/assets/models/`. It then appears in the app's **Model**
picker automatically — the picker enumerates manifests and filters to those
whose weights actually exist.

- **Afterwards:** `pytest tests/test_shipped_manifests.py` validates every
  shipped manifest and fails if one names a runtime it cannot resolve weights
  for. Then check the on-device self-test.
- **Watch out:** `distribution: excluded` is enforced in code — the registry
  refuses to load it. That is how CC-BY-NC weights are kept out.

### Exporting a YOLO checkpoint to TFLite

The flags matter and getting them wrong produces a model that loads fine and
detects nothing:

```bash
# 1. ONNX, from a venv with ultralytics + torch
python -c "from ultralytics import YOLO; \
  YOLO('best.pt').export(format='onnx', imgsz=320, opset=13, dynamic=False)"

# 2. TFLite, from a venv with onnx2tf + tensorflow. NO -qt flag.
python -m onnx2tf -i best.onnx -o out -osd -coion -b 1 -tb tf_converter -oiqt
#    take out/*_dynamic_range_quant.tflite
```

**Do not pass `-qt per-tensor`.** It collapses the classification head; the
model then returns `maxScore=1.000` with fifty boxes of one class. Three models
shipped that way once and only the on-device self-test caught it.

**Verify before it reaches a phone:**

```python
y = interpreter.get_tensor(out["index"])[0]
print("box rows max", y[:4].max(), "class max", y[4:].max())
# healthy: box rows ≈ input size (320), class max 0.6-0.9
# broken:  class max 1.0000 or 0.0000
```

Keep the two virtualenvs separate. Installing export dependencies into the
training venv once left torch and torchvision ABI-mismatched and blocked every
evaluation.

## "I want to change how far away things are"

**Files:** `prototype/sarathi/perception/distance.py`,
`android/.../perception/Geometry.kt` — **both, identically.**

Distance comes from two independent estimates that are fused when they agree:
the ground-plane contact point, and a size prior. Priors live in
`training/taxonomy/size_priors.yaml`:

```yaml
priors:
  person: { h_m: 1.65, spread: 0.15, grounded: true }
  clock:  { h_m: 0.30, spread: 0.30, grounded: false }   # on a wall, not the floor
```

`grounded: false` matters. The ground-plane estimator turns a box bottom into
metres, and for a wall-mounted object that bottom is nowhere near the floor.

- **Watch out:** the dominant term is **camera pitch**, not the priors.
  `height / tan(pitch + pixel angle)` — the same pixel gives 4.4 m at 0° and
  1.2 m at 30°. Android reads it from the gravity sensor (`perception/Tilt.kt`);
  the prototype takes `--camera-pitch`, so pass the real value or the numbers
  are meaningless.
- **Afterwards:** `pytest tests/test_distance.py`.

## "I want to change the screen"

**File:** `android/.../MainActivity.kt` (built in code, no XML layouts)

- Controls are ≥64 dp and every one has a `contentDescription`; the app must
  stay fully operable by ear with the display dark.
- Colour never carries meaning alone.
- **Read state from the service**, never from a local boolean. A cached flag
  drifts the moment the service is stopped from its notification, and then
  every control silently refuses to fire. That is what "the app is broken" felt
  like the first time.
- A control that cannot work should be *absent or explained*, never a button
  that accepts a press and does nothing.

## "I want to change the training data"

**Files:** `training/datasets/*.yaml`, `training/taxonomy/sarathi77.yaml`

Every source declares its licence and its role. `role: eval_only` keeps a
dataset out of training entirely — IDD is licensed for student use, and Sarathi
ships weights to people who are not students, so it is held out and used only
to measure the domain gap.

```bash
.venv/bin/python -m sarathi.cli dataset --out data/build
.venv/bin/python -m sarathi.cli taxonomy      # coverage and blind spots
```

- **Watch out:** the builder clears its output directory, but only one it
  created. Writing a new label set into a directory holding an old one makes
  the trainer log `ignoring corrupt image/label` and train on a silently
  reduced subset, with exit code 0.
- **Watch out:** splitting is stratified per source and per contiguous block.
  Grouping by directory once put an entire 13,928-image dataset into
  validation, and the run looked excellent.

## "I want to run the web demo"

```bash
cd web && python3 -m http.server 8000
# then open http://localhost:8000
```

`file://` will not work — the browser blocks fetching the model. It needs an
HTTP server, and a camera needs `localhost` or HTTPS.

The page shares `sarathi.js` between the browser and the Node verification
script, so the decode is tested against the real ONNX rather than assumed.

## "I want to release a new version"

```bash
cd android
./gradlew assembleRelease     # app/build/outputs/apk/release/app-release.apk
```

Bump `versionCode` and `versionName` in `app/build.gradle.kts`.

The release is signed with `android/sarathi-release.jks`, committed on purpose:
this is handed out as an APK, not shipped through a store, and an unsigned
build will not install. **Anyone shipping this seriously should replace that
keystore with a private one** — but keep using the same key thereafter, or
updates fail with a signature mismatch instead of upgrading.

R8 shrinking is deliberately **off**. It strips classes reached only
reflectively, and this app resolves adapters and YAML types that way.

---

## What to run before you push

```bash
cd prototype && .venv/bin/python -m pytest tests -q     # 406 tests
cd android   && ./gradlew assembleDebug
```

Then, if the change touches perception at all, put it in front of a camera:

```bash
cd prototype && .venv/bin/python -m sarathi.desktop
```

and on the device, confirm the self-tests:

```
adb logcat -s SarathiService SarathiDelegate | grep -E "self-test|backend"
```

A healthy launch looks like:

```
backend xnnpack-2t (34.5 ms)
self-test: maxScore=0.754 detections=2 [clock 0.75, car 0.68] in 28ms
ocr self-test (LATIN): PLATFORM 3. Exit via stairs. Lift out of order. Room 214
```

---

## Things that will bite you

| Symptom | Cause |
|---|---|
| App detects nothing, `maxScore` under 0.1 | frame rotation or YUV stride — check `Yuv.kt` |
| `maxScore=1.000`, 50 boxes, one class | broken quantization; re-export without `-qt` |
| Distances wildly wrong | camera pitch, not the size priors |
| Rate stuck at 1 Hz | thermal governor. Unplug the phone — charging alone puts a Pixel 8a near throttling |
| "Model not installed" with weights present | a runtime name missing from the engine-format table in **both** parsers |
| A backend that is fastest and finds nothing | the GPU delegate. The agreement check rejects it; do not "fix" that by loosening the check |
| Desktop app shows no spoken text | it was gated on audio being enabled once; the speech policy must run even when muted |

## Where things live

```
prototype/sarathi/     the pipeline: sources, models, perception, guidance, runtime
  desktop.py           the windowed app
  cli.py               probe / run / bench / gate / read / dataset / models / licenses
android/app/src/main/  the phone app
  .../perception/      detector, geometry, tilt, delegate selection
  .../guidance/        tracking, saliency, voice
  .../runtime/         scheduler, foreground service, state bus
web/                   the browser demo
models/manifests/      one YAML per model - the contract both sides read
phrases/               everything spoken, both languages
training/              taxonomy, size priors, dataset definitions
docs/                  design, measurements, results, this guide
```

## Documentation map

| | |
|---|---|
| [00-overview](00-overview.md) | Product, users, how success is judged |
| [01-architecture](01-architecture.md) | Design, frame lifecycle, known limitations |
| [02-model-selection](02-model-selection.md) | Candidates and licence analysis |
| [03-optimization](03-optimization.md) | Every measure with before/after |
| [04-datasets](04-datasets.md) | Sources, licences, the domain-gap experiment |
| [05-vlm](05-vlm.md) | Scene description |
| [06-ocr](06-ocr.md) | Text reading |
| [07-results](07-results.md) | Held-out accuracy per class |
| [08-baseline](08-baseline.md) | Where the time goes |
