# Optimization

> Every measure on this page is either **measured**, with the device and
> command that produced the number, or **planned**, and labelled as such. A
> claimed optimization with no before/after is a claim, not a result.

## The one number that matters

Battery life on this product reduces to a single quantity: **what fraction of
captured frames never reach the model.** Everything else — quantization,
delegate choice, input resolution — changes what one inference costs. Gating
changes how many happen at all, and it is worth more than the rest combined.

Measure it yourself:

```bash
.venv/bin/python -m sarathi.cli gate --source 0 --seconds 20
```

## Measured

### 1. Motion gating and adaptive rate — **implemented**

M3 MacBook, built-in camera, 30 fps capture, defaults (`max_inference_hz=8`,
`idle_inference_hz=1`, `keepalive_hz=0.2`, `motion_threshold=0.012`,
`settle_s=2.0`). Inference cost is charged at a placeholder 25 ms per pass
until a real detector is benchmarked, so the *ratios* below are real and the
absolute seconds are provisional.

| Scene | Frames | Reached the model | Skipped: static | Skipped: rate | Inference work vs no gating |
|---|---:|---:|---:|---:|---:|
| **Stationary**, real webcam | 600 | **18 (3.0%)** | 71.3% | 25.7% | **33× less** |
| **Stationary**, synthetic still clip | 600 | 18 (3.0%) | 71.2% | 25.8% | 33× less |
| **Walking**, synthetic moving clip | 600 | **106 (17.7%)** | 4.3% | 78.0% | **5.7× less** |

Three things worth reading off this table:

**Real sensor noise does not trip the gate.** The live webcam and the perfectly
static synthetic clip produce the same 3.0%. That was not a given — a threshold
tuned too tight would have had noise waking the detector continuously, and the
whole idle saving would have evaporated. At 0.012 it does not.

**When stationary, the gate does the work; when walking, the rate limiter
does.** Standing still, 71% of frames are dropped as unchanged. Walking, only
4% are — but 78% are dropped by the rate limit, because there is no value in
running faster than a person can react. Two different mechanisms carrying the
load in two different situations is the design working as intended.

**Walking is the honest number, not standing.** 33× flatters the system: it
describes a phone on a desk. A real user's saving sits between 5.7× and 33×
weighted by how much they actually walk. Any battery claim should quote the
walking figure.

### Caveats

- Synthetic clips only for the walking case. A real walk has camera shake,
  exposure changes and rolling shutter, all of which will push the static-skip
  rate down. The number will get worse with real footage, and that is the
  number that should be published.
- 25 ms per inference is a placeholder. It gets replaced by a measured
  Pixel 8a figure once a detector is benchmarked.
- The stationary test is a desk, not a person standing on a street. Standing at
  a crossing with traffic moving is much closer to the walking case.

## Implemented, not yet quantified

| Measure | What it does |
|---|---|
| **Keepalive floor** | Inference runs at 0.2 Hz regardless of any gate. A blind user cannot tell "nothing to report" from "stopped working", so no gate is allowed to silence the system completely. |
| **Thermal shedding** | Frame rate ramps down between `thermal_soft` and `thermal_hard` rather than stepping, because a step change in announcement rate is audible. Reported separately from plain rate limiting so measurements can tell "we chose to" from "the device made us". |
| **Staleness drop** | Frames older than 250 ms at decision time are discarded rather than processed. Acting on them describes somewhere the user no longer is. |
| **Tier separation** | Depth runs at 2 Hz and only while moving; OCR and the VLM only on request. |
| **Speech drop-not-queue** | Utterances are discarded rather than queued when the channel is busy, so a backlog cannot form. |

## Planned

| Measure | Expected mechanism | Status |
|---|---|---|
| INT8 quantization | Per-channel post-training quantization; QAT only if PTQ accuracy loss is unacceptable | Blocked on detector selection |
| Delegate cascade | NPU → GPU → XNNPACK, auto-benchmarked on first run. **NNAPI is deprecated as of Android 15**, so the working TPU path on a Tensor G3 needs verifying on-device rather than assumed | Android phase |
| Input resolution ladder | 320 px for detection; higher only on demand. Worth testing a portrait-aware crop — WOTR's second most common resolution is 1020×1360, and square letterboxing discards most of a portrait frame | Android phase |
| Screen-off operation | Foreground service, hardware-button triggers. On a mid-range phone an always-on display can consume more than the inference does | Android phase |
| Zero-copy input | YUV → tensor without a per-frame bitmap allocation | Android phase |
| IMU-driven activity | Step cadence from the phone's IMU instead of inferring activity from the image. More reliable, and it still works when the camera is external — though then the camera's motion is no longer the phone's, which is an open question | Android phase |

## How these numbers are produced

`sarathi gate` runs the real scheduler against a real source and reports the
skip breakdown by reason. It charges a configurable per-inference cost rather
than running a model, which keeps the measurement about *scheduling* and makes
it reproducible on any machine.

Reproduce the table above:

```bash
# stationary
.venv/bin/python -m sarathi.cli gate --source 0 --seconds 20

# walking, from a recorded clip
.venv/bin/python -m sarathi.cli gate --source walk.mp4 --seconds 20

# with real thermal pressure read from the OS
.venv/bin/python -m sarathi.cli gate --source 0 --seconds 20 --thermal
```

Tuning knobs are exposed so the trade can be explored rather than argued
about: `--threshold`, `--settle`, `--max-hz`, `--idle-hz`, `--keepalive-hz`.

---

## Measured on the target device

Pixel 8a, Tensor G3, Android 17. YOLO11n at 320 px, **float32 on the CPU via
XNNPACK** — no quantization, no GPU delegate. This is the unoptimized baseline
every later measure gets compared against.

| | |
|---|---|
| Inference | **69–107 ms** per pass |
| Frames reaching the model | ~5% (rate limit at 8 Hz against ~30 fps capture) |
| Detection quality | max class score 0.95; `person` and `table` detected and announced correctly at ~1.9 m |
| Thermal after sustained use | skin **43.6 °C**, battery **44.9 °C**, `THERMAL_STATUS_MODERATE`, headroom **0.95** |

### The finding that matters

**Sustained camera plus fp32 CPU inference drives a Pixel 8a to moderate
thermal throttling.** That is not a marginal effect — Android reported 0.95
headroom, where 1.0 is the severe threshold, and the governor correctly cut the
inference rate to the 1 Hz idle floor.

Two consequences.

The thermal governor earns its place. Without it the app would keep pushing 8 Hz
into a phone the platform was about to throttle anyway, and the OS would take
the decision instead — abruptly, and without the app being able to shed the
*right* work.

And 69–107 ms per inference is the actual problem. It is far too expensive to
sustain, and the fix is the optimization work that has not happened yet:

| Measure | Expected effect | Status |
|---|---|---|
| INT8 quantization | ~3–4× faster, ~4× smaller | Not done — needs a calibration set |
| GPU delegate | Moves work off the CPU, reducing skin temperature directly | Not done — `litert-gpu` is a dependency already |
| NPU / Tensor TPU | Best case, but NNAPI is deprecated as of Android 15 and the working path on a G3 needs verifying on-device | Not done |

Until those land, no battery-life claim should be made from this baseline. The
number to quote today is the one above: *unoptimized fp32 CPU inference heats
the device to moderate throttling*.

### A threshold calibrated by measurement rather than guess

The governor originally shed rate above 0.30 headroom. Against real readings
that was badly wrong: a phone merely warm from ordinary use sat above it, so
the rate collapsed from 8 Hz to 1 Hz on a device that was coping — an eightfold
loss of responsiveness for nothing.

Now `soft = 0.60`, `hard = 0.95`. Since 1.0 is Android's severe threshold, that
sheds across a band that still leaves genuine headroom, without reacting to
warmth the device is handling.


## Quantization, measured

Every variant `onnx2tf` can produce, scored against 250 real calibration frames
stratified across all five source datasets. Latency here is x86 server CPU; the
on-device figures follow.

| model | size | ms | max class score | verdict |
|---|---:|---:|---:|---|
| fp32 baseline | 10.60 MB | 7.0 | 0.876 | reference |
| **dynamic-range** | **2.87 MB** | **5.7** | **0.878** | **shipped** |
| float16 | 5.36 MB | 6.3 | 0.876 | unusable on CPU |
| full INT8 | 2.95 MB | 3.7 | **0.000** | broken |
| INT8 + int16 activations | 3.07 MB | **307** | 0.879 | 46x slower |

**Full INT8 is the fastest and detects nothing.** Box regression survives while
the classification head collapses entirely: box rows still peak at 333.8 while
class rows go to 0.0000. Post-sigmoid class scores occupy a very small numeric
range, and a per-tensor int8 scale rounds every one of them to zero.

The textbook fix for that is int16 activations. It restores accuracy exactly
and costs **46x** the latency, because TFLite has no optimised int16 kernel and
falls back to reference implementations.

Dynamic-range quantization - int8 weights, float32 activations computed at
runtime - is the only variant that is smaller, faster and numerically
equivalent at once. It also requires no calibration set at all, so the
stratified 250-image set built for full INT8 turned out to be unnecessary for
what shipped.

### On the Pixel 8a

Same app, same scene, only the model changed:

| | fp32 | dynamic-range |
|---|---:|---:|
| Inference | 69-107 ms | **23-29 ms** |
| Model size | 10.60 MB | **2.87 MB** |
| Thermal headroom | 0.95 (at the severe threshold) | **0.72-0.80** |
| Governor rate | clamped to the 1 Hz idle floor | **5.7 Hz** |

A **3-4x latency reduction**, and the second-order effect matters as much as
the first: the device runs cool enough that the thermal governor stops having
to intervene, so the inference rate is set by the product's own policy rather
than by heat.

## The GPU delegate: measured, and rejected

The GPU was the obvious next lever, and it does not work on this device. The
result is worth reading carefully, because the failure mode is the dangerous
kind.

The app benchmarks candidate backends on first launch and keeps the winner
(`android/.../perception/Delegates.kt`). Every candidate is also scored against
a single-threaded CPU run on the same real image, and rejected if it disagrees.
On a Pixel 8a, LiteRT 1.2.0:

| model | xnnpack-4t | xnnpack-2t | gpu | gpu, fp32 | gpu, no-quant | gpu, OpenGL |
|---|---:|---:|---:|---:|---:|---:|
| dynamic-range (shipped) | 60 ms OK | 33 ms OK | **15-32 ms WRONG** | 26 ms WRONG | 22 ms WRONG | no shader impl. |
| fp32 | 73 ms OK | 71 ms OK | delegate fails | delegate fails | delegate fails | delegate fails |
| fp16 | interpreter fails | interpreter fails | interpreter fails | - | - | - |

Reproduce it on any phone:

```bash
adb shell run-as in.sarathi.app touch files/survey   # then start guidance
adb logcat -s SarathiDelegate
```

**The GPU is the fastest thing on the device and it computes the wrong
answer.** It accepts the entire graph - `Replacing 377 out of 377 node(s) with
delegate (TfLiteGpuDelegateV2)` - runs roughly twice as fast as the CPU, and
returns a tensor that deviates from the CPU by up to 264.5 absolute on a
reference value of 43.5. That is not fp16 rounding. It is 5717 of 176400
elements outside a tolerance of `0.03 + 2%`, scattered across indices 0-165825,
which spans both the four box-regression rows and the eighty class rows. The
whole delegation is unsound rather than one op.

Three configuration probes rule out the usual explanations:

- `setPrecisionLossAllowed(false)` changes the error from 264.539 to 264.695.
  Not precision.
- `setQuantizedModelsAllowed(false)` produces **byte-identical** wrong output,
  so the delegate is not treating this as a quantized model at all and the int8
  weight path is not implicated.
- Forcing the OpenGL backend fails at init with "No shader implementation",
  so there is no second GPU path to fall back to.

And the two float variants remove the last hope of a workaround: the GPU
delegate refuses the fp32 graph outright ("Error applying delegate", with the
interpreter warning that `tensor#69` is dynamically sized and the delegate
requires static shapes), while the fp16 graph cannot even be loaded on this
build - `conv.cc:360 input_type == kTfLiteFloat32 || ... was not true`, the same
whole-graph-fp16 problem documented above.

So there is no correct GPU path for this detector on this device, and the app
runs on `xnnpack-2t` at ~33 ms.

### Why this is the finding, not a footnote

Had the agreement check not been there, this would have shipped. The app would
have selected the GPU for being fastest, run at double the rate, drawn less
power, and **announced nothing at all** - because garbage in the class rows
produces zero detections, and zero detections is exactly what an empty room
also produces. A blind user would have had a device that felt like it was
working and silently reported no hazards.

The first version of the check nearly failed in the opposite direction. It used
a flat absolute tolerance across the output, which asks 0.017% relative accuracy
of a box coordinate near 300 - something fp16 cannot supply by construction. It
rejected the GPU for a reason that would have been wrong. The fix was the mixed
rule `|a-b| <= atol + rtol*|b|`, and re-running it produced a rejection with
evidence instead of a rejection by accident.

Two lessons, both cheap to reuse: **benchmark on a real image**, because a blank
tensor drives every class score near zero and any two backends agree on noise;
and **log the deviation whether or not it fails**, because "rejected" is not a
result, while "264.5 absolute at reference 43.5, 5717 of 176400 elements, spread
across the whole head" is.

### A second-order cost of losing the GPU

CPU latency is thermally unstable in a way the GPU's was not. Across repeated
survey runs on a warm device, `xnnpack-4t` measured 34.6, 35.7, 58.8, 59.9,
60.6 and 72.9 ms for the same work, while the GPU stayed between 15 and 32 ms
throughout. Keeping detection on the CPU therefore means the thermal governor
is not a nicety - it is the only thing standing between a warm phone and an
inference rate that quietly halves.

### What is left to try

- Re-export with fully static shapes, so the GPU delegate will at least accept
  the fp32 graph. `tensor#69` being dynamic is a property of the export, not of
  YOLO11.
- Retest on a future LiteRT. This is a delegate correctness bug on a widely
  used detector, and it is the kind of thing that gets fixed upstream.
- The survey is committed, so re-checking either of those is one command.

### The export command, recorded because not having it cost a day

The exact flags were never written down, and re-deriving them from memory
produced three broken models that loaded fine and returned `maxScore=1.000`
with fifty boxes of one class - saturated garbage, caught only by the on-device
self-test. The culprit was adding `-qt per-tensor`, which is the same
per-tensor scaling documented below as collapsing the classification head; it
damages the dynamic-range output too, not just full INT8.

```bash
# ONNX from the training venv (ultralytics + torch)
.venv/bin/python -c "
from ultralytics import YOLO
YOLO('best.pt').export(format='onnx', imgsz=320, opset=13, dynamic=False)"

# TFLite from the export venv (onnx2tf + tensorflow). NO -qt flag.
.venv-export/bin/python -m onnx2tf -i best.onnx -o out \
    -osd -coion -b 1 -tb tf_converter -oiqt

# take *_dynamic_range_quant.tflite
```

Verify before it reaches a phone - a broken export is not visibly broken:

```python
y = interpreter.get_tensor(out["index"])[0]
print("box rows max", y[:4].max(), "class max", y[4:].max())
# healthy: box rows ~ the input size (320), class max 0.6-0.9
# broken:  class max 1.0000, or 0.0000
```

### Three dead ends, recorded so nobody repeats them

- `tensorflow-cpu` publishes no Apple Silicon wheels. The x86 wheel installs
  and then aborts under Rosetta on missing AVX.
- onnx2tf's default `flatbuffer_direct` backend **cannot quantize** at all -
  it fails with "flatbuffer_direct fast path failed". Quantization needs
  `-tb tf_converter`, which in turn needs the optional `tf_keras` dependency.
- onnx2tf silently applies **ImageNet normalisation** to calibration data.
  Feeding it images already scaled to [0,1] double-normalises them and
  calibrates every activation range against a distribution the model never
  sees, unless `-qnm` and `-qns` are passed explicitly.
