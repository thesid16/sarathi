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
