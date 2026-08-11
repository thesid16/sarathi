# Baseline: where the time actually goes

> Pixel 8a (Tensor G3, Android 17), thermal headroom 0.86–0.94 throughout —
> the phone was charging over USB, which on this device puts the CPU cluster
> near throttling. Every figure is therefore **pessimistic**; on battery the
> same work runs faster. Averaged over hundreds of frames, not timed once.
>
> Reproduce: `adb logcat -s SarathiService` and read the `stages` line.

## Per-stage, one processed frame

| Stage | Time | Runs on | Share |
|---|---:|---|---:|
| Motion gate | **1.4 ms** | every captured frame | — |
| YUV → upright bitmap | **61 ms** | frames that pass the gate | **51%** |
| Detect (preprocess + inference + NMS) | **57 ms** | frames that pass the gate | **48%** |
| Geometric distance | 0.02 ms | per frame | 0.02% |
| Tracking | 0.12 ms | per frame | 0.1% |
| Saliency | 0.10 ms | per frame | 0.1% |
| Phrasing + speech dispatch | <0.1 ms | when something is said | — |

**Everything after detection is free.** Distance, tracking, saliency and
phrasing together cost under a quarter of a millisecond — about 0.2% of a
frame. All the design attention in this project went into those stages, and all
the time is spent before them.

**The single largest cost was not the model.** Converting the camera frame cost
as much as running the detector on it, and nothing had ever measured it: the
logs reported inference latency, and inference latency looked fine. It only
became visible when every stage was instrumented at once, which is the entire
argument for doing this.

### What the conversion was doing

`ImageProxy` → NV21 → **JPEG encode** → **JPEG decode** → Bitmap → rotated
Bitmap. Four full-image passes, two of them a codec, to change a pixel format.

| Implementation | Time |
|---|---:|
| NV21 → JPEG → Bitmap → rotate (original) | 78–105 ms |
| One loop reading planes directly | **133 ms** — *worse* |
| Bulk-copy planes, then convert from arrays | **61 ms** |

The middle row is worth keeping. The obvious rewrite — a single loop doing the
colour conversion with rotation folded into the index — was measurably *slower*
than the JPEG round-trip it replaced. A per-pixel `ByteBuffer.get()` is a
virtual call on a direct buffer, and 307,200 pixels × 3 planes is a million of
them, against libjpeg's hand-tuned native code. Beating native code from the
JVM needs the data in a primitive array first, where the JIT emits plain loads.

Same arithmetic, same output, 2.2× apart depending only on where the bytes live.

### Still the bottleneck

61 ms is an improvement and it is not a fix. The remaining work is unnecessary:
the frame is converted at 640×480 and then scaled to 320×320 for the detector,
so three quarters of the pixels are converted and thrown away. Converting
directly into the letterboxed input — or skipping the `Bitmap` entirely and
writing YUV straight into the float tensor — should take this under 15 ms.

That is the next optimization and it is worth more than any remaining model
choice.

## Model ladder

Measured back-to-back on the same input by the on-device survey, so these are
comparable in a way that live runs are not:

| Model | Classes | Size | xnnpack-4t | xnnpack-2t | What it is for |
|---|---:|---:|---:|---:|---|
| **sarathi26-320** | 26 | 2.8 MB | **26.3 ms** | **25.3 ms** | Stairs, manholes, tactile paving |
| yolo11n-coco-256 | 80 | 2.9 MB | 28.3 ms | 26.6 ms | Battery and heat |
| yolo11n-coco-320 | 80 | 2.9 MB | 46.8 ms | 44.1 ms | General indoor default |
| yolo11s-coco-320 | 80 | 9.8 MB | 119.5 ms | 103.4 ms | Accuracy, on a cool phone |

Each on the bundled self-test image:

```
sarathi26-320     pole 0.66, car 0.66, tree 0.60
yolo11n-coco-256  clock 0.74, car 0.63
yolo11n-coco-320  clock 0.75, car 0.68
yolo11s-coco-320  car 0.84, clock 0.72
```

### The fine-tuned model is both more useful and faster

`sarathi26-320` is the detector this project trained, and it is the **fastest**
of the four despite sharing an architecture and input size with
`yolo11n-coco-320`. 26 classes instead of 80 makes the head output
`[1, 30, 2100]` rather than `[1, 84, 2100]`, which is less compute in the last
layers and 64% less work in decode and NMS on every frame.

It is also the only one that knows what a staircase or an open manhole is —
COCO contains neither, so the COCO builds are structurally blind to the hazards
this product exists to warn about. They are kept because COCO covers indoor
clutter (chairs, tables, doorways, people) that the 26-class set deliberately
does not.

Held-out accuracy is in [docs/07-results.md](07-results.md): mAP50 **0.597**
overall, `stairs_up` recall 0.983, `open_manhole` 0.930 — and 0.219 on Indian
road footage held out entirely, which is the honest transfer number.

### Choosing

- **Walking outdoors** → `sarathi26-320`. Fastest, and the only one that sees
  the hazards.
- **Indoors, or a demo** → `yolo11n-coco-320`. People, chairs, doorways.
- **Hot phone or long walk** → `yolo11n-coco-256`.
- **Cool phone, accuracy first** → `yolo11s-coco-320`, at 2.5× the latency.

Switch from the **Model** button; guidance must be stopped first, because
swapping mid-run would leave tracks and cooldowns attached to objects the new
model has never seen.

## The GPU delegate, across four models

Every model, every GPU configuration, still wrong:

```
sarathi26-320     gpu  30.8 ms  worst abs 152.1   WRONG
yolo11n-coco-256  gpu  25.9 ms  worst abs 160.1   WRONG
yolo11n-coco-320  gpu  22.8 ms  worst abs 264.5   WRONG
yolo11s-coco-320  gpu  30.8 ms  worst abs 227.1   WRONG
```

The earlier finding was one model; this is four, spanning two architectures,
two class counts and two input sizes. The LiteRT GPU delegate miscomputes this
whole model family on a Tensor G3, and it is consistently the *fastest* option
on offer — which is exactly why the agreement check exists. See
[docs/03-optimization.md](03-optimization.md).

## On-demand tiers

| | Cold | Warm |
|---|---:|---:|
| Text reading (ML Kit) | 798 ms | 550 ms |
| Scene description (Gemma 4 E2B) | 5.8 s load + 9.1 s | 3.6 s |

Both are off the safety path and behind a button, so their latency is bounded
by how often it is pressed. Gemma also costs 3.0 GB resident while loaded and
is released after 90 s idle — see [docs/05-vlm.md](05-vlm.md).

## What this changes

1. **Optimize the conversion, not the model.** A 46 ms saving is available in
   `Yuv.toBitmap` for no accuracy cost at all. Moving from
   `yolo11n-coco-320` to `sarathi26-320` saves 20 ms *and* improves relevance,
   which makes it the better default for outdoor use — but neither is the big
   number.
2. **Guidance logic is not worth optimizing.** Saliency, tracking, phrasing and
   distance sum to 0.24 ms. Any effort there is misdirected.
3. **The gate earns its place.** 1.4 ms on every frame to avoid ~118 ms on most
   of them; at 97–99% skip rates it is paying for itself many times over.
4. **Measure on battery.** Every number here was taken on a charging phone
   sitting near the throttling threshold. That is the honest worst case, and it
   is not the number to quote for a walk.
