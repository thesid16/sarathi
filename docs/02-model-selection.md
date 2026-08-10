# Model selection and licensing

> **Status: licence analysis complete, benchmarks not yet run.** The
> performance tables are empty on purpose. They get filled in from
> `sarathi.bench` output on a Pixel 8a — not from papers.
>
> This is engineering analysis, not legal advice.

## 1. What the AGPL-3.0 decision buys and costs

Sarathi is released under **AGPL-3.0**. That resolves the licence question that
usually dominates a project like this — Ultralytics YOLO, the default choice in
nearly every tutorial and itself AGPL-3.0, is now **available to us**.

It does not make licensing a non-issue. It changes the question from *"can we
use this?"* to *"can we redistribute this, and does it restrict our users?"*

Three rules follow, and the registry enforces them in code:

| Rule | Why |
|---|---|
| **Apache-2.0 / MIT / BSD models: bundle freely** | One-way compatible with AGPL-3.0. Safest and most reusable. Preferred where competitive. |
| **AGPL models: usable, bundleable** | Same licence as the project. No conflict. |
| **Non-commercial models (CC-BY-NC): excluded entirely** | These would make Sarathi non-free and block the students, NGOs and researchers this exists for. A downstream accessibility startup could not ship it. Not worth the accuracy. |
| **Use-restricted models: user-downloaded, never bundled** | Terms that add restrictions AGPL-3.0 cannot absorb would muddy our own licence. As an optional pack the user fetches, the terms stay between the user and the publisher. |
| **Very large models: user-downloaded regardless of licence** | A multi-gigabyte model in the APK makes the app uninstallable for users on 4 GB phones and metered connections — which is much of the audience. Permissive licensing does not make a 2 GB download acceptable. |

Those last two rows used to be one row. Gemma was the motivating example for
the first, because Gemma 3n's terms carried use restrictions; **Gemma 4 is
Apache-2.0** and moved to the second. The model-pack architecture serves both,
which is why the conclusion did not change when the reason did — see §5.

### What the registry enforces

Every manifest declares:

```yaml
license: AGPL-3.0
distribution: bundled        # bundled | user_download | excluded
```

- `excluded` models refuse to load at all.
- `user_download` models are never committed and never packed into the APK;
  they are fetched by the user, who accepts the model's terms at that point.
- `sarathi licenses` prints the licence and distribution mode of every loaded
  model, so any build can be audited in one command.

## 2. Detector candidates

With AGPL available, the shortlist widens and the question becomes purely
technical: **best accuracy per millisecond at 320 px INT8 on a Tensor G3.**

| Model | Licence | Notes |
|---|---|---|
| **YOLO11n** (Ultralytics) | AGPL-3.0 | Now a real candidate, not just a baseline. Best-in-class tooling, export and quantization path. Likely lead. |
| **YOLOv8n** (Ultralytics) | AGPL-3.0 | Older, extremely well-understood, huge community. Fallback if v11 export misbehaves. |
| **YOLOX-Nano / Tiny** (Megvii) | Apache-2.0 | Anchor-free, mobile-first, mature ONNX path. Preferred if it lands close to YOLO11n — Apache is strictly better for reuse. |
| **NanoDet-Plus** (RangiLyu) | Apache-2.0 | Extremely light, built for ARM. Strong at the low end of the budget. |
| **PP-PicoDet** (PaddleDetection) | Apache-2.0 | Excellent accuracy-per-FLOP; Paddle-first export adds friction. |
| **EfficientDet-Lite0** (Google) | Apache-2.0 | Dependable, first-class LiteRT support and quantization behaviour. |
| **RF-DETR** (Roboflow) | Apache-2.0 | Recent and strong; transformer decode cost on a mobile NPU needs verifying. |
| **RT-DETR** — Baidu original | Apache-2.0 | Apache **only** in the PaddleDetection original; the Ultralytics port is AGPL. Implementation matters as much as architecture. |
| **YOLO-NAS** (Deci) | Non-commercial weights | ❌ Excluded. Weights restrict downstream users. |

**Plan:** benchmark YOLO11n, YOLOX-Nano and NanoDet-Plus at 320 px INT8 on the
Pixel 8a. Prefer the Apache option when within ~2 mAP — reusability is worth
more to this project than the last two points. If YOLO11n is decisively better,
take it; the project is AGPL anyway.

### Performance — *to be measured*

| Model | Input | Quant | mAP@50-95 | Latency (Pixel 8a) | Latency (M3) | Size |
|---|---|---|---|---|---|---|
| YOLO11n | 320 | INT8 | — | — | — | — |
| YOLOX-Nano | 320 | INT8 | — | — | — | — |
| NanoDet-Plus | 320 | INT8 | — | — | — | — |
| EfficientDet-Lite0 | 320 | INT8 | — | — | — | — |

## 3. Depth candidates

| Model | Licence | Ships? | Notes |
|---|---|---|---|
| **Depth Anything V2 — Small** | Apache-2.0 | ✅ | Best quality in the size class. **Only the Small variant** is Apache. |
| Depth Anything V2 — Base / Large | CC-BY-NC-4.0 | ❌ | Excluded. Non-commercial terms would restrict our users. |
| **MiDaS v2.1 small** (Intel ISL) | MIT | ✅ | Older, less sharp, but a proven TFLite INT8 path. Safe fallback. |
| **DPT-Hybrid** | MIT | ⚠️ | Too heavy for the mobile budget; reference only. |

Row 2 is a genuine trap worth naming: "Depth Anything V2 is Apache-2.0" is true
of the checkpoint we can use and false of the two better ones in the same
repository. Manifests therefore record licence **per checkpoint**, not per
model family.

## 4. OCR candidates

| Option | Licence | Ships? | Notes |
|---|---|---|---|
| **RapidOCR** (ONNX) | Apache-2.0 | ✅ | PaddleOCR models converted to ONNX. Prototype default. |
| **PaddleOCR mobile** | Apache-2.0 | ✅ | Same models upstream; heavier runtime dependency. |
| **ML Kit Text Recognition v2** | Google ToS, free | ⚠️ | On-device, excellent, **supports Devanagari**. Ties the app to Google Play Services — fine on a Pixel, excludes AOSP and de-Googled phones. |
| Tesseract | Apache-2.0 | ❌ | Built for scanned documents; poor on photographed scene text. |

**Recommendation:** RapidOCR in the prototype. On Android, ML Kit behind the
same plugin interface with a RapidOCR/LiteRT fallback, so the Play Services
dependency is swappable rather than structural.

## 5. VLM candidates — the scene-description slot

| Model | Size | Licence | Distribution |
|---|---|---|---|
| **SmolVLM2-500M** | ~0.5 B | Apache-2.0 | Bundled — default. Small enough to ship; genuinely usable for "what's in front of me". |
| **SmolVLM2-256M** | ~0.26 B | Apache-2.0 | Bundled — low-RAM fallback. |
| **Gemma 4 E2B** | 2.3 B eff. (5.1 B w/ embeddings) | **Apache-2.0** | **User download — for size, not licence.** Multimodal, 128k context. Reported to run on a Raspberry Pi 5 with 8 GB, so viable on a Pixel 8a. |
| **Gemma 4 E4B** | 4.5 B eff. (8 B w/ embeddings) | **Apache-2.0** | User download. Only for high-RAM devices. |
| **Moondream 2** | ~1.9 B | Apache-2.0 | User download. Strong at exactly this task. |
| **Qwen2.5-VL 3B** | ~3 B | Apache-2.0 *(verify)* | User download. Top of the RAM budget. |

### Correction: Gemma's licence changed, and it changes the design

This document previously recorded Gemma as commercially usable but **not**
OSI-open, carrying a use-restriction policy that AGPL-3.0 cannot absorb — which
made bundling it impossible and put an item on the legal review list.

**Gemma 4, released 2 April 2026, is Apache-2.0.** Verified against
[Google's model card](https://ai.google.dev/gemma/docs/core/model_card_4) and
the [Hugging Face release](https://huggingface.co/blog/gemma4). The constraint
is gone, and the legal item is closed.

The *conclusion* is unchanged — Gemma stays a user download — but the reason is
now completely different, and the difference matters:

| | Before (Gemma 3n) | Now (Gemma 4) |
|---|---|---|
| Why not bundled | **Licence.** Use restrictions AGPL cannot pass on. | **Size.** 2.3 B effective params is a multi-gigabyte download. |
| Blocking? | Yes — needed legal sign-off | No — an ordinary product decision |
| Redistributable | No | Yes |

The Pixel 8a's 8 GB makes Gemma 4 E2B genuinely viable as your development
target. It will not be viable on the 4 GB phones much of the audience actually
owns, and forcing a multi-gigabyte download on everyone would make the app
uninstallable for the people who need it most. Hence: a small VLM bundled as
the default that always works, larger models offered as opt-in packs gated on
available RAM.

**No number is claimed for Gemma 4 anywhere in this project.** It has not been
run on a phone. Latency, memory and description quality all stay blank until it
has.

## 6. Datasets

Moved to its own document: **[`04-datasets.md`](04-datasets.md)**, which
supersedes what follows.

Short version: no data is collected with a camera, everything comes from
public sources, and having actually searched, that turns out to be far more
workable than first assumed. A pedestrian-viewpoint obstacle dataset exists
under MIT (WOTR, ~190k objects), open-manhole data exists under CC BY 4.0, and
Mapillary supplies worldwide street-level imagery under CC BY-SA with free API
access — which, combined with open-vocabulary auto-labelling, covers the
Indian domain gap without anyone picking up a camera.

<details>
<summary>Superseded first-pass notes</summary>

### Usable base

| Dataset | Terms | Value here |
|---|---|---|
| **COCO** | Annotations CC-BY-4.0; images Flickr, per-image | ~30 of our classes. The reliable core. |
| **Open Images V7** | Annotations CC-BY-4.0; images CC-BY-2.0 | 600 classes and far more geographically diverse than COCO. The most valuable single source. |
| **LVIS** | CC-BY-4.0, COCO images | 1200 fine-grained everyday objects — good for the indoor half of the taxonomy. |
| **ADE20K / SUN RGB-D / NYU Depth v2** | Research-oriented | Indoor scenes, stairs and doorways. Licence review needed before weights derived from them ship. |
| **Roboflow Universe** | Varies per dataset — many CC-BY-4.0 / MIT | Best realistic route to India-specific classes. Must be filtered per-dataset by licence. |

### Excluded or flagged

| Dataset | Problem |
|---|---|
| Objects365 | Research-oriented terms; unclear for AGPL weight release |
| Mapillary Vistas | Research-only licence |
| Cityscapes | Research-only licence; European scenes anyway |
| IDD (India Driving Dataset) | ⚠️ Exactly the Indian road content we want, but research-oriented terms. Needs a licence read before use — potentially the highest-value item on this list. |

### The gap, stated plainly

Without self-collection, these classes will be **under-represented or absent**:

- open/missing manhole covers
- Indian kerb and footpath step geometry
- parked-scooter and cable clutter on footpaths
- auto-rickshaws at pedestrian range (driving datasets see them from a car, not from a pavement at 2 m)
- loose street dogs and cattle

These are not incidental — several are the *most dangerous* things the product
should catch. Three mitigations, none of which fully closes the gap:

1. **Lean on Open Images V7**, which has meaningfully better non-Western
   coverage than COCO.
2. **Mine Roboflow Universe** for permissively-licensed datasets covering
   specific missing classes, and record the licence of each in the training
   config.
3. **Resolve the IDD licence question.** If IDD is usable, it closes most of the
   road-scene gap on its own.

Whatever remains uncovered gets written into the evaluation as a **known
limitation with a named class list**, not quietly omitted. An assistive tool
that silently fails on open manholes is worse than one that documents that it
does.

</details>

### One point on redistribution

We ship *model weights*, not images. Training on a dataset and redistributing
that dataset are different acts, and the former is standard practice. Recorded
here because it is the question a careful reader will ask.

## 7. Open items

1. **IDD licence** — read the terms; it is the highest-value dataset for this
   taxonomy.
2. **Qwen2.5-VL 3B licence** — confirm Apache-2.0 for the 3 B checkpoint
   specifically.
3. **ML Kit** — acceptable to depend on Play Services for OCR, given a
   RapidOCR fallback exists?
4. **NNAPI on Tensor G3** — NNAPI is deprecated as of Android 15. The working
   TPU path on a Pixel 8a needs verifying on-device rather than assuming; the
   delegate cascade is written to auto-benchmark and pick a winner for exactly
   this reason.
