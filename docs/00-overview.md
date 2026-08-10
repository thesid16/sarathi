# Overview

## The product

A phone app that watches a camera feed, understands what is in front of the
user, and says it out loud — offline, in real time, without flattening the
battery.

## Who it is for

Blind and low-vision users, primarily in India, on mid-range Android phones.

That last clause is a design constraint, not a demographic footnote. It rules
out approaches that assume a flagship NPU, a large RAM budget, an unmetered
data connection, or a user who will tolerate a 3 GB download before the app
does anything. Several otherwise-reasonable architectures fail on this alone.

### Development target

**Pixel 8a** — Google Tensor G3, 8 GB RAM, Android 14+.

Chosen because it is the device on hand, and it is a reasonable proxy for the
mid-range: a competent but not flagship SoC. Two caveats to keep in mind so the
development device does not quietly become the assumed device:

- **8 GB of RAM is generous** for this audience. It makes a 2 B-parameter VLM
  viable here that will not fit the 4 GB phones much of the target audience
  owns. Hence the small-by-default, larger-by-opt-in model policy.
- **A Pixel has a TPU that most mid-range phones do not**, and NNAPI — the
  historical route to it — is deprecated as of Android 15. The delegate cascade
  auto-benchmarks on first run rather than assuming any particular accelerator
  exists, so results do not silently depend on Pixel-specific hardware.

Every benchmark in this documentation names the device it was run on for this
reason.

## What the user actually experiences

The phone is in a pocket or on a lanyard. The screen is off. The user hears:

- **While walking** — brief, sparse callouts about what is in the way:
  *"Chair, two o'clock, one and a half metres."* *"Step down ahead."*
  Nothing at all when there is nothing worth saying.
- **On a button press** — text read aloud: *"Door reads: Lab 204."*
- **On a long press** — a described scene: *"A corridor with a doorway on the
  left and two people walking towards you."*

No screen interaction is required for any of this. The screen is the single
largest power consumer on the device and is useless to the primary user.

## Requirements

### Functional

| ID | Requirement |
|---|---|
| F1 | Detect and announce obstacles and everyday objects with direction and approximate distance |
| F2 | Detect walking hazards that have no clean bounding box — steps, kerbs, drop-offs |
| F3 | Read text on demand: signs, door numbers, labels, packaging |
| F4 | Describe the scene on demand, in a full sentence |
| F5 | Speak in English or Hindi, selectable |
| F6 | Accept video from the phone camera or an external camera, interchangeably |
| F7 | Operate fully with the screen off, driven by hardware buttons |

### Non-functional

| ID | Requirement | Rationale |
|---|---|---|
| N1 | Fully offline; no inference leaves the device | Works in a basement or with no data balance; no per-user cost; video never leaves the phone |
| N2 | Glass-to-speech latency low enough to act on while walking | An obstacle warning that arrives after the obstacle is worthless |
| N3 | Battery cost low enough for a day of intermittent use | An aid that dies at 2 pm is not an aid |
| N4 | Speaks sparingly | Over-narration is the top reason users abandon assistive vision tools |
| N5 | Models swappable without an app update | New models — including a future Gemma — ship as model packs |
| N6 | Nothing bundled that restricts downstream users | Sarathi is AGPL-3.0 and public. A non-commercial model would block the students, NGOs and researchers it exists for — see [`02-model-selection.md`](02-model-selection.md) |
| N7 | Degrades rather than fails: camera dropouts reconnect silently | The user cannot see that the app went quiet |

N2 and N3 have no numeric targets yet **on purpose**. Targets set before
measuring are guesses that later get quietly relaxed. They will be fixed after
the first end-to-end measurement on the actual target phone, and recorded here
with the device named.

## Explicitly out of scope

- **Face recognition.** Materially changes the privacy and legal posture, and
  is not needed for navigation.
- **Turn-by-turn wayfinding.** This tells you what is *around* you; it is not a
  maps replacement, and pretending otherwise would be dangerous.
- **Smart-glasses support.** Not technically possible today — see the README.
  The architecture leaves the door open; the roadmap does not walk through it.
- **A white-cane replacement.** This is a complement. Nothing in the product
  should imply otherwise, in the UI or the marketing.

## Phasing

| Phase | Contents | Status |
|---|---|---|
| **0** | Repo, config, camera source layer, capture benchmarking | ✅ Done |
| **1** | Model registry, detection, geometric distance, tracking | Next |
| **2** | Saliency, phrasing, speech — the first genuinely usable walk | |
| **3** | Power policy: gating, adaptive rate, thermal governor | |
| **4** | On-demand OCR and VLM | |
| **5** | Taxonomy, dataset, fine-tune, INT8 export | |
| **6** | Android app | |
| **7** | Documentation package and web presentation | Ongoing throughout |

Phases 1–4 happen in Python on the desktop. Behavioural mistakes are cheap to
fix there and expensive to fix on a phone: tuning a saliency threshold is a
two-second edit and a re-run, versus a rebuild, reinstall and walk around the
building.

## How success is judged

Not by mAP. A detector with excellent mAP that announces a wall while missing a
staircase is a failed product. The measures that matter:

1. **Hazard recall** — of the genuinely dangerous things in a test walk, what
   fraction got announced, in time?
2. **Utterance precision** — of the things it said, what fraction were worth
   saying? Chattiness is a defect with a number attached.
3. **Glass-to-speech latency** — measured end to end with an external
   millisecond clock, not summed from stage timings.
4. **Battery cost per hour of walking**, measured on the target phone.
5. **Recovery** — after a camera dropout, does it come back without the user
   doing anything?

These need a recorded evaluation walk to measure against, which is a Phase 2
deliverable.
