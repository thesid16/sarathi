# Results

YOLO11n fine-tuned at 320 px, 120 epochs, 2.5 hours on one RTX PRO 6000.
14,818 train / 3,427 val / 40,772 domain-gap images. 26 shipped classes.

Two sets, two questions, deliberately not averaged.

## Did it learn what it was taught?

`val` — held-out blocks from the training sources, stratified so every source
and therefore every class appears.

**mAP50 0.597 · mAP50-95 0.406 · 26 of 26 classes evaluated**

| class | P | R | mAP50 |
|---|---:|---:|---:|
| **stairs_up** | 0.930 | **0.983** | **0.986** |
| **stairs_down** | 0.932 | **0.950** | **0.980** |
| **manhole_cover** | 0.772 | 0.932 | 0.907 |
| **open_manhole** | 0.838 | **0.930** | **0.954** |
| dog | 0.866 | 0.731 | 0.804 |
| pole | 0.684 | 0.727 | 0.757 |
| fire_hydrant | 0.913 | 0.725 | 0.813 |
| sign_board | 0.838 | 0.683 | 0.750 |
| motorcycle | 0.725 | 0.619 | 0.660 |
| bus | 0.714 | 0.579 | 0.643 |
| traffic_light_red | 0.740 | 0.562 | 0.642 |
| traffic_light_green | 0.731 | 0.547 | 0.636 |
| tree | 0.696 | 0.528 | 0.617 |
| step_up | 0.666 | 0.482 | 0.533 |
| step_down | 0.569 | 0.473 | 0.493 |
| car | 0.720 | 0.455 | 0.538 |
| cycle_rickshaw | 0.487 | 0.444 | 0.491 |
| person | 0.765 | 0.444 | 0.552 |
| traffic_cone | 0.795 | 0.439 | 0.546 |
| truck | 0.781 | 0.420 | 0.560 |
| bicycle | 0.664 | 0.358 | 0.403 |
| crosswalk | 0.721 | 0.313 | 0.432 |
| bin | 0.473 | 0.244 | 0.309 |
| tactile_paving | 0.151 | 0.087 | 0.043 |
| **barrier** | **0.000** | **0.000** | 0.019 |

**The hazard classes are the strongest thing in the model.** Stairs and
manholes — precisely the things a white cane finds late and no other stage of
this pipeline can catch — sit at 0.93–0.98 recall. That is the right way round,
and it is the single most encouraging number in the project.

### Three classes that do not work

- **`barrier` — 0.000 recall** on 4,402 training instances. Not a data-volume
  problem, so it is a label problem: WOTR's `roadblock` almost certainly covers
  several visually unrelated things mapped onto one class. Needs the source
  annotations inspected, not more epochs.
- **`tactile_paving` — 0.087.** 2,381 instances of a long, thin, ground-plane
  texture. A 320 px letterboxed input may simply not resolve it; it is also the
  class whose Hindi name is flagged for review, which is a coincidence worth
  nothing except that the class needs attention generally.
- **`bin` — 0.244.** Marginal.

All three are reported as effectively undetected rather than being allowed to
sit in the label set looking supported.

## Does it transfer to the country the product is for?

`val_domain` — the entire India Driving Dataset, never trained on, held out for
licensing reasons that turned out to be the better methodology anyway.

**mAP50 0.219 · mAP50-95 0.129 · 7 classes overlap**

| class | P | R | mAP50 |
|---|---:|---:|---:|
| car | 0.528 | 0.417 | 0.418 |
| bus | 0.419 | 0.317 | 0.292 |
| motorcycle | 0.753 | 0.244 | 0.320 |
| truck | 0.344 | 0.229 | 0.179 |
| person | 0.618 | 0.217 | 0.248 |
| bicycle | 0.130 | 0.105 | 0.057 |
| sign_board | 0.307 | 0.027 | 0.023 |

**A 2.7× drop, and that is the number that matters.** 0.597 in-distribution
against 0.219 on Indian roads is the domain gap made explicit: the model was
trained on Chinese footpaths (WOTR) and Chinese campus stairs (Mendeley), and
Indian street scenes are denser, more cluttered and differently lit.

`person` falling from 0.444 to 0.217 is the clearest single illustration.
`sign_board` collapses from 0.683 to 0.027 — Indian signage looks nothing like
the Chinese signage it learned on.

### The caveat that matters most

**The hazard classes are not in this table.** IDD is a driving dataset; it has
no stairs, no manholes and no tactile paving. So the classes that score 0.93+
in-distribution are **completely unmeasured on Indian streets**, and nothing
here licenses a claim that they transfer.

That gap cannot be closed with the data this project has. It needs footage of
Indian footpaths with the hazards annotated — which is exactly the collection
that was ruled out at the start, and remains the single highest-value addition
anyone could make.

## What these numbers are not

- Not a claim about the deployed model. These are fp32 GPU numbers; the phone
  runs a dynamic-range quantized export.
- Not a guidance metric. mAP says nothing about whether the right thing was
  said at the right moment — `sarathi bench` measures that, and it needs an
  annotated evaluation walk that does not exist yet.
- Not final. 120 epochs on 14,818 images is a first honest baseline, not a
  tuned result.
