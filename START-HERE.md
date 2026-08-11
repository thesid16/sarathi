# Sarathi — start here

An offline assistive-vision app: it watches through a camera and tells a blind
user what is in front of them. No internet, no account, nothing leaves the
device.

## Pick one

| I want to… | Do this |
|---|---|
| **Try it on a phone** | Copy `sarathi.apk` to an Android phone and open it. Allow the camera. It starts by itself. |
| **See it without installing** | `cd web && python3 -m http.server 8000`, then open <http://localhost:8000> |
| **See it on this computer** | Double-click `run-desktop.command` |
| **Change something** | [`docs/09-edit-guide.md`](docs/09-edit-guide.md) |
| **Know what was built and measured** | [`README.md`](README.md), then [`docs/`](docs/) |

Full install notes, including the optional 2.5 GB scene-description model and
what to do when something misbehaves: [`INSTALL.md`](INSTALL.md).

## What is here

```
sarathi.apk            the Android app — signed, installs, starts on its own
web/                   browser demo, runs the same model client-side
prototype/             the Python pipeline + the desktop app
android/               Android source
models/                model manifests, labels, weights
phrases/               everything it says, English and हिन्दी
training/              taxonomy, size priors, dataset definitions
docs/                  design, measurements, results, edit guide
```

Not included: the 2.5 GB Gemma scene-description weights (download separately —
see `INSTALL.md`), the datasets, and the depth model. Everything needed to run,
read, build and modify the app is here.

## In one paragraph

The hard problem is not detection. A detector at 5 Hz produces hundreds of true
detections a minute, and speaking them is not assistance — it is noise. The
documented reason people abandon assistive vision tools is that they talk too
much, so **restraint is the product**: work is split into tiers by how often it
must happen, ranking is by metres to the side of the walking line rather than
by bearing, and the app stays quiet unless something is genuinely in the way —
and tells you which rule kept it quiet.

Everything in `docs/` is measured on a real Pixel 8a, including the parts that
did not work.

**AGPL-3.0.** Model weights and datasets carry their own terms, listed in the
repository with generated attribution.

An independent open-source project by Siddharth Patel, developed during an
internship at STMicroelectronics. Not an STMicroelectronics product.
