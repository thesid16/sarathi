# Installing Sarathi

Three ways to try it. The phone is the product; the others need nothing
installed at all.

---

## On an Android phone

**Requires** Android 8.0 or newer. Tested on a Pixel 8a.

1. Copy `sarathi.apk` to the phone — cable, email, or any file-sharing app.
2. Open it. Android will say *"For your security, your phone is not allowed to
   install unknown apps from this source"* — tap **Settings**, turn on the
   permission, and go back. This is normal for any app not from the Play Store.
3. Tap **Install**, then **Open**.
4. Allow **Camera**. Allow notifications too — the app runs as a foreground
   service so it keeps working with the screen off, and Android requires a
   notification for that.

**It starts guiding by itself.** There is nothing else to set up.

### Using it without looking

| | |
|---|---|
| **Volume down** | start / stop |
| **Volume up**, tapped | describe the scene |
| **Volume up**, held | read any text |

The screen is optional. It exists so a sighted helper, or you, can see what the
app is seeing — the live camera with boxes around what it has found, what it
last said, and why it is staying quiet when it is.

### First run

Point it at a person, a chair or a doorway from a couple of metres away. It
should box them and say something. If it says nothing, that is often correct —
it deliberately stays quiet about things that are not in your way, and the line
under the camera tells you which: `quiet — laptop: low hazard, context only`.

### Scene description needs a separate download

The **Describe scene** button uses Gemma 4, a 2.5 GB model that is not in the
APK. Bundling it would make the app a 2.6 GB install on phones that often have
4 GB of storage in total, so it is optional.

Without it the button says the model is not installed and everything else works
normally.

To add it (needs 8 GB RAM and about 3.5 GB free):

```bash
# download once, ~2.5 GB
curl -L -o gemma-4-E2B-it.litertlm \
  https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm

adb push gemma-4-E2B-it.litertlm /data/local/tmp/
adb shell "run-as in.sarathi.app mkdir -p files/models"
adb shell "run-as in.sarathi.app cp /data/local/tmp/gemma-4-E2B-it.litertlm files/models/"
```

Make sure it is the file named exactly `gemma-4-E2B-it.litertlm`. The build
called `-gpu` is smaller and *cannot see* — it contains no vision encoder and
fails the moment an image is attached.

### Choosing a detector

**Model** offers four. Stop guidance first.

| | For |
|---|---|
| `sarathi26-320` | **Walking outdoors.** Stairs, manholes, tactile paving. Fastest. |
| `yolo11n-coco-320` | Indoors — people, chairs, doorways. The default. |
| `yolo11n-coco-256` | A hot phone or a long walk. |
| `yolo11s-coco-320` | Most accurate, 2.5× slower. |

Only `sarathi26-320` knows what a staircase or an open manhole is; COCO
contains neither.

### If it feels slow

**Unplug it.** Charging alone puts a Pixel 8a's CPU near its thermal limit, so
the app drops to 1 Hz to avoid making it worse. The stats line says
`(heat-limited)` when that is what is happening. On battery it runs several
times faster.

---

## In a browser

No install. Works on a laptop or a phone, and the video never leaves the
device.

```bash
cd web
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

Opening `index.html` directly will not work — browsers block loading the model
from `file://`, and cameras need `localhost` or HTTPS.

To share it, put the `web/` folder on any static host. GitHub Pages works.

This is a demonstration, not the product: same model and same decoding as the
phone, but no distances, because without a calibrated camera and a tilt sensor
any number would be a guess.

---

## On a Mac or Linux desktop

The real pipeline in a window — the same detector, geometry, saliency and
phrasing that run on the phone.

**Double-click `run-desktop.command`** in the project root.

Or from a terminal (note the quotes — the path has a space in it):

```bash
cd "path/to/STM Vission App/prototype"
uv venv && uv pip install -e .          # first time only
.venv/bin/python -m sarathi.desktop
```

| | |
|---|---|
| `--source walk.mp4` | replay a recording instead of the camera |
| `--speak` | speak out loud as well as show |
| `--lang hi` | Hindi |
| `--camera-pitch 20` | tell it how far down the camera points — distances depend on this more than anything else |

**Describe scene** and **Read text** work here too. Scene description needs the
Gemma weights in `models/weights/`; it takes about 5 seconds to load and 3
seconds to answer on an M3.

---

## What is in the box

```
sarathi.apk            the Android app, signed and installable
web/                   the browser demo
prototype/             the Python pipeline, desktop app, training, evaluation
android/               the Android source
models/                manifests, labels, and small weights
phrases/               everything the app says, English and Hindi
docs/                  design notes and every measurement
run-desktop.command    double-click to run the desktop app
INSTALL.md             this file
docs/09-edit-guide.md  how to change any of it
```

## If something goes wrong

| | |
|---|---|
| "App not installed" | An older Sarathi signed with a different key. Uninstall it first. |
| Black screen where the camera should be | Camera permission was denied. Settings → Apps → Sarathi → Permissions. |
| Detects nothing at all | Check `adb logcat -s SarathiService \| grep self-test`. A healthy line reads `maxScore=0.754 detections=2 [clock 0.75, car 0.68]`. |
| Says nothing while clearly detecting | Usually correct. Read the `quiet — …` line; it says which rule kept it silent. |
| Describe scene does nothing | The Gemma weights are not installed. See above. |

Licensed **AGPL-3.0**. Model weights and datasets carry their own terms, listed
in the repository with generated attribution.
