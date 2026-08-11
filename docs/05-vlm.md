# Scene description

> Every number on this page was measured on a Pixel 8a (Tensor G3, Android 17)
> running Gemma 4 E2B through LiteRT-LM 0.15.0, on the weights named in
> `models/manifests/gemma-4-e2b-vlm.yaml`.

## What it is, and what it is deliberately not

Press volume-up and the phone says one sentence about what is in front of you.

```
"Ahead is a wooden door with a dark panel next to it."
```

That is the entire feature. It is **not** part of the safety path, and keeping
it out is a design decision rather than an oversight.

| | Detector + geometry | Gemma 4 |
|---|---|---|
| Latency | ~30 ms | 3.6-12 s |
| Memory | 182 MB total app | 3,189 MB |
| Runs | continuously, gated | only when asked |
| Evaluated against | held-out data, per-class recall | nothing quantitative |
| Wrong looks like | a missing box | a fluent, confident, false sentence |

The last row is the one that decides it. A missed detection is silence, and
silence is something a blind user already knows how to interpret. A language
model that says "the path ahead is clear" about a stairwell is worse than
saying nothing at all, and there is no way for the listener to catch it.

So: hazards come from the detector, which is bounded and measured. Gemma
answers a question, when asked, and nothing waits on its answer. If the weights
are never installed the app is exactly as safe as it is with them.

## Measured on the device

| | |
|---|---:|
| Backend selected | **GPU** (LiteRT MLDrift; XNNPACK for the vision adapter) |
| First-ever engine load | **48.9 s** |
| Engine load, kernel cache warm | **4.4-5.9 s** |
| Description, engine already resident | **3.6 s** |
| Description, first one after a load | 7.1-12.1 s |
| App memory, detector only | **182 MB PSS** |
| App memory, engine live | **3,189 MB PSS** |
| App memory, 90 s after last use | **253 MB PSS** |
| Model file | 2.59 GB |
| Compiled-kernel cache | 915 MB |

### And on a laptop

The same weights, the same manifest and the same adapter interface, through
`litert-lm`'s WebGPU backend on an M3 MacBook:

| | Pixel 8a (GPU) | M3 MacBook (WebGPU) |
|---|---:|---:|
| Engine load, cache warm | 4.4 s | **4.8 s** |
| Description | 3.6 s | **2.7 s** |

Close enough that the desktop app is a fair rehearsal for the phone — which is
the reason it drives the real pipeline rather than a mock. Press **Describe
scene**:

```bash
.venv/bin/python -m sarathi.desktop
```

Live from the laptop camera, first attempt:

```
"Ahead of you is a man with dark, curly hair wearing a light pink shirt."
```

One sentence, position first, plain nouns, no preamble about images — the
system instruction doing its job.

Read the logs yourself:

```bash
adb logcat -s SarathiVLM
adb shell dumpsys meminfo in.sarathi.app | grep TOTAL
```

### The kernel cache is worth 11x

The first load on a device compiles GPU shaders for a 2.6 GB model and takes
**48.9 seconds**. Passing `cacheDir` to `EngineConfig` persists the compiled
programs, and every load after that takes **4.4 seconds**. Same model, same
backend, same phone.

That single parameter is the difference between a feature nobody would press
twice and one that answers in a few seconds. It costs 915 MB of cache, which is
worth it and worth knowing about.

### Memory is the real constraint, not speed

3,189 MB is not what the vendor's model card reports (676 MB on a Galaxy S26
Ultra). The difference is what "memory" is being counted: PSS here includes the
memory-mapped model pages that are actually resident after a full prefill.

It works on the 8 GB Pixel 8a. It cannot work on a 4 GB phone, and much of the
audience this project is for has a 4 GB phone. Hence:

- `distribution: user_download` — the weights are never bundled, so installing
  the app costs 81 MB rather than 2.6 GB.
- `min_ram_gb: 8` in the manifest, revised up from a guess of 6 once this was
  measured.
- **Unload after 90 s idle**, which is measured to return the memory:
  3,189 MB → 253 MB. Without it the app would hold three gigabytes for the rest
  of the walk after one press.

### Why the first press says something different

```
engine not loaded  →  "Loading the scene model, one moment"
engine resident    →  "Looking"
```

A blind user cannot see a spinner. Saying "Looking" and then going quiet for
forty-nine seconds is how a working device convinces someone it has crashed, so
the acknowledgement is chosen to match the wait that is actually coming.

## The 1.9 GB mistake, recorded so nobody repeats it

`litert-community/gemma-4-E2B-it-litert-lm` publishes several builds. The
obvious pick is `gemma-4-E2B-it-gpu.litertlm`: smaller than the base build at
1.9 GB, and named after the backend the vendor recommends.

**It is text-only.** It loads without complaint in 11 s, and then fails the
moment an image is attached:

```
LiteRtLmJniException: Failed to create conversation:
NOT_FOUND: TF_LITE_VISION_ENCODER not found in the model.
```

Nothing on the model card says this. The container does. A `.litertlm` lists
its section names in the first kilobyte:

```bash
head -c 4096 gemma-4-E2B-it-gpu.litertlm | strings | grep tf_lite
# tf_lite_artisan_text_decoder          <- that is the whole list

head -c 4096 gemma-4-E2B-it.litertlm | strings | grep tf_lite
# tf_lite_mtp_drafter, tf_lite_prefill_decode, tf_lite_vision_adapter,
# tf_lite_vision_encoder, tf_lite_audio_encoder_hw, tf_lite_per_layer_embedder
```

Check before downloading, with a 4 KB range request:

```bash
curl -sL -r 0-4096 <url> | strings | grep tf_lite
```

Both implementations now run that check at load time — `hasVisionEncoder()` in
`SceneDescriber.kt`, `has_vision_encoder()` in `gemma_vlm.py` — so a text-only
build is refused in milliseconds instead of discovered eleven seconds into a
request the user is waiting on.

## How the answer is shaped

A general-purpose VLM describes a photograph. Someone standing in a place needs
to know what is there, which is a different question. The system instruction
does most of the work:

> You describe scenes for a blind person who is standing where the camera is
> pointing. Answer in one short sentence, under 30 words. Lead with what is
> directly ahead and closest. Name things plainly. Give positions as left,
> ahead or right. Do not mention the image, the photo, or the camera. Do not
> guess at anything you cannot see. If the view is too dark or blurred to read,
> say exactly that.

Then three mechanical guards, because an instruction is a request and not a
constraint:

- **`maxOutputToken = 64`.** A hard ceiling on generation, which bounds the
  wait and the length of speech.
- **Opener stripping.** Instruction-tuned models like "The image shows…", which
  costs a second of speech to convey nothing. Stripped, along with markdown,
  which is worse than useless aloud.
- **220-character cap, cut at a word boundary.** A paragraph read aloud
  occupies the audio channel long enough to displace a real hazard
  announcement. This is a safety limit wearing the clothes of a style rule.

`tidy()` is a pure function in both languages and is unit-tested without the
model (`prototype/tests/test_vlm.py`), because it decides what a blind person
actually hears.

The prompt itself lives in `phrases/{en,hi}.yaml` next to the spoken strings —
same translation job, since the answer comes back in the language it was asked
in.

## Swapping the model

Nothing above is specific to Gemma. The manifest names the file, the runtime
and the licence; the adapter implements `SceneDescriber.describe()`. To try a
different VLM, write a manifest and — only if it needs a different runtime — an
adapter. No pipeline code changes.

```yaml
runtime:
  android: litert-lm
  prototype: litert-lm
files:
  litertlm:
    path: some-other-model.litertlm
```

A regression test reads every shipped manifest and fails if one names a runtime
it cannot actually resolve weights for. That test exists because
`runtime.android: litert-lm` silently resolved nothing for a while, and the app
reported the feature as "not installed" while sitting on 1.9 GB of weights.

## Licensing

Gemma 4 is **Apache 2.0** and the repository is ungated, both verified against
the Hugging Face API rather than assumed. This is a correction to an earlier
decision in this project: Gemma 3n shipped under the Gemma Terms of Use, whose
use-restriction policy must be passed downstream and which AGPL-3.0 cannot
absorb. That blocker is gone. What remains is a size decision, which is a much
better place to be.

## Honest gaps

- **No quality evaluation.** Every description here was correct, and "every"
  means a handful, of one doorway. There is no benchmark, no hallucination
  rate, and no measurement of how it degrades in the dark or in motion. Until
  there is, this stays off the safety path — which is where it was always
  going to stay anyway.
- **Hindi output is unverified.** The prompt is translated and the model is
  multilingual, so it should answer in Hindi. Nobody has confirmed it does, or
  that what it says is idiomatic.
- **Audio input is unused.** The E-series encodes audio too, and the container
  we ship carries `tf_lite_audio_encoder_hw`. There may be something there for
  a user who wants to ask a question out loud rather than press a button.
