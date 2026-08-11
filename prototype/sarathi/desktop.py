"""Sarathi on the desktop: the same pipeline, with a window.

Why this exists at all, given the product is a phone app:

Everything in this project is decided by measurement, and for a long time
every measurement arrived as a line in a log. That is fine for a threshold and
useless for a demonstration — nobody can be handed a terminal and told to
imagine the walk. It also hid a real bug for a whole session: analysis frames
were reaching the detector rotated ninety degrees, live scores never rose above
0.06, and the logs said only "0 detections", which is exactly what an empty
room says too. One glance at boxes drawn on a picture would have caught it
immediately.

So this is a window onto the real thing, not a mock. It drives
`sarathi.runtime.Pipeline` — the same detector, geometry, tracker, saliency,
phrasing and speech that run on the phone, reading the same manifests and the
same phrase tables. What is on screen is what the product would say.

Tkinter, deliberately: it is in the standard library, so the demo has no
dependency the pipeline does not already have, and it runs anywhere Python
does.

    python -m sarathi.desktop                    # default camera
    python -m sarathi.desktop --source walk.mp4  # a recorded walk
    python -m sarathi.desktop --speak            # with the voice
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Any

from .config import load_config
from .sources import LatestFrame, SourceError, create_source
from .util.log import get_logger

log = get_logger(__name__)

# One dark palette, stated once. The window is often pointed at a camera in a
# room with the lights down, and a bright grey chrome around a dark video feed
# is the thing that makes a demo look unfinished.
INK = "#101315"
PANEL = "#171B1E"
RULE = "#262C30"
PAPER = "#E7EAE6"
MUTED = "#7F8A8E"
AMBER = "#E5A83C"
SLATE = "#262C30"
CYAN = "#78D6C8"

#: Hazard level -> outline colour. Four colours that mean urgency, rather than
#: twenty-six that mean class identity: the question a viewer is actually
#: asking is "has it understood what matters here", and a glance should answer
#: it.
HAZARD_COLOURS = {
    "CRITICAL": "#FF5252",
    "HIGH": "#FF9130",
    "MEDIUM": "#FFD147",
    "LOW": "#78D6A8",
}


def _weights_present(registry: Any, manifest: Any) -> bool:
    """Whether this model's weights are on disk for the prototype runtime."""
    if getattr(manifest, "vendored_weights", False):
        return True
    try:
        spec = manifest.file_for(registry.runtime)
        return spec.resolve(registry.weights_dir).exists()
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


class DesktopApp:
    """A window over the live pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.running = False
        self.frames = queue.Queue(maxsize=1)
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.orphans: list[threading.Thread] = []
        self._orphans = self.orphans
        #: Incremented per run. A payload from an earlier run is discarded
        #: rather than drawn, so a dying worker cannot report into a live one.
        self.generation = 0
        self.latest: dict[str, Any] = {}
        self.spoken_history: list[str] = []
        self.started_at = 0.0
        #: Most recent frame, kept so an on-demand request has something to
        #: work on the instant it is pressed rather than waiting for the next.
        self.last_image: Any = None
        #: Loaded on first use and kept, because loading is the expensive half.
        self.describer: Any = None
        self.reader: Any = None
        self.on_demand_busy = False
        self.answers = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Sarathi — assistive vision")
        self.root.configure(bg=INK)
        self.root.geometry("1180x760")
        self.root.minsize(880, 600)
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        style = ttk.Style()
        with_theme = "clam" if "clam" in style.theme_names() else style.theme_use()
        style.theme_use(with_theme)
        style.configure("TFrame", background=INK)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=INK, foreground=PAPER)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("TkDefaultFont", 10))
        style.configure("Stat.TLabel", background=PANEL, foreground=PAPER, font=("TkFixedFont", 13))
        style.configure("Head.TLabel", background=PANEL, foreground=MUTED,
                        font=("TkDefaultFont", 9, "bold"))

        outer = ttk.Frame(self.root, style="TFrame")
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        # -- the feed --
        left = ttk.Frame(outer, style="TFrame")
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # What it said, given the largest type on the screen, because it is the
        # product's actual output. Everything else is diagnostics.
        self.spoken = tk.Label(
            left, text="Point the camera ahead.",
            bg=INK, fg=PAPER, font=("TkDefaultFont", 17), anchor="w", justify="left",
            wraplength=760, padx=4, pady=12,
        )
        self.spoken.pack(fill="x")

        controls = ttk.Frame(left, style="TFrame")
        controls.pack(fill="x", pady=(4, 0))
        self.start_button = tk.Button(
            controls, text="Start", command=self._toggle,
            bg=AMBER, fg=INK, activebackground=AMBER, relief="flat",
            font=("TkDefaultFont", 13, "bold"), padx=26, pady=9, cursor="hand2",
        )
        self.start_button.pack(side="left")
        # On-demand tiers, the same two the phone puts on volume-up. They are
        # buttons here rather than gestures because a laptop has no volume
        # rocker to hold, and because a demo needs them to be visible.
        self.describe_button = tk.Button(
            controls, text="Describe scene", command=self._describe,
            bg=SLATE, fg=PAPER, activebackground=SLATE, relief="flat",
            font=("TkDefaultFont", 12), padx=16, pady=9, cursor="hand2",
        )
        self.describe_button.pack(side="left", padx=(10, 0))
        self.read_button = tk.Button(
            controls, text="Read text", command=self._read_text,
            bg=SLATE, fg=PAPER, activebackground=SLATE, relief="flat",
            font=("TkDefaultFont", 12), padx=16, pady=9, cursor="hand2",
        )
        self.read_button.pack(side="left", padx=(8, 0))

        tk.Label(controls, text="  detector  ", bg=INK, fg=MUTED).pack(side="left")
        self.model_choice = ttk.Combobox(controls, state="readonly", width=22, values=[])
        self.model_choice.pack(side="left")
        self.model_choice.bind("<<ComboboxSelected>>", lambda _e: self._model_changed())

        # -- the readouts --
        right = ttk.Frame(outer, style="Panel.TFrame", width=320)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)

        self.status = tk.Label(
            right, text="STOPPED", bg=MUTED, fg=INK,
            font=("TkDefaultFont", 11, "bold"), pady=7,
        )
        self.status.pack(fill="x")

        self.stats = tk.Label(
            right, text="", bg=PANEL, fg=PAPER, font=("TkFixedFont", 12),
            justify="left", anchor="nw", padx=14, pady=14,
        )
        self.stats.pack(fill="x")

        ttk.Label(right, text="DETECTED", style="Head.TLabel").pack(
            anchor="w", padx=14, pady=(6, 4)
        )
        self.detected = tk.Listbox(
            right, bg=PANEL, fg=PAPER, highlightthickness=0, borderwidth=0,
            font=("TkFixedFont", 11), selectbackground=RULE, height=11,
        )
        self.detected.pack(fill="x", padx=10)

        ttk.Label(right, text="SPOKEN", style="Head.TLabel").pack(
            anchor="w", padx=14, pady=(14, 4)
        )
        self.said = tk.Listbox(
            right, bg=PANEL, fg=AMBER, highlightthickness=0, borderwidth=0,
            font=("TkFixedFont", 11), selectbackground=RULE, height=9,
        )
        self.said.pack(fill="both", expand=True, padx=10, pady=(0, 12))

        self._load_models()

    def _load_models(self) -> None:
        from .models import ModelRegistry

        try:
            registry = ModelRegistry()
            # Weights must exist, not merely be declared. `yolox-nano` has a
            # manifest and no exported .onnx, and offering it means the picker
            # can start a run that dies immediately - while the window keeps
            # the previous model's numbers under the new model's name, which is
            # the exact "plausible but wrong" failure this project keeps
            # finding. A control that cannot work does not belong on screen.
            ids = sorted(
                m.id for m in registry.manifests.values()
                if getattr(m.task, "value", m.task) == "detection"
                and m.loadable
                and _weights_present(registry, m)
            )
        except Exception as exc:  # noqa: BLE001 - a demo must still open
            log.warning("could not list models: %s", exc)
            ids = []
        if not ids:
            ids = [self.args.detector]
        self.model_choice["values"] = ids
        current = self.args.detector if self.args.detector in ids else ids[0]
        self.model_choice.set(current)
        self.args.detector = current

    def _model_changed(self) -> None:
        chosen = self.model_choice.get()
        if chosen == self.args.detector:
            return
        self.args.detector = chosen
        # Restarting is the honest thing: swapping a detector mid-run would
        # leave tracks and cooldowns from the previous model attached to
        # objects the new one has never seen.
        if self.running:
            self._stop()
            self._start()

    # -- lifecycle --------------------------------------------------------

    def _toggle(self) -> None:
        self._stop() if self.running else self._start()

    def _start(self) -> None:
        # A fresh flag per run, so a previous worker that has not finished
        # cannot be resurrected by this one clearing the flag it is watching.
        self.stop_flag = threading.Event()
        self.generation += 1
        # Flag and source are passed in, not read off self inside the thread.
        # Reading them dynamically is what let a replaced flag un-stop an
        # orphan: `_run` checked `self.stop_flag`, `_start` rebound it, and the
        # abandoned worker saw a fresh un-set Event and carried on.
        self.worker = threading.Thread(
            target=self._run,
            args=(self.stop_flag, self.generation, self.args.source),
            name="sarathi-pipeline",
            daemon=True,
        )
        self.worker.start()
        self.running = True
        self.started_at = time.monotonic()
        self.start_button.configure(text="Stop", bg=RULE, fg=PAPER)
        self.status.configure(text="RUNNING", bg=AMBER, fg=INK)
        self.model_choice.configure(state="disabled")
        self._set_on_demand_enabled(True)

    def _stop(self) -> None:
        """Stop, without blocking the window and without abandoning a thread.

        The obvious version - `join(timeout=3)` then drop the reference - has
        two faults that only appear together. The join runs on the Tk main
        thread, so a camera blocked in `open()` freezes the window for three
        seconds and reads as a hang. And when the join times out the worker is
        still alive, still owns the stop flag, and the next Start clears that
        flag and revives it: measured, an abandoned RTSP attempt failed thirty
        seconds into a *later, healthy* session and stopped it, reporting an
        error about a URL the user had already given up on.

        So each run gets its own flag, and a worker that outlives its stop is
        remembered rather than forgotten - it can no longer be un-stopped, and
        it cannot push into a session it does not belong to.
        """
        self.stop_flag.set()
        worker = self.worker
        if worker is not None:
            worker.join(timeout=0.25)
            if worker.is_alive():
                # Blocked in a camera open. It will exit on its own; until
                # then it holds a flag that is already set, so it is inert.
                log.info("camera thread still finishing; it will exit on its own")
                self._orphans.append(worker)
        self.worker = None
        self.running = False
        self.start_button.configure(text="Start", bg=AMBER, fg=INK)
        self.status.configure(text="STOPPED", bg=MUTED, fg=INK)
        self.model_choice.configure(state="readonly")
        self.last_image = None
        if not self.on_demand_busy:
            self._set_on_demand_enabled(False)

    def _on_close(self) -> None:
        if self.running:
            self._stop()
        for model in (self.describer, self.reader):
            if model is not None:
                try:
                    model.close()
                except Exception:  # noqa: BLE001 - closing must not raise
                    pass
        self.root.destroy()

    # -- the pipeline thread ----------------------------------------------

    def _run(self, stop: threading.Event, generation: int, source: str) -> None:
        """Own thread: a stalled camera must not freeze the window."""
        from .guidance.speech import EarconPlayer, MacSpeaker, NullSpeaker, VoiceOutput
        from .runtime import Pipeline, PipelineConfig, SchedulerConfig

        speaker: Any = NullSpeaker()
        if self.args.speak:
            try:
                speaker = MacSpeaker()
            except RuntimeError as exc:
                log.warning("%s - running silently", exc)

        pipeline = Pipeline(
            PipelineConfig(
                detector=self.args.detector,
                depth=self.args.depth,
                lang=self.args.lang,
                speak=self.args.speak,
                camera_height_m=self.args.camera_height,
                camera_pitch_deg=self.args.camera_pitch,
                scheduler=SchedulerConfig(max_inference_hz=self.args.max_hz),
            ),
            voice=VoiceOutput(speaker, EarconPlayer(enabled=self.args.speak)),
        )

        try:
            cam = LatestFrame(create_source(source), reconnect=True).start()
        except SourceError as exc:
            self._push({"error": str(exc), "generation": generation})
            return

        last_detections: list[Any] = []
        try:
            while not stop.is_set():
                frame = cam.get(timeout=2.0)
                if frame is None:
                    if cam.ended:
                        break
                    continue
                result = pipeline.process(frame)

                # A gated frame produces no detections, which is the scheduler
                # working - not the scene emptying. Holding the previous boxes
                # keeps the display honest about what the system currently
                # believes, instead of flickering to nothing at 8 Hz.
                if result.decision.run:
                    last_detections = result.detections

                self._push({
                    "image": frame.image,
                    "detections": last_detections,
                    "result": result,
                    "generation": generation,
                })
        finally:
            cam.stop()
            # The clip ran out, or the camera went away. Say so, rather than
            # leaving the window on a frozen last frame with a RUNNING badge
            # and a disabled model picker - which is how every `--source clip`
            # demo used to end.
            if not stop.is_set():
                self._push({"ended": True, "generation": generation})

    def _push(self, payload: dict[str, Any]) -> None:
        """Hand the newest frame to the UI, dropping any it has not drawn.

        The same drop-don't-queue rule the speech layer uses, for the same
        reason: a backlog of frames renders a world the camera has already left.
        """
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(payload)
        except queue.Full:
            pass

    # -- on-demand tiers ---------------------------------------------------

    def _describe(self) -> None:
        """Ask the VLM what is in front of the camera.

        Deliberately not part of the per-frame pipeline. Gemma takes seconds
        and holds gigabytes; more importantly it is wrong in fluent, confident
        prose, which is the worst possible failure mode for someone who cannot
        check it. Hazards come from the detector, which is bounded and
        measured. This answers a question, when asked.
        """
        self._on_demand(
            "Describing…",
            lambda image: self._vlm().describe(image),
            loading="Loading Gemma (first time is slower)…",
            needs_load=self.describer is None,
        )

    def _read_text(self) -> None:
        """Read any text in view - a sign, a door number, a label."""
        def read(image):
            lines = self._ocr().read(image)
            if not lines:
                return "(no text found)"
            return ". ".join(text for text, _ in lines)

        self._on_demand("Reading text…", read, needs_load=self.reader is None)

    def _vlm(self):
        """The scene-description model, loaded on first use and kept.

        Resolved from the manifests rather than named here, so dropping a new
        VLM manifest into `models/manifests/` makes it usable without touching
        this file - which is the point of the manifest system, and was
        previously true of the detector only.
        """
        if self.describer is None:
            # getattr, not attribute access: DesktopApp is constructed with a
            # plain Namespace in tests and by anything embedding it, and a
            # missing optional flag should not take out the feature.
            self.describer = self._load_first("vlm", getattr(self.args, "vlm", None))
        return self.describer

    def _ocr(self):
        if self.reader is None:
            self.reader = self._load_first("ocr", None)
        return self.reader

    def _load_first(self, task: str, preferred: str | None):
        from .models import ModelRegistry

        registry = ModelRegistry()
        candidates = [
            m for m in registry.manifests.values()
            if getattr(m.task, "value", m.task) == task
            and m.loadable
            and _weights_present(registry, m)
        ]
        if preferred:
            candidates = [m for m in candidates if m.id == preferred] or candidates
        if not candidates:
            raise RuntimeError(
                f"no usable {task} model. Weights go in models/weights/; "
                f"see `sarathi models` for what each manifest expects."
            )
        return registry.load(sorted(candidates, key=lambda m: m.id)[0].id)

    def _on_demand(self, busy_text, work, *, loading=None, needs_load=False) -> None:
        """Run `work` on the latest frame, off the UI thread.

        Dropped rather than queued while one is already running, which is the
        same rule the speech layer follows: an answer about a scene the user
        has walked out of is worse than no answer.
        """
        if self.on_demand_busy:
            log.info("on-demand request dropped; one already running")
            return
        image = self.last_image
        if image is None:
            self.spoken.configure(text="Start the camera first.")
            return

        self.on_demand_busy = True
        self._set_on_demand_enabled(False)
        # Said before the work starts. Loading the model alone takes seconds,
        # and a window that goes still is indistinguishable from one that has
        # hung.
        self.spoken.configure(text=(loading if needs_load else busy_text))
        self.root.update_idletasks()

        frame = image.copy()

        def run():
            try:
                answer = work(frame)
            except Exception as exc:  # noqa: BLE001 - report, never crash the demo
                log.warning("on-demand failed: %s", exc)
                answer = f"({type(exc).__name__}: {str(exc)[:120]})"
            self.answers.put(answer)

        threading.Thread(target=run, name="sarathi-on-demand", daemon=True).start()

    def _set_on_demand_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (self.describe_button, self.read_button):
            button.configure(state=state)

    # -- drawing ----------------------------------------------------------

    def _tick(self) -> None:
        try:
            answer = self.answers.get_nowait()
        except queue.Empty:
            answer = None
        if answer is not None:
            self.on_demand_busy = False
            self._set_on_demand_enabled(self.running)
            self.spoken.configure(text=f"“{answer}”")
            self.said.insert(tk.END, f"{time.strftime('%H:%M:%S')}  {answer}")
            self.said.see(tk.END)

        try:
            payload = self.frames.get_nowait()
        except queue.Empty:
            payload = None

        if payload is not None:
            if payload.get("generation") != self.generation:
                pass          # from a run that has already been stopped
            elif "error" in payload:
                self.spoken.configure(text=f"Camera error: {payload['error']}")
                self._stop()
            elif payload.get("ended"):
                self.spoken.configure(text="Source ended.")
                self._stop()
            else:
                self._draw(payload)

        self.root.after(33, self._tick)

    def _draw(self, payload: dict[str, Any]) -> None:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageTk

        image = payload["image"]
        self.last_image = image
        detections = payload["detections"]
        result = payload["result"]

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 10 or height < 10:
            return

        # BGR from OpenCV; RGB for PIL. Getting this backwards is invisible on
        # a grey wall and obvious on a face, which is a bad way to find out.
        rgb = image[:, :, ::-1]
        source_h, source_w = rgb.shape[:2]
        scale = min(width / source_w, height / source_h)
        drawn = Image.fromarray(np.ascontiguousarray(rgb)).resize(
            (max(1, int(source_w * scale)), max(1, int(source_h * scale)))
        )

        canvas_draw = ImageDraw.Draw(drawn)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
        except OSError:
            font = ImageFont.load_default()

        self.detected.delete(0, tk.END)
        for det in detections:
            hazard = getattr(det.hazard, "name", str(det.hazard))
            colour = HAZARD_COLOURS.get(hazard, "#78D6A8")
            x1, y1, x2, y2 = (v * scale for v in det.box)
            canvas_draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
            # Feet, matching the phrase books and the phone overlay.
            distance = f"  {det.distance_m * 3.28084:.0f} ft" if det.distance_m else ""
            caption = f"{det.label}{distance}"
            text_w = canvas_draw.textlength(caption, font=font)
            label_top = y1 - 21 if y1 > 24 else y2 + 2
            canvas_draw.rectangle(
                [x1, label_top, x1 + text_w + 10, label_top + 20], fill=colour
            )
            canvas_draw.text((x1 + 5, label_top + 2), caption, fill="#101315", font=font)
            self.detected.insert(
                tk.END, f"{det.label:<16}{det.score:.2f}{distance:>9}"
            )

        photo = ImageTk.PhotoImage(drawn)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=photo, anchor="center")
        self.canvas._photo = photo  # keep a reference or Tk garbage-collects it

        # Shown whether or not audio is on. `spoke` now means "passed the
        # speech policy" rather than "a speaker was attached", so muting the
        # demo no longer empties the one panel the demo is about.
        if result.utterance is not None and result.spoke:
            text = result.utterance.text
            if not self.spoken_history or self.spoken_history[-1] != text:
                self.spoken_history.append(text)
                self.said.insert(tk.END, f"{time.strftime('%H:%M:%S')}  {text}")
                self.said.see(tk.END)
            self.spoken.configure(text=f"“{text}”")

        elapsed = max(1e-6, time.monotonic() - self.started_at)
        best = max((d.score for d in detections), default=0.0)
        skip = "-" if result.decision.run else getattr(
            result.decision.reason, "name", str(result.decision.reason)
        ).lower()
        lines = [
            f"{self.args.detector}",
            "",
            f"{result.total_ms:6.1f} ms  this frame",
            f"{result.frame_seq / elapsed:6.1f} fps captured",
            f"{result.decision.target_hz:6.1f} Hz target",
            f"{len(detections):6d}     detected",
            f"{best:6.2f}     best score",
            f"{getattr(result.decision.activity, 'name', '?').lower():>6}     activity",
            f"{skip:>6}     skip reason",
        ]
        for stage, ms in sorted(result.stage_ms.items()):
            lines.append(f"{ms:6.1f} ms  {stage}")
        self.stats.configure(text="\n".join(lines))

    def run(self) -> None:
        self.root.after(60, self._tick)
        if self.args.autostart:
            # Straight into the live view. This is a demonstration tool - a
            # window that opens idle and waits to be told to begin makes the
            # first ten seconds of every demo a hunt for a button, in front of
            # an audience. --no-autostart for the times you want to choose a
            # model or a source first.
            self.root.after(120, self._start)
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sarathi.desktop",
        description="Sarathi with a window: the real pipeline, visible.",
    )
    parser.add_argument("--source", default="0", help="camera index, video path, or URL")
    parser.add_argument("--detector", default="yolo11n-coco-320")
    parser.add_argument("--depth", default=None, help="depth manifest id, or omit for none")
    parser.add_argument(
        "--vlm", default=None,
        help="scene-description manifest id (default: the first usable one)",
    )
    parser.add_argument("--lang", default="en", choices=("en", "hi"))
    parser.add_argument("--speak", action="store_true", help="speak out loud as well as show")
    parser.add_argument(
        "--no-autostart", dest="autostart", action="store_false",
        help="open idle instead of starting the camera immediately",
    )
    parser.set_defaults(autostart=True)
    parser.add_argument("--max-hz", type=float, default=8.0)
    # Matched to PipelineConfig and Android's CameraModel, not chosen
    # separately. Different defaults here would mean the window on stage shows
    # different distances from the phone in your hand for the same scene, and
    # the whole point of this app is that what it shows is what the product
    # does. The ground-plane tier wants ~20 degrees of downward pitch to have
    # near-field floor to fit against - pass --camera-pitch 20 for that, on
    # both sides.
    parser.add_argument("--camera-height", type=float, default=1.20)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        load_config(args.config, args.set)
    except Exception as exc:  # noqa: BLE001 - never block the demo on config
        log.warning("config not loaded (%s); using defaults", exc)

    DesktopApp(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
