"""Command-line entry point.

    sarathi probe --source 0                       # local webcam
    sarathi probe --source http://192.168.4.1:81/stream   # ESP32-CAM
    sarathi probe --source rtsp://192.168.1.60:8554/cam   # Pi / IP camera
    sarathi probe --source clip.mp4 --seconds 5 --save-frame first.png

`probe` exists to answer one question before anything else is debugged: is this
camera actually delivering usable frames, and at what rate and jitter? Almost
every "the app is laggy" report on a wireless camera turns out to be the
capture link, not the model, and this separates the two in ten seconds.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from .config import load_config
from .sources import LatestFrame, SourceError, create_source
from .util.log import configure, get_logger

log = get_logger(__name__)


def _fmt(values: list[float], unit: str = "ms") -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return f"min {ordered[0]:.1f} / p50 {p50:.1f} / p95 {p95:.1f} / max {ordered[-1]:.1f} {unit}"


def cmd_probe(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.set)
    spec = args.source if args.source is not None else cfg.get("source")

    try:
        source = create_source(spec)
    except SourceError as exc:
        log.error("%s", exc)
        return 2

    print(f"opening {spec!r} ...", flush=True)
    started = time.monotonic()
    try:
        cam = LatestFrame(source, reconnect=False).start()
    except SourceError as exc:
        log.error("could not open source: %s", exc)
        return 2
    open_ms = (time.monotonic() - started) * 1000.0

    intervals: list[float] = []
    sizes: set[tuple[int, int]] = set()
    frames = 0
    last_ts: float | None = None
    first_saved = False
    t_start = time.monotonic()
    deadline = t_start + args.seconds

    try:
        while time.monotonic() < deadline:
            frame = cam.get(timeout=2.0)
            if frame is None:
                if cam.ended:
                    break
                log.warning("no frame for 2s")
                continue
            frames += 1
            sizes.add((frame.width, frame.height))
            if last_ts is not None:
                intervals.append((frame.ts_capture - last_ts) * 1000.0)
            last_ts = frame.ts_capture

            if args.save_frame and not first_saved:
                import cv2

                out = Path(args.save_frame).expanduser()
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), frame.image)
                print(f"saved first frame -> {out}")
                first_saved = True
    except KeyboardInterrupt:
        print()
    finally:
        cam.stop()

    elapsed = max(1e-6, time.monotonic() - t_start)

    print()
    print(f"  source          {source.kind}  ({source.source_id})")
    print(f"  open latency    {open_ms:.0f} ms")
    print(f"  resolution      {', '.join(f'{w}x{h}' for w, h in sorted(sizes)) or 'none'}")
    print(f"  frames          {frames} captured, {cam.frames_dropped} dropped")
    print(f"  effective fps   {frames / elapsed:.1f}")
    print(f"  frame interval  {_fmt(intervals)}")
    if intervals and len(intervals) > 2:
        jitter = statistics.pstdev(intervals)
        print(f"  jitter (sd)     {jitter:.1f} ms")
        # A wireless link that stalls will show a fine median and a terrible
        # tail. That tail is what the user actually feels.
        if jitter > 0.5 * statistics.median(intervals):
            print("  ! high jitter - the transport is stalling, not the camera sensor")
    if cam.error is not None:
        print(f"  error           {cam.error}")
        return 1
    if frames == 0:
        print("  ! no frames received")
        return 1
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Measure how much inference the scheduler actually avoids.

    Every claim about battery life in this project reduces to one number: what
    fraction of frames never reach the model. This measures it on a real feed
    rather than asserting it.
    """
    from .runtime import MacThermalReader, NullThermalReader, Scheduler, SchedulerConfig

    cfg = load_config(args.config, args.set)
    spec = args.source if args.source is not None else cfg.get("source")

    scheduler = Scheduler(
        SchedulerConfig(
            max_inference_hz=args.max_hz,
            idle_inference_hz=args.idle_hz,
            motion_threshold=args.threshold,
            settle_s=args.settle,
            keepalive_hz=args.keepalive_hz,
        ),
        MacThermalReader() if args.thermal else NullThermalReader(),
    )

    try:
        cam = LatestFrame(create_source(spec), reconnect=False).start()
    except SourceError as exc:
        log.error("%s", exc)
        return 2

    print(f"sampling {spec!r} for {args.seconds:.0f}s ...", flush=True)
    deadline = time.monotonic() + args.seconds
    simulated_ms = 0.0
    try:
        while time.monotonic() < deadline:
            frame = cam.get(timeout=2.0)
            if frame is None:
                if cam.ended:
                    break
                continue
            now = time.monotonic()
            decision = scheduler.decide(frame, now)
            if decision.run:
                # No model loaded yet, so charge a representative cost. Replaced
                # by real measurements once a detector is benchmarked.
                scheduler.note_inference(args.inference_ms)
                simulated_ms += args.inference_ms
    except KeyboardInterrupt:
        print()
    finally:
        cam.stop()

    elapsed = args.seconds
    stats = scheduler.stats
    print()
    print(stats.summary())
    if stats.considered:
        naive_ms = stats.considered * args.inference_ms
        print()
        print(f"  duty cycle       {100 * simulated_ms / (elapsed * 1000):.1f}% of wall clock")
        print(f"  inference work   {simulated_ms / 1000:.1f}s of {elapsed:.0f}s sampled")
        print(f"  without gating   {naive_ms / 1000:.1f}s  ({naive_ms / max(1, simulated_ms):.1f}x more)")
        print(f"  final activity   {scheduler.activity.value}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """The real pipeline: camera in, guidance out."""
    from .guidance.speech import EarconPlayer, MacSpeaker, NullSpeaker, VoiceOutput
    from .runtime import Pipeline, PipelineConfig, SchedulerConfig

    cfg = load_config(args.config, args.set)
    spec = args.source if args.source is not None else cfg.get("source")

    speaker = NullSpeaker()
    if args.speak:
        try:
            speaker = MacSpeaker()
        except RuntimeError as exc:
            log.warning("%s - running silently", exc)

    pipeline = Pipeline(
        PipelineConfig(
            detector=args.detector,
            depth=args.depth,
            lang=args.lang,
            speak=args.speak,
            camera_height_m=args.camera_height,
            camera_pitch_deg=args.camera_pitch,
            scheduler=SchedulerConfig(max_inference_hz=args.max_hz),
        ),
        voice=VoiceOutput(speaker, EarconPlayer(enabled=args.speak)),
    )

    try:
        cam = LatestFrame(create_source(spec), reconnect=True).start()
    except SourceError as exc:
        log.error("%s", exc)
        return 2

    print(f"running {spec!r} for {args.seconds:.0f}s   detector={args.detector}   lang={args.lang}\n")
    deadline = time.monotonic() + args.seconds
    seen: dict[str, int] = {}
    said = 0
    ground_seen = False
    try:
        while time.monotonic() < deadline:
            frame = cam.get(timeout=2.0)
            if frame is None:
                if cam.ended:
                    break
                continue
            result = pipeline.process(frame)
            for det in result.detections:
                seen[det.label] = seen.get(det.label, 0) + 1
            if result.ground is not None and result.ground.anomaly and not ground_seen:
                ground_seen = True
                print(f"   ground: {result.ground.anomaly} @ "
                      f"{result.ground.anomaly_distance_m:.1f} m, "
                      f"free {result.ground.free_distance_m:.1f} m, "
                      f"fit {result.ground.fit_quality:.2f}")
            if result.utterance is not None:
                said += 1
                elapsed = args.seconds - (deadline - time.monotonic())
                mark = "  " if result.spoke or not args.speak else "x "
                print(f"{mark}{elapsed:5.1f}s  {result.utterance.urgency.name:<7} "
                      f"{result.utterance.text}")
    except KeyboardInterrupt:
        print()
    finally:
        cam.stop()
        pipeline.close()

    print()
    print(pipeline.scheduler.stats.summary())
    if seen:
        top = sorted(seen.items(), key=lambda kv: -kv[1])[:12]
        print("\ndetections by class:")
        for label, count in top:
            hazard = pipeline.bridge._hazards.get(label)
            tag = hazard.name.lower() if hazard else "unmapped"
            print(f"    {label:<16} {count:>5}   {tag}")
    print(f"\nutterances: {said}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Replay a clip through the pipeline and score the guidance."""
    from .bench import SpokenEvent, evaluate, load_truth
    from .runtime import Pipeline, PipelineConfig, SchedulerConfig
    from .sources.cv import FileSource

    truth, clip_name, duration = [], Path(args.source).name, 0.0
    if args.truth:
        clip_name, duration, truth = load_truth(args.truth)

    pipeline = Pipeline(PipelineConfig(
        detector=args.detector, depth=args.depth, lang=args.lang,
        camera_height_m=args.camera_height, camera_pitch_deg=args.camera_pitch,
        scheduler=SchedulerConfig(max_inference_hz=args.max_hz),
    ))

    # Media clock: run flat out, but keep every cooldown and budget behaving as
    # it would in real time.
    #
    # Read the source SYNCHRONOUSLY rather than through LatestFrame. The
    # drop-don't-queue policy is correct for live capture and wrong for replay:
    # offline the reader outruns the consumer, the single-slot buffer discards
    # most frames, and which ones survive depends on how fast the machine is.
    # A benchmark that scores differently on a faster laptop is not a
    # benchmark. Here every frame reaches the scheduler, and the scheduler
    # decides what to skip - which is the behaviour under test.
    source = FileSource("bench", args.source, realtime=False, media_clock=True)
    source.open()

    spoken: list[SpokenEvent] = []
    last_t = 0.0
    started = time.monotonic()
    try:
        while True:
            frame = source.grab()
            if frame is None:
                break
            now = frame.ts_capture
            last_t = max(last_t, now)
            result = pipeline.process(frame, now)
            if result.utterance is not None and result.chosen is not None:
                spoken.append(SpokenEvent(
                    t=now, label=result.chosen.track.label,
                    text=result.utterance.text, urgency=result.utterance.urgency,
                ))
    except KeyboardInterrupt:
        print()
    finally:
        source.close()
        pipeline.close()

    wall = time.monotonic() - started
    duration = duration or last_t
    outcome = evaluate(truth, spoken, clip=clip_name, duration_s=duration)

    print(f"replayed {last_t:.0f}s of video in {wall:.1f}s "
          f"({last_t / max(wall, 1e-6):.0f}x realtime)\n")
    if args.transcript:
        print("transcript:")
        for s_ in spoken:
            print(f"  {s_.t:6.1f}s  {s_.urgency.name:<7} {s_.text}")
        print()
    if truth:
        print(outcome.summary())
    else:
        print(f"no ground truth given - {len(spoken)} utterances, "
              f"{outcome.utterances_per_min:.1f}/min")
        print("annotate the clip and pass --truth to score recall and precision")
    print()
    print(pipeline.scheduler.stats.summary())
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    """Assemble one training set from every configured source."""
    import glob

    import yaml as _yaml

    from .datasets import (
        ReadStats,
        build_dataset,
        read_mendeley_stairs,
        read_voc,
        write_attribution,
    )
    from .taxonomy import Taxonomy

    repo = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root).expanduser()
    configs = [
        _yaml.safe_load(Path(f).read_text())
        for f in sorted(glob.glob(str(repo / "training" / "datasets" / "*.yaml")))
    ]
    taxonomy = Taxonomy.load()

    samples = []
    for config in configs:
        for name, spec in (config.get("sources") or {}).items():
            root = data_root / spec.get("root", "")
            if not root.exists():
                print(f"  {name:<24} SKIP - not present at {root}")
                continue
            label_map = spec.get("label_map") or {}
            stats = ReadStats()
            fmt = spec.get("format")
            if fmt == "voc":
                got = list(read_voc(root, label_map, source=name, stats=stats))
            elif fmt == "mendeley_stairs":
                got = list(read_mendeley_stairs(
                    root, {int(k): v for k, v in label_map.items()},
                    source=name, stats=stats))
            else:
                print(f"  {name:<24} SKIP - unknown format {fmt!r}")
                continue
            samples.extend(got)
            print(f"  {name}")
            print(stats.summary())

    if not samples:
        print("\nno samples found - check --data-root")
        return 1

    out = Path(args.out).expanduser()
    print(f"\nbuilding -> {out}")
    stats = build_dataset(samples, out, taxonomy,
                          val_fraction=args.val_fraction, copy_images=args.copy_images)
    write_attribution(configs, out / "ATTRIBUTION.md")
    print()
    print(stats.report(taxonomy))
    print(f"\nattribution written to {out / 'ATTRIBUTION.md'}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    print(load_config(args.config, args.set))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List every model the registry can see, and whether it is present."""
    from .models import ModelRegistry

    registry = ModelRegistry()
    manifests = registry.list(task=args.task)
    if not manifests:
        print(f"no manifests found in {registry.manifest_dir}")
        return 1

    for manifest in manifests:
        try:
            spec = manifest.file_for("prototype")
            path = spec.resolve(registry.weights_dir)
            state = "present" if path.exists() else "not downloaded"
        except Exception:  # noqa: BLE001 - reported as a state, not raised
            state = "no prototype build"
        mark = "  " if manifest.loadable else "x "
        print(f"{mark}{manifest.describe()}   [{state}]")

    if any(not m.loadable for m in manifests):
        print("\nx = excluded by licence policy; will not load")
    return 0


def cmd_licenses(args: argparse.Namespace) -> int:
    """Audit the licence of every model in one command."""
    from .models import ModelRegistry

    registry = ModelRegistry()
    rows = registry.licence_table()
    if not rows:
        print("no models found")
        return 1

    if args.attribution:
        print(registry.attribution_text())
        return 0

    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    header = ("model", "task", "licence", "distribution")
    widths = [max(w, len(h)) for w, h in zip(widths, header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return 0


def cmd_taxonomy(args: argparse.Namespace) -> int:
    """Report what the detector knows and, more usefully, what it cannot."""
    from .taxonomy import Taxonomy

    taxonomy = Taxonomy.load(args.file)

    if args.labels:
        out = Path(args.labels).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(taxonomy.names(args.lang)) + "\n")
        print(f"wrote {len(taxonomy)} labels ({args.lang}) -> {out}")
        return 0

    print(taxonomy.coverage_report())
    return 0


#: A scripted walk, used by `sarathi speak` to exercise the whole guidance
#: chain without a camera or a model. Each entry is
#: (start_s, end_s, label, distance_at_start, distance_at_end, bearing_deg).
#: It is deliberately a scenario with awkward overlaps rather than a tidy
#: sequence: a car closes while a chair is being announced, and a manhole
#: appears while the car is mid-sentence.
DEMO_WALK = [
    (0.0, 4.0, "chair", 2.2, 1.7, 12.0),  # in the corridor - announced
    (2.0, 9.0, "car", 6.0, 2.2, 6.0),  # closes while the chair is being said
    (5.5, 9.0, "open_manhole", 2.4, 1.4, 0.0),  # urgent, cuts in
    (9.5, 13.0, "person", 4.2, 3.0, -40.0),  # well off to the side - ignored
]


def cmd_speak(args: argparse.Namespace) -> int:
    """Run a scripted walk through tracking, saliency, phrasing and speech."""
    import time as _time

    from .guidance import MacSpeaker, NullSpeaker, Phraser, SaliencyEngine, VoiceOutput
    from .guidance.speech import EarconPlayer
    from .perception.tracking import Tracker
    from .taxonomy import Taxonomy
    from .types import Detection

    taxonomy = Taxonomy.load()
    phraser = Phraser(lang=args.lang)

    if args.silent:
        speaker = NullSpeaker()
    else:
        try:
            speaker = MacSpeaker()
        except RuntimeError as exc:
            log.error("%s", exc)
            return 2
        voice_name = speaker.voice_for(args.lang)
        print(f"voice: {voice_name or 'system default'}   language: {args.lang}\n")

    voice = VoiceOutput(speaker, EarconPlayer(enabled=not args.silent))
    tracker = Tracker(min_hits=2)
    engine = SaliencyEngine()

    dt = 0.3
    started = _time.monotonic()
    step = 0
    peak: dict[str, float] = {}
    announced: set[str] = set()
    while True:
        now = step * dt
        if now > 13.5:
            break

        frame: list[Detection] = []
        for idx, (t0, t1, label, d0, d1, bearing) in enumerate(DEMO_WALK):
            if not (t0 <= now <= t1):
                continue
            progress = (now - t0) / max(1e-6, t1 - t0)
            distance = d0 + (d1 - d0) * progress
            cls = taxonomy[label]
            x = idx * 200
            frame.append(
                Detection(
                    box=(x, 100, x + 60, 300), score=0.9, class_id=cls.id, label=label,
                    distance_m=distance, bearing_deg=bearing, hazard=cls.hazard,
                )
            )

        tracks = tracker.update(frame, now)
        for candidate in engine.rank(tracks):
            label = candidate.track.label
            peak[label] = max(peak.get(label, 0.0), candidate.score)
        chosen = engine.select(tracks, now)
        if chosen is not None:
            announced.add(chosen.track.label)
            utterance = phraser.utterance(chosen)
            said = voice.say(utterance, now=now)
            mark = "  " if said else "x "
            earcon = f"  [{utterance.earcon}]" if utterance.earcon else ""
            print(f"{mark}{now:5.1f}s  {utterance.urgency.name:<7} {utterance.text}{earcon}")

        step += 1
        if not args.silent:
            # Real time, so the utterance budget and the drop-while-busy rule
            # behave exactly as they would on a walk.
            slack = started + now + dt - _time.monotonic()
            if slack > 0:
                _time.sleep(slack)

    print(
        f"\nspoken {voice.spoken_count}  dropped {voice.dropped_count}  "
        f"interrupted {voice.interrupted_count}"
    )

    # Restraint is the feature, so make it visible. An object that was tracked
    # the whole way and deliberately never mentioned is the system working,
    # not the system failing to notice.
    ignored = sorted(
        ((label, score) for label, score in peak.items() if label not in announced),
        key=lambda kv: -kv[1],
    )
    if ignored:
        floor = engine.config.score_floor
        print(f"\nseen and deliberately not announced (floor {floor:.2f}):")
        for label, score in ignored:
            print(f"    {label:<14} peak score {score:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sarathi", description=__doc__.split("\n")[0])
    parser.add_argument("--config", "-c", help="YAML config file (or a name under configs/)")
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set runtime.max_inference_hz=4",
    )
    parser.add_argument("--log", default=None, help="log level (DEBUG/INFO/WARNING)")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="check a camera source and report capture health")
    probe.add_argument("--source", "-s", help="index, URL or file path (overrides config)")
    probe.add_argument("--seconds", type=float, default=10.0, help="how long to sample")
    probe.add_argument("--save-frame", help="write the first frame to this path")
    probe.set_defaults(func=cmd_probe)

    show = sub.add_parser("config", help="print the fully merged configuration")
    show.set_defaults(func=cmd_config)

    models = sub.add_parser("models", help="list known models and whether weights are present")
    models.add_argument("--task", choices=["detection", "depth", "ocr", "vlm"])
    models.set_defaults(func=cmd_models)

    run = sub.add_parser("run", help="the real pipeline: camera in, guidance out")
    run.add_argument("--source", "-s", help="index, URL or file path")
    run.add_argument("--seconds", type=float, default=20.0)
    run.add_argument("--detector", default="yolo11n-coco-320")
    run.add_argument("--depth", default=None,
                     help="depth model id, e.g. depth-anything-v2-small (Tier 2, ~2 Hz)")
    run.add_argument("--lang", default="en", choices=["en", "hi"])
    run.add_argument("--speak", action="store_true", help="actually talk")
    run.add_argument("--max-hz", type=float, default=8.0)
    run.add_argument("--camera-height", type=float, default=1.20, metavar="M")
    run.add_argument("--camera-pitch", type=float, default=0.0, metavar="DEG")
    run.set_defaults(func=cmd_run)

    ds = sub.add_parser("dataset", help="assemble the training set from all sources")
    ds.add_argument("--data-root", default="~/sarathi", help="where data/raw/ lives")
    ds.add_argument("--out", default="~/sarathi/data/sarathi77")
    ds.add_argument("--val-fraction", type=float, default=0.15)
    ds.add_argument("--copy-images", action="store_true",
                    help="copy rather than symlink (tens of GB - usually wrong)")
    ds.set_defaults(func=cmd_dataset)

    bench = sub.add_parser("bench", help="replay a clip and score the guidance")
    bench.add_argument("--source", "-s", required=True, help="video file")
    bench.add_argument("--truth", help="ground-truth YAML for the clip")
    bench.add_argument("--detector", default="yolo11n-coco-320")
    bench.add_argument("--depth", default=None)
    bench.add_argument("--lang", default="en", choices=["en", "hi"])
    bench.add_argument("--max-hz", type=float, default=8.0)
    bench.add_argument("--camera-height", type=float, default=1.40, metavar="M")
    bench.add_argument("--camera-pitch", type=float, default=20.0, metavar="DEG")
    bench.add_argument("--transcript", action="store_true", help="print every utterance")
    bench.set_defaults(func=cmd_bench)

    gate = sub.add_parser("gate", help="measure how much inference the scheduler avoids")
    gate.add_argument("--source", "-s", help="index, URL or file path")
    gate.add_argument("--seconds", type=float, default=20.0)
    gate.add_argument("--max-hz", type=float, default=8.0)
    gate.add_argument("--idle-hz", type=float, default=1.0)
    gate.add_argument("--keepalive-hz", type=float, default=0.2)
    gate.add_argument("--threshold", type=float, default=0.012)
    gate.add_argument("--settle", type=float, default=2.0)
    gate.add_argument("--inference-ms", type=float, default=25.0,
                      help="assumed cost of one detector pass until one is benchmarked")
    gate.add_argument("--thermal", action="store_true", help="read real thermal pressure")
    gate.set_defaults(func=cmd_gate)

    speak = sub.add_parser("speak", help="run a scripted walk through the guidance chain")
    speak.add_argument("--lang", default="en", choices=["en", "hi"])
    speak.add_argument("--silent", action="store_true", help="print only, make no sound")
    speak.set_defaults(func=cmd_speak)

    tax = sub.add_parser("taxonomy", help="report class coverage and blind spots")
    tax.add_argument("--file", help="taxonomy YAML (defaults to training/taxonomy/)")
    tax.add_argument("--labels", metavar="PATH", help="write a label file instead of a report")
    tax.add_argument("--lang", default="en", choices=["en", "hi"])
    tax.set_defaults(func=cmd_taxonomy)

    lic = sub.add_parser("licenses", help="audit the licence of every model")
    lic.add_argument(
        "--attribution", action="store_true", help="emit the generated attribution notice"
    )
    lic.set_defaults(func=cmd_licenses)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure(args.log)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
