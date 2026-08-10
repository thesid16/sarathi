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
