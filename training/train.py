#!/usr/bin/env python
"""Fine-tune the detector on the assembled dataset.

    python training/train.py --data ~/sarathi/data/sarathi77/data.yaml --device 1

Deliberate choices, each with a reason that is not "the tutorial said so":

**Start from COCO-pretrained weights, not scratch.** 15k images is nowhere near
enough to learn general visual features. Almost everything useful in the
backbone comes from COCO; fine-tuning teaches it our label set and our
viewpoint.

**320 px, matching the deployment resolution.** Training at 640 and deploying
at 320 is a common and quiet mistake: the model learns object scales it will
never see again, and small-object recall collapses on the phone.

**One GPU, and not GPU 0.** The lab machine is shared and GPU 0 usually has
somebody else's job on it. Taking a whole node because it is idle right now is
how you become the reason someone's week restarts.

**No mosaic in the final epochs.** Mosaic augmentation helps early and hurts
late - it trains on composites the deployed model never sees. Ultralytics
disables it for the last `close_mosaic` epochs, which is left on.

The class distribution is violently long-tailed - person at 35k against
stairs_down at 279 - so per-class metrics matter far more than the headline
mAP. A model can score well overall while never once detecting a staircase,
and the summary at the end prints per-class recall for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="data.yaml from `sarathi dataset`")
    parser.add_argument("--model", default="yolo11n.pt", help="starting checkpoint")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="1", help="GPU index - avoid 0 on a shared box")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="sarathi-yolo11n-320")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(Path(args.data).expanduser()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        # Deterministic enough to compare two runs; not bit-exact on GPU.
        seed=0,
        deterministic=False,
        # Pedestrian viewpoint: the camera does not turn upside down, and
        # vertical flips would teach it that ceilings are floors.
        flipud=0.0,
        fliplr=0.5,
        # Indian street lighting varies far more than COCO's. Push photometric
        # augmentation harder than the default and geometric augmentation less.
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=5.0,
        scale=0.5,
        close_mosaic=15,
        plots=True,
        val=True,
    )

    save_dir = Path(results.save_dir)
    print(f"\nweights: {save_dir / 'weights' / 'best.pt'}")

    # Per-class recall is the number that matters here. A long-tailed dataset
    # lets a model score respectably overall while never detecting a staircase.
    metrics = model.val(data=str(Path(args.data).expanduser()), imgsz=args.imgsz,
                        device=args.device, plots=False)
    names = metrics.names if hasattr(metrics, "names") else {}
    summary = {}
    try:
        for i, cls_idx in enumerate(metrics.ap_class_index):
            summary[names.get(int(cls_idx), str(cls_idx))] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
                "mAP50_95": round(float(metrics.box.ap[i]), 4),
            }
    except (AttributeError, IndexError, TypeError) as exc:
        print(f"could not extract per-class metrics: {exc}")

    out = save_dir / "per_class.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"per-class metrics: {out}\n")

    if summary:
        print(f"{'class':<22}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
        for name, m in sorted(summary.items(), key=lambda kv: kv[1]["recall"]):
            print(f"{name:<22}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                  f"{m['mAP50']:>9.3f}{m['mAP50_95']:>10.3f}")
        worst = [n for n, m in summary.items() if m["recall"] < 0.25]
        if worst:
            print(f"\n{len(worst)} classes with recall below 0.25:")
            print("    " + ", ".join(worst))
            print("These are effectively undetected. For a hazard class that is a")
            print("product failure, not a metrics footnote - report them as absent")
            print("rather than letting the headline mAP imply coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
