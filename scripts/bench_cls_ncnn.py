#!/usr/bin/env python3
"""Benchmark + sanity-check the NCNN-exported disease classifier on the Pi.

Pi-side runner. Loads the NCNN cls bundle written by
`scripts/export_cls_ncnn.py`, runs warmup + timed iters, and
optionally measures top-1 accuracy on the val crop set written by
`scripts/build_disease_crops.py`.

Usage on the Pi:
    python scripts/bench_cls_ncnn.py \\
        --model models/disease/yolo26n_cls_ncnn_model \\
        --crops data/disease_crops/val \\
        --warmup 10 --iters 200

Output:
    reports/cls_ncnn_bench.md  — pasteable summary
    stdout JSON               — machine-readable summary
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark NCNN disease classifier on Pi 5.",
    )
    p.add_argument(
        "--model", type=Path,
        default=REPO_ROOT / "models/disease/yolo26n_cls_ncnn_model",
        help="Path to the NCNN model directory (contains model.ncnn.param/bin).",
    )
    p.add_argument(
        "--crops", type=Path,
        default=REPO_ROOT / "data/disease_crops/val",
        help="Directory of class-subdir crops. Bench draws inputs from here "
             "and computes top-1 accuracy if labels are present.",
    )
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup iterations (not timed).",
    )
    p.add_argument(
        "--iters", type=int, default=200,
        help="Timed iterations.",
    )
    p.add_argument(
        "--report", type=Path,
        default=REPO_ROOT / "reports/cls_ncnn_bench.md",
    )
    return p.parse_args()


def shell(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        return out.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unavailable"


def gather_pi_env() -> dict:
    return {
        "temp":      shell(["vcgencmd", "measure_temp"]),
        "throttled": shell(["vcgencmd", "get_throttled"]),
        "governor":  shell([
            "cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        ]),
        "uname":     shell(["uname", "-a"]),
    }


def collect_crops(crops_root: Path) -> list[tuple[Path, str]]:
    """Walk class subdirs and return [(crop_path, true_label), ...]."""
    samples: list[tuple[Path, str]] = []
    if not crops_root.is_dir():
        return samples
    for cls_dir in sorted(p for p in crops_root.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.glob("*.jpg")):
            samples.append((img, cls_dir.name))
    return samples


def main() -> int:
    args = parse_args()

    if not args.model.is_dir():
        print(f"NCNN model dir not found: {args.model}", file=sys.stderr)
        return 1

    samples = collect_crops(args.crops)
    if not samples:
        print(f"no crops under {args.crops}", file=sys.stderr)
        return 1

    env_pre = gather_pi_env()

    from ultralytics import YOLO

    print(f"[load] {args.model}", flush=True)
    t0 = time.perf_counter()
    model = YOLO(str(args.model), task="classify")
    load_ms = (time.perf_counter() - t0) * 1000
    pred_names = list(model.names.values())
    print(f"[load] {load_ms:.0f} ms | nc={len(pred_names)} | "
          f"names={pred_names}", flush=True)

    print(f"[warmup] {args.warmup} iters", flush=True)
    sample = str(samples[0][0])
    for _ in range(args.warmup):
        model(sample, imgsz=args.imgsz, verbose=False)

    print(f"[bench] {args.iters} iters over {len(samples)} crops "
          f"@ imgsz={args.imgsz}", flush=True)
    timings_ms: list[float] = []
    correct = 0
    confusion: Counter = Counter()
    for i in range(args.iters):
        path, true_label = samples[i % len(samples)]
        t0 = time.perf_counter()
        res = model(str(path), imgsz=args.imgsz, verbose=False)
        timings_ms.append((time.perf_counter() - t0) * 1000)
        if res and res[0].probs is not None:
            top1_idx = int(res[0].probs.top1)
            pred_label = pred_names[top1_idx]
            confusion[(true_label, pred_label)] += 1
            if pred_label == true_label:
                correct += 1

    env_post = gather_pi_env()

    timings_sorted = sorted(timings_ms)
    nt = len(timings_sorted)
    p50 = timings_sorted[nt // 2]
    p95 = timings_sorted[int(nt * 0.95) - 1]
    mean = statistics.fmean(timings_ms)
    stdev = statistics.pstdev(timings_ms)
    accuracy = correct / args.iters if args.iters else 0.0

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cls NCNN bench — yolo26n_cls",
        "",
        f"- model: `{args.model.relative_to(REPO_ROOT) if str(args.model).startswith(str(REPO_ROOT)) else args.model}`",
        f"- crops dir: `{args.crops}` ({len(samples)} crops across "
        f"{len({s[1] for s in samples})} classes)",
        f"- imgsz: {args.imgsz}",
        f"- warmup: {args.warmup} | iters: {args.iters}",
        "",
        "## Timing (ms / image, NCNN FP32)",
        "",
        f"- mean:   {mean:7.2f}",
        f"- p50:    {p50:7.2f}",
        f"- p95:    {p95:7.2f}",
        f"- max:    {max(timings_ms):7.2f}",
        f"- min:    {min(timings_ms):7.2f}",
        f"- stdev:  {stdev:7.2f}",
        "",
        "## Accuracy",
        "",
        f"- top-1 over {args.iters} bench iters: {accuracy:.3f} ({correct}/{args.iters})",
        "",
        "  Note: this is sanity accuracy on the val split that the model trained against — ",
        "  not a held-out test number. For the real test number, run `yolo val` against ",
        "  data/disease_crops/test on Mac/Colab post-training.",
        "",
        "## Confusion (top entries)",
        "",
    ]
    for (t, p), c in sorted(confusion.items(), key=lambda kv: -kv[1])[:20]:
        flag = "" if t == p else "  ✗"
        lines.append(f"- true=`{t}` -> pred=`{p}`: {c}{flag}")
    lines += [
        "",
        "## Pi env",
        "",
        f"- before: temp `{env_pre['temp']}` | throttled `{env_pre['throttled']}` "
        f"| governor `{env_pre['governor']}`",
        f"- after:  temp `{env_post['temp']}` | throttled `{env_post['throttled']}` "
        f"| governor `{env_post['governor']}`",
        f"- uname:  `{env_pre['uname']}`",
        "",
    ]
    args.report.write_text("\n".join(lines))

    summary = {
        "model": str(args.model),
        "imgsz": args.imgsz,
        "iters": args.iters,
        "crops_total": len(samples),
        "ms": {"mean": mean, "p50": p50, "p95": p95,
               "max": max(timings_ms), "min": min(timings_ms),
               "stdev": stdev},
        "accuracy_top1": accuracy,
        "env_pre": env_pre,
        "env_post": env_post,
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] report: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
