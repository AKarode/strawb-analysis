#!/usr/bin/env python3
"""Benchmark the NCNN-exported detector on the Pi.

Pi-side runner. Loads the NCNN bundle written by
`scripts/export_detect_ncnn.py`, warms up, then times inference on the
Zenodo validation set. Emits a Markdown report under `reports/` so it can
be pasted back to the Mac side.

Reference baseline (.planning/PROJECT.md): Pi 5 NCNN FP32 YOLO26n
~67.69 ms/image. Numbers wildly off this — escalate, don't paper over.

Usage (run on the Pi inside .venv with ultralytics + ncnn installed):
    python scripts/bench_detect_ncnn.py \\
        --model models/detect/yolo26n_zenodo_ncnn_model \\
        --images data/zenodo/strawberries/validation \\
        --warmup 5 --iters 100

Outputs:
    reports/detect_ncnn_bench.md      — pasteable summary
    reports/detect_ncnn_samples/*.jpg — annotated detections (3 by default)
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark NCNN detector on Pi 5.",
    )
    p.add_argument(
        "--model", type=Path,
        default=REPO_ROOT / "models/detect/yolo26n_zenodo_ncnn_model",
        help="Path to the NCNN model directory (contains model.ncnn.param/bin).",
    )
    p.add_argument(
        "--images", type=Path,
        default=REPO_ROOT / "data/zenodo/strawberries/validation",
        help="Directory of .jpg images to time inference on.",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations (not timed). NCNN cold start is significant.",
    )
    p.add_argument(
        "--iters", type=int, default=100,
        help="Timed iterations. If fewer images exist, cycles through them.",
    )
    p.add_argument(
        "--samples", type=int, default=3,
        help="Number of annotated detections to save under reports/.",
    )
    p.add_argument(
        "--report", type=Path,
        default=REPO_ROOT / "reports/detect_ncnn_bench.md",
    )
    return p.parse_args()


def shell(cmd: list[str]) -> str:
    """Run a shell command, return stdout stripped, or 'unavailable' on error."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        return out.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unavailable"


def gather_pi_env() -> dict:
    """Collect Pi environment metadata per AGENTS.md reporting contract."""
    return {
        "temp_pre":  shell(["vcgencmd", "measure_temp"]),
        "throttled": shell(["vcgencmd", "get_throttled"]),
        "governor":  shell([
            "cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        ]),
        "uname":     shell(["uname", "-a"]),
    }


def main() -> int:
    args = parse_args()

    if not args.model.is_dir():
        print(f"NCNN model dir not found: {args.model}", file=sys.stderr)
        return 1
    if not args.images.is_dir():
        print(f"images dir not found: {args.images}", file=sys.stderr)
        return 1

    jpgs = sorted(args.images.rglob("*.jpg"))
    if not jpgs:
        print(f"no .jpg under {args.images}", file=sys.stderr)
        return 1

    env_pre = gather_pi_env()

    from ultralytics import YOLO

    print(f"[load] {args.model}", flush=True)
    t0 = time.perf_counter()
    model = YOLO(str(args.model), task="detect")
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[load] {load_ms:.0f} ms | nc={len(model.names)} | "
          f"names={list(model.names.values())}", flush=True)

    print(f"[warmup] {args.warmup} iters", flush=True)
    sample = str(jpgs[0])
    for _ in range(args.warmup):
        model(sample, imgsz=args.imgsz, verbose=False)

    print(f"[bench] {args.iters} iters over {len(jpgs)} images "
          f"@ imgsz={args.imgsz}", flush=True)
    timings_ms: list[float] = []
    detections_total = 0
    for i in range(args.iters):
        img = str(jpgs[i % len(jpgs)])
        t0 = time.perf_counter()
        res = model(img, imgsz=args.imgsz, verbose=False)
        timings_ms.append((time.perf_counter() - t0) * 1000)
        if res and res[0].boxes is not None:
            detections_total += len(res[0].boxes)

    env_post = gather_pi_env()

    timings_sorted = sorted(timings_ms)
    n = len(timings_sorted)
    p50 = timings_sorted[n // 2]
    p95 = timings_sorted[int(n * 0.95) - 1]
    mean = statistics.fmean(timings_ms)
    stdev = statistics.pstdev(timings_ms)

    samples_dir = REPO_ROOT / "reports/detect_ncnn_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, jpg in enumerate(jpgs[: args.samples]):
        res = model(str(jpg), imgsz=args.imgsz, verbose=False)
        annotated = res[0].plot()
        try:
            from PIL import Image
            out = samples_dir / f"sample_{i:02d}_{jpg.stem}.jpg"
            Image.fromarray(annotated[..., ::-1]).save(out, quality=85)
            saved.append(out.name)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] sample save failed for {jpg.name}: {exc}",
                  flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Detect NCNN bench — yolo26n_zenodo",
        "",
        f"- model: `{args.model.relative_to(REPO_ROOT)}`",
        f"- images dir: `{args.images.relative_to(REPO_ROOT)}` ({len(jpgs)} jpgs)",
        f"- imgsz: {args.imgsz}",
        f"- warmup: {args.warmup} | iters: {args.iters}",
        f"- detections across iters: {detections_total} "
        f"(mean {detections_total / args.iters:.1f} per image)",
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
        "Reference baseline: 67.69 ms/image (.planning/PROJECT.md).",
        "",
        "## Pi env",
        "",
        f"- before: temp `{env_pre['temp_pre']}` | throttled `{env_pre['throttled']}` "
        f"| governor `{env_pre['governor']}`",
        f"- after:  temp `{env_post['temp_pre']}` | throttled `{env_post['throttled']}` "
        f"| governor `{env_post['governor']}`",
        f"- uname:  `{env_pre['uname']}`",
        "",
        "## Annotated samples",
        "",
        *(f"- `reports/detect_ncnn_samples/{s}`" for s in saved),
        "",
    ]
    args.report.write_text("\n".join(lines))

    machine_summary = {
        "model": str(args.model.relative_to(REPO_ROOT)),
        "imgsz": args.imgsz,
        "iters": args.iters,
        "ms": {"mean": mean, "p50": p50, "p95": p95,
               "max": max(timings_ms), "min": min(timings_ms),
               "stdev": stdev},
        "env_pre": env_pre,
        "env_post": env_post,
        "detections_total": detections_total,
    }
    print(json.dumps(machine_summary, indent=2), flush=True)
    print(f"[done] report: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
