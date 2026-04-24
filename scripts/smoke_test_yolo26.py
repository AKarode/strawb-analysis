#!/usr/bin/env python3
"""YOLO26 env smoke test for Strawberry Vision Pi.

Verifies:
  1. ultralytics 8.4.40 imports cleanly
  2. torch + MPS available (Apple Silicon)
  3. YOLO26n detect weights download + load
  4. YOLO26n-cls weights download + load
  5. NCNN export path works for detect @ 640
  6. NCNN export path works for cls @ 224  (OPEN RESEARCH ITEM from 4/21 log)
  7. PyTorch inference runs on a real Zenodo sample

Output is captured and pasted into .planning/research/ as verification evidence.
"""
import os
import sys
import time
import traceback
from pathlib import Path

print("=" * 64)
print("YOLO26 smoke test - Strawberry Vision Pi")
print("=" * 64)

# --- ultralytics import ---
try:
    import ultralytics
    from ultralytics import YOLO
    print(f"[OK]  ultralytics {ultralytics.__version__}")
except Exception as e:
    print(f"[FATAL] ultralytics import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- torch + MPS ---
try:
    import torch
    print(f"[OK]  torch {torch.__version__}")
    print(f"      MPS available: {torch.backends.mps.is_available()}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"[FAIL] torch import: {e}")

# --- Detect load ---
print("\n" + "-" * 64)
print("YOLO26n detect: load")
print("-" * 64)
try:
    t0 = time.time()
    det = YOLO("models/detect/yolo26n.pt")
    print(f"[OK]  load {time.time()-t0:.2f}s | task={det.task} | nc={len(det.names)}")
    print(f"      names: {list(det.names.values())[:10]}{'...' if len(det.names)>10 else ''}")
except Exception as e:
    print(f"[FAIL] detect load: {e}")
    traceback.print_exc()
    det = None

# --- Cls load ---
print("\n" + "-" * 64)
print("YOLO26n-cls: load")
print("-" * 64)
try:
    t0 = time.time()
    cls = YOLO("models/disease/yolo26n-cls.pt")
    print(f"[OK]  load {time.time()-t0:.2f}s | task={cls.task} | nc={len(cls.names)}")
except Exception as e:
    print(f"[FAIL] cls load: {e}")
    traceback.print_exc()
    cls = None

# --- NCNN export: detect ---
print("\n" + "-" * 64)
print("NCNN export: detect @ 640")
print("-" * 64)
if det is not None:
    try:
        t0 = time.time()
        det_ncnn_path = det.export(format="ncnn", imgsz=640)
        print(f"[OK]  export {time.time()-t0:.2f}s")
        print(f"      output: {det_ncnn_path}")
    except Exception as e:
        print(f"[FAIL] detect NCNN export: {e}")
        traceback.print_exc()
else:
    print("[SKIP] detect not loaded")

# --- NCNN export: cls (the open research item) ---
print("\n" + "-" * 64)
print("NCNN export: cls @ 224   [OPEN RESEARCH ITEM from 4/21 log]")
print("-" * 64)
if cls is not None:
    try:
        t0 = time.time()
        cls_ncnn_path = cls.export(format="ncnn", imgsz=224)
        print(f"[OK]  export {time.time()-t0:.2f}s")
        print(f"      output: {cls_ncnn_path}")
    except Exception as e:
        print(f"[FAIL] cls NCNN export: {e}")
        traceback.print_exc()
else:
    print("[SKIP] cls not loaded")

# --- Zenodo inference ---
print("\n" + "-" * 64)
print("PyTorch inference: Zenodo sample")
print("-" * 64)
data_dir = Path(__file__).resolve().parent.parent / "data/zenodo/strawberries/training"
if det is not None and data_dir.exists():
    jpgs = sorted(data_dir.glob("*.jpg"))
    if jpgs:
        sample = str(jpgs[0])
        print(f"sample: {Path(sample).name}")
        try:
            _ = det(sample, verbose=False)  # warmup
            t0 = time.time()
            res = det(sample, verbose=False)
            ms = (time.time() - t0) * 1000
            n = len(res[0].boxes) if res and res[0].boxes is not None else 0
            print(f"[OK]  PyTorch inference: {ms:.1f} ms | detections: {n}")
        except Exception as e:
            print(f"[FAIL] inference: {e}")
            traceback.print_exc()
    else:
        print(f"[SKIP] no .jpg in {data_dir}")
else:
    missing = []
    if det is None: missing.append("det")
    if not data_dir.exists(): missing.append(f"dir {data_dir}")
    print(f"[SKIP] missing: {missing}")

print("\n" + "=" * 64)
print("smoke test complete")
print("=" * 64)
