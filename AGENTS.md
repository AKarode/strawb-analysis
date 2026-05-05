# AGENTS.md — Pi-side agent guidance

This repo is dual-machine. A Mac dev workstation **authors** code, scripts, and exported model artifacts. This Raspberry Pi **executes** inference and benchmarks. If you are an AI coding agent (Cursor, Codex, etc.) running on the Pi, this file is for you. If you are running on the Mac, read `CLAUDE.md` instead.

Read `.planning/PROJECT.md` for full project context. Read `.planning/ROADMAP.md` for phase sequence. Don't re-derive those — they're authoritative.

## Your role

**You execute. The Mac-side agent authors.**

DO:
- Pull the latest commit (`git pull`).
- Install runtimes (Python deps, Hailo runtime, NCNN) per instructions in this file.
- Run scripts in `scripts/` and modules in `src/` exactly as the user / Mac-side asks.
- Time inference runs, capture outputs, report results back.
- Surface Pi-environment-specific issues (missing system packages, permissions, kernel module not loaded, Hailo firmware mismatches).

DON'T:
- Rewrite training code, refactor `src/`, redesign the pipeline, or touch `.planning/`.
- Re-train models, edit dataset manifests, or modify model artifacts in `models/`.
- Make architectural decisions (backend choice, model selection, label schema).
- Push commits to `master` from the Pi. Results come back via paste-to-user, or via a scoped `pi/<phase>` branch if explicitly set up.

If something looks wrong with the Mac-authored code, **stop and report it** — don't fix it on the Pi side. The Mac-side agent owns code correctness.

## Hardware

- Raspberry Pi 5 (4-core Cortex-A76, 16 GB RAM)
- **AI HAT+ 2 with Hailo-10H (40 TOPS)** — confirmed by on-device inspection in April 2026.
- Access via Raspberry Pi Connect (no SSH).

The earlier `.planning/PROJECT.md` and `.planning/REQUIREMENTS.md` references to "Hailo-8 / original AI HAT+" are stale. `README.md` is the source of truth. Toolchain implication: Hailo-10H uses a newer Dataflow Compiler (DFC) and has a sparser pre-built model zoo than Hailo-8 — pre-compiled HEFs from older Hailo Model Zoo releases may not exist for this chip.

Sanity check after every Pi reboot or runtime upgrade:

```bash
hailortcli fw-control identify
# Must report: Device Architecture: HAILO10H
```

If it reports HAILO8 or fails, stop and surface to the user — something is misconfigured.

## One-time setup

### 1. Clone the repo (public, HTTPS — no auth needed)

```bash
# Use existing clone if one already exists on this Pi; otherwise clone fresh.
# Current canonical path on the project Pi is /home/anand/adit/strawb-analysis;
# any path under your home directory is fine for a new clone.
git clone https://github.com/AKarode/strawb-analysis.git ~/strawb-analysis
cd ~/strawb-analysis    # or cd /home/anand/adit/strawb-analysis if reusing
```

If the user later switches to write-back from the Pi (committing benchmark results), they'll set up a separate write-scoped credential. Read-only HTTPS is the default flow.

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note on Python 3.13: the Pi ships Python 3.13 by default. Phase 1 deps (`Pillow`, `PyYAML`) have aarch64 / 3.13 wheels and install cleanly. Heavier deps land in later phases (`ultralytics`, `torch`, `ncnn`, `opencv-python`, `pandas`, `hailo-platform`); if any of those have no 3.13 / aarch64 wheel when we get there, fall back to Python 3.11 in a separate venv rather than building from source on the Pi.

### 3. Hailo runtime (only after hardware is confirmed)

Install the Hailo runtime matching the chip identified above. Follow the current vendor instructions for Pi 5; do not guess at package names from training data, the install path changes between versions. Verify with:

```bash
hailortcli fw-control identify   # must succeed, must report the expected chip
```

## Daily flow

```bash
cd ~/strawb-analysis
git pull
[[ requirements.txt -nt .venv/.last-install ]] && pip install -r requirements.txt && touch .venv/.last-install
# then run whatever the user / Mac-side asks
```

## Commands you'll typically be asked to run

These entry points may not exist yet — they are deliverables of later phases. Don't fabricate them.

```bash
# CPU smoke (Phase 4, after src/run_inference.py lands)
python -m src.run_inference \
  --backend cpu \
  --images <path-to-image-folder> \
  --model models/detect/yolo26n_ncnn_model \
  --classifier models/disease/yolo26n_cls_ncnn_model \
  --out reports/cpu_smoke.csv

# Hailo (Phase 5, after HEFs land in models/)
python -m src.run_inference \
  --backend hailo \
  --images <path-to-image-folder> \
  --hef-detect models/detect/yolo26n.hef \
  --hef-classifier models/disease/yolo26n_cls.hef \
  --out reports/hailo_smoke.csv

# Evaluator (Phase 4)
python -m src.evaluator \
  --predictions reports/cpu_smoke.csv \
  --ground-truth data/zenodo/labels/ \
  --out reports/eval_cpu.json
```

## Reporting results

When you run a benchmark, always report:

- Image count and source folder.
- Model identifier (file path or weight checksum).
- Backend (`cpu` | `hailo`) and runtime version (`hailortcli --version`, NCNN version).
- ms/image: mean, median, p95, max.
- Pi temperature before/after: `vcgencmd measure_temp`.
- Any thermal throttling: `vcgencmd get_throttled` (non-zero = throttled).
- CPU governor: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` (should be `performance` or `ondemand` for honest timings).

Reference baseline (per `.planning/PROJECT.md`): Pi 5 NCNN FP32 YOLO26n ~67.69 ms/image. Numbers wildly off this — escalate, don't paper over.

## Out of scope on the Pi

- **No training.** Training runs on the Mac dev workstation or Colab. Don't `yolo train` on the Pi.
- **No dataset re-downloads** without confirming with the user. Datasets are large; prefer rsync from Mac over re-pulling from Zenodo/Kaggle.
- **No model conversion** (export to NCNN / HEF) on the Pi unless explicitly asked. Conversion is a Mac-side task.

## Where to find more

- `CLAUDE.md` — Mac-side agent context, full project overview.
- `docs/cursor-bootstrap-prompt.md` — paste-ready bootstrap prompt for spinning up a fresh Cursor session on the Pi.
- `.planning/PROJECT.md` — core value, constraints, key decisions.
- `.planning/REQUIREMENTS.md` — 25 v1 requirements with REQ-IDs.
- `.planning/ROADMAP.md` — 5-phase plan and dependencies.
- `.planning/STATE.md` — current phase / progress.
- `README.md` — public-facing overview and source of truth for hardware.
