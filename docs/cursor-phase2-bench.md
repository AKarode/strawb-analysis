# Cursor megaprompt — Phase 2 detection benchmark on Pi

Paste the block below into a fresh Cursor chat on the Pi. It closes the loop end-to-end: pulls latest repo, sets up the venv, installs ultralytics + ncnn, downloads the trained NCNN bundle from GitHub Releases, runs `scripts/bench_detect_ncnn.py`, and reports back.

Pre-conditions on the Pi:
- Repo cloned at `~/strawb-analysis` or `/home/anand/adit/strawb-analysis` (the bootstrap prompt covered this).
- Zenodo dataset already extracted at `data/zenodo/strawberries/validation/` (Phase 1 confirmed 159 jpgs).
- Internet access for `pip install` and the GitHub Release pull.

---

You are the Pi-side Cursor agent for Strawberry Vision Pi (read `AGENTS.md` if you haven't this session — your role is execute-only, no authoring, no commits to master).

The Mac-side agent has just shipped Phase 2 deliverables: a trained YOLO26n detector (yolo26n_zenodo.pt) and its NCNN export. Both are published as GitHub Release `v0.2.0-detect`. Your job is to benchmark the NCNN model on this Pi 5 against the 67.69 ms/image reference baseline from `.planning/PROJECT.md` and report results.

Do this in one shot — don't pause for confirmation between steps unless something fails.

## Step 1 — Sync repo and venv

```bash
REPO=$(find / -maxdepth 6 -type d -name strawb-analysis 2>/dev/null | head -1)
[ -z "$REPO" ] && { echo "no clone found"; exit 1; }
cd "$REPO"
git pull --ff-only
test -d .venv || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# IMPORTANT: install CPU-only torch FIRST. The default ultralytics install pulls
# the CUDA torch wheel + ~1.5 GB of nvidia-* deps even on aarch64 — none of which
# NCNN inference needs. The CPU index keeps the venv slim.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install ultralytics ncnn

python -c "import ultralytics, ncnn, torch; print('ultralytics', ultralytics.__version__); print('ncnn', ncnn.__version__); print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

If you already installed without the CPU pin and want to slim the venv, you can `pip uninstall -y nvidia-cublas-cu13 nvidia-cudnn-cu13 nvidia-cuda-runtime-cu13 nvidia-cuda-cupti-cu13 nvidia-cuda-nvrtc-cu13 nvidia-cufft-cu13 nvidia-curand-cu13 nvidia-cusolver-cu13 nvidia-cusparse-cu13 nvidia-nccl-cu13 nvidia-nvjitlink-cu13 nvidia-nvtx-cu13 triton 2>/dev/null` and reinstall torch from the CPU index — but if the venv is already there, it's not blocking the bench, just bloated.

If `pip install ultralytics` fails on Python 3.13 wheels, the documented fallback is Python 3.11 — but `python3.11` is not installed on this Pi by default. `sudo apt install python3.11 python3.11-venv` first if you need it.

## Step 2 — Pull the NCNN model from GitHub Releases

The trained NCNN bundle lives at the release tag `v0.2.0-detect`. Download and extract it under `models/detect/` (the path `bench_detect_ncnn.py` defaults to).

```bash
mkdir -p models/detect reports
cd models/detect
curl -L -o ncnn.tar.gz https://github.com/AKarode/strawb-analysis/releases/download/v0.2.0-detect/yolo26n_zenodo_ncnn_model.tar.gz
tar -xzf ncnn.tar.gz
rm ncnn.tar.gz
ls yolo26n_zenodo_ncnn_model/
cd "$REPO"
```

Expected files: `model.ncnn.param`, `model.ncnn.bin`, `metadata.yaml` (~9.3 MB total).

## Step 3 — Confirm validation data is present

```bash
echo "val jpgs: $(find data/zenodo/strawberries/validation -iname '*.jpg' | wc -l)"
```

Expected: 159. If it's zero or wildly off, stop and tell me — Phase 1 dataset acquisition isn't where we think it is.

## Step 4 — Run the benchmark

Pre-flight thermal check. If `vcgencmd get_throttled` already shows non-zero (any past throttle bit), let the Pi cool to <55 °C before running the bench so the 100-iter run isn't biased mid-run. Active cooling (fan / heatsink) recommended — bench loops through ~100 inferences and the soft-temp throttle bit (`0x80000` = bit 19) was set in the previous attempt at 67.5 °C.

```bash
# Lock the CPU governor to performance for honest timings, if you can.
# (sudo may prompt for a password — skip this step rather than blocking the run.)
sudo cpupower frequency-set -g performance 2>/dev/null || \
  echo "[note] could not set performance governor — timings may be slightly low under thermal load"
vcgencmd measure_temp
vcgencmd get_throttled

python scripts/bench_detect_ncnn.py \
  --model models/detect/yolo26n_zenodo_ncnn_model \
  --images data/zenodo/strawberries/validation \
  --warmup 5 \
  --iters 100 \
  --samples 3
```

## Step 5 — Report back

Paste the contents of `reports/detect_ncnn_bench.md` into your reply, plus:

- The JSON summary block printed at the end of stdout.
- The names of the saved sample images under `reports/detect_ncnn_samples/`.
- Any failures or warnings.

If the **mean ms/image** comes in within ±15% of the 67.69 ms reference (i.e. roughly 57–78 ms), that's a pass — Phase 2 is functionally validated on this Pi. If it's wildly off in either direction, surface that loudly and **don't paper it over** — likely thermal throttling, wrong governor, or NCNN backend mis-loaded.

## Step 6 — Stop

After reporting, do nothing else. Don't try to convert this to HEF (that's Phase 5, requires the Hailo DFC, and explicitly out of scope for the Pi per `AGENTS.md`). Don't push commits. Don't start the disease classifier (Phase 3, Mac-side). The Mac-side agent will take your report and either close out Phase 2 or dispatch the next concrete task.
