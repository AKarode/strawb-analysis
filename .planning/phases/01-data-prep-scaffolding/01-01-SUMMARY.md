---
phase: 01-data-prep-scaffolding
plan: 01
status: partial
created: 2026-04-24
requirements: [DATA-05]
verification: env-blocked
---

# Plan 01-01 Summary — Skeleton, Weight Migration, and Dataset Licenses

## Status: partial (verification blocked by venv import deadlock)

The deterministic scaffolding and authoring work of this plan is complete and on disk. The smoke-test verification step in the plan's `<verify><automated>` block could not be executed in this session because `.venv/bin/python` hangs on native-extension imports (`numpy`, `torch`). See **Environment Blocker** below.

All non-Python verification (`test -f`, `git check-ignore`, `grep`) passes.

## Objective (restated)

Create the minimal repo skeleton (`src/`, `models/{detect,disease}/`, `reports/`), migrate the loose pretrained YOLO26 weights and NCNN exports out of the repo root into `models/`, update `scripts/smoke_test_yolo26.py` to load from the new paths, and author one `LICENSE.md` per dataset folder so downstream phases (and the commercial-deployment audit) can see per-source license posture at a glance.

## Skeleton layout on disk

```
src/
  __init__.py
models/
  detect/
    yolo26n.pt
    yolo26n_ncnn_model/
      metadata.yaml
      model.ncnn.bin
      model.ncnn.param
      model_ncnn.py
  disease/
    yolo26n-cls.pt
    yolo26n-cls_ncnn_model/
      metadata.yaml
      model.ncnn.bin
      model.ncnn.param
      model_ncnn.py
reports/
  .gitkeep
```

Matches CLAUDE.md "Repo Layout" section.

## Migration audit

| src (repo root) | dst (under models/) | method |
|---|---|---|
| `yolo26n.pt` | `models/detect/yolo26n.pt` | `mv` (file was gitignored, no `git mv` needed) |
| `yolo26n-cls.pt` | `models/disease/yolo26n-cls.pt` | `mv` |
| `yolo26n_ncnn_model/` | `models/detect/yolo26n_ncnn_model/` | `mv` (full dir) |
| `yolo26n-cls_ncnn_model/` | `models/disease/yolo26n-cls_ncnn_model/` | `mv` (full dir) |

Post-move checks (all pass):

- `test ! -e yolo26n.pt` ✓
- `test ! -e yolo26n-cls.pt` ✓
- `test ! -e yolo26n_ncnn_model` ✓
- `test ! -e yolo26n-cls_ncnn_model` ✓
- `test -f models/detect/yolo26n.pt` ✓
- `test -f models/disease/yolo26n-cls.pt` ✓
- `test -f models/detect/yolo26n_ncnn_model/model.ncnn.param` ✓
- `test -f models/disease/yolo26n-cls_ncnn_model/model.ncnn.param` ✓

## `scripts/smoke_test_yolo26.py` updates

Line 50: `det = YOLO("yolo26n.pt")` → `det = YOLO("models/detect/yolo26n.pt")`
Line 64: `cls = YOLO("yolo26n-cls.pt")` → `cls = YOLO("models/disease/yolo26n-cls.pt")`
Line 107: stale `/Users/adit/Desktop/Pyranthus/products/multi-spectral/data/zenodo/...` → `Path(__file__).resolve().parent.parent / "data/zenodo/strawberries/training"` (repo-relative, no legacy multi-spectral reference).

`grep -n 'multi-spectral' scripts/smoke_test_yolo26.py` returns no matches ✓.

## Dataset LICENSE.md files

Seven tracked `LICENSE.md` files authored verbatim from the plan's Task 2 bodies:

| Path | License | SPDX | Commercial use |
|---|---|---|---|
| `data/zenodo/LICENSE.md` | CC BY 4.0 | CC-BY-4.0 | Permitted w/ attribution |
| `data/disease/LICENSE.md` | CC BY 4.0 | CC-BY-4.0 | Permitted w/ attribution |
| `data/roboflow/afzaal-bbox-v4/LICENSE.md` | CC BY 4.0 (inherited) | CC-BY-4.0 | Permitted w/ attribution |
| `data/roboflow/matt-lucky-ripeness/LICENSE.md` | CC BY 4.0 | CC-BY-4.0 | Permitted w/ attribution |
| `data/roboflow/research-proj-disease/LICENSE.md` | MIT | MIT | Permitted w/ MIT attribution |
| `data/strawdi/LICENSE.md` | Non-commercial academic | (no SPDX) | **NOT permitted** (load-bearing) |
| `data/osf-ej5qv/LICENSE.md` | Mixed (CC-BY-4.0 disease; ripe/unripe unverified) | CC-BY-4.0 (partial) | Partial; ripe/unripe deferred |

StrawDI's non-commercial constraint is spelled verbatim in its LICENSE.md as specified:

- `grep 'NON-COMMERCIAL ACADEMIC USE ONLY' data/strawdi/LICENSE.md` ✓
- `grep 'DO NOT mix OSF disease classes' data/osf-ej5qv/LICENSE.md` ✓

## `.gitignore` edits

Two semantic changes plus one pragmatic cleanup:

1. **`data/` → `data/**` + `!data/**/` + `!data/**/LICENSE.md`** — changed from directory-ignore to glob-ignore so the `!data/**/LICENSE.md` negation can actually re-include the 7 LICENSE.md files. Git's documented behavior: you cannot un-ignore a file inside an ignored directory — the parent must stay traversable.
2. **`reports/` → `reports/**`** for the same reason, so `!reports/.gitkeep` works. The `*_ncnn_model/` pattern was also added (belt-and-suspenders in case a non-`models/` path produces an NCNN export).
3. **`.claude/settings.local.json` → `.claude/`** — broadened because Claude Code also drops `.claude/scheduled_tasks.lock`. Neither file is project state.

Validation (`git check-ignore -q` exit codes):

| Path | Expected | Actual |
|---|---|---|
| `data/strawdi/LICENSE.md` | 1 (tracked) | **1** ✓ |
| `data/zenodo/LICENSE.md` | 1 (tracked) | **1** ✓ |
| `data/roboflow/afzaal-bbox-v4/LICENSE.md` | 1 | **1** ✓ |
| `data/osf-ej5qv/LICENSE.md` | 1 | **1** ✓ |
| `reports/.gitkeep` | 1 | **1** ✓ |
| `data/strawdi/foo.jpg` | 0 (ignored) | **0** ✓ |
| `reports/foo.csv` | 0 (ignored) | **0** ✓ |
| `models/detect/yolo26n.pt` | 0 (ignored) | **0** ✓ |

## Environment Blocker — smoke test unverified

`.venv/bin/python -u scripts/smoke_test_yolo26.py` hangs indefinitely at native-extension import time (observed multiple runs, 0% CPU, >8 minutes elapsed, zero output past the banner).

Diagnostic findings:

- Pure stdlib imports (`sys`, `hashlib`, `json`, `pathlib`) succeed instantly in the venv.
- `import numpy` **hangs ≥60s** (`timeout 60 .venv/bin/python -u -c "import numpy"` → exit 124 with no output past `start`).
- `import torch` also hangs ≥60s (same timeout symptom).
- `/Users/adit/anaconda3/bin/python -u -c "import numpy"` (base anaconda, no venv) succeeds instantly — numpy 1.26.3.
- `KMP_DUPLICATE_LIB_OK=TRUE MKL_SERVICE_FORCE_INTEL=1 OMP_NUM_THREADS=1` env-var overrides did not help.
- No `com.apple.quarantine` xattrs on `.venv` (only benign `com.apple.provenance`).
- venv was created by `uv 0.8.4` on top of `/Users/adit/anaconda3/bin` Python 3.11.5 (`pyvenv.cfg` confirms).
- venv `numpy` is 2.4.4 ARM64 native dylib (`numpy/_core/_multiarray_umath.cpython-311-darwin.so`).

**Working hypothesis:** the uv-installed numpy 2.4.4 native `.so` fails to link its dependent runtime libraries when Python is launched by Claude Code's shell without conda activation context. Possibly an OpenMP/BLAS runtime conflict with anaconda's MKL, possibly a macOS `syspolicyd` first-launch verification that isn't completing in the Bash-tool subprocess.

**This blocker affects all Phase 1 Python work**, not just Plan 01-01. Plans 01-02 (`build_manifest.py`, `dedup_audit.py` — uses Pillow + PyYAML) and 01-03 (`build_disease_crops.py` — uses Pillow + numpy + cv2) cannot run their verification until the venv is fixed.

## Deviations from plan

None in substance. Deviations in scope of `.gitignore` edits only:

- Plan specified `!data/**/LICENSE.md` and `!reports/.gitkeep` negations; we additionally had to change `data/` and `reports/` from directory-ignore to glob-ignore patterns for the negations to take effect (git's documented parent-directory rule).
- `*_ncnn_model/` added (belt-and-suspenders).
- `.claude/settings.local.json` → `.claude/` (broader) — previous session had added the narrower rule; widened to catch `.claude/scheduled_tasks.lock`.

## Next steps for the user

1. **Fix the venv import hang.** Likely paths:
   - Rebuild the venv: `rm -rf .venv && uv venv .venv --python 3.11 && uv pip sync requirements.txt`
   - Or: activate conda before spawning (`conda activate base && source .venv/bin/activate` — then re-invoke Claude Code in that shell)
   - Or: swap to a non-anaconda Python 3.11 base
2. Re-run the smoke test to validate Task 1:
   ```bash
   .venv/bin/python -u scripts/smoke_test_yolo26.py
   ```
   Expected: `[OK]  ultralytics 8.4.40`, `[OK]  load ...` for both detect and cls, `[OK]  NCNN export` for both, `[OK]  PyTorch inference ...`, and the trailing `smoke test complete` banner.
3. Only after the smoke test passes, re-invoke `/gsd-execute-phase 1` to complete Plans 01-02 and 01-03.
