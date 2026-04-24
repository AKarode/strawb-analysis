# License — OSF ej5qv (partial Kaggle re-host)

- **License:** Inherits CC BY 4.0 for disease classes (Kaggle Afzaal derivative); ripe/unripe classes have **UNVERIFIED PROVENANCE**.
- **SPDX:** CC-BY-4.0 (disease classes only)
- **Source:** https://osf.io/ej5qv/
- **Provenance:** Inspection (see research/2026-04-21-dataset-inventory-and-splits.md §4) shows ~92% basename overlap with Kaggle Afzaal disease dataset and 16/200 byte-identical MD5 collisions on sampled pairs. Disease classes are a classification-reorganized re-host of Kaggle (polygons stripped, classes as subdirs). Ripe/unripe classes have unknown origin — not in the OSF README, not referenced in any prior research source.
- **Commercial use:** Disease classes: permitted with Kaggle attribution. Ripe/unripe classes: **DO NOT USE** until provenance is verified.
- **Usage constraints (LOAD-BEARING):**
  - Directory name ` angular/` has a leading space (archive bug). Handle explicitly.
  - **DO NOT mix OSF disease classes into training alongside Kaggle Afzaal** — same source images, split leakage risk.
  - Treated as a non-training source in this project: use Kaggle polygon form as the canonical disease dataset.

2967 files, 9 class subdirs.
