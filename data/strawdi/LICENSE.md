# License — StrawDI_Db1 (Huelva)

- **License:** NON-COMMERCIAL ACADEMIC USE ONLY
- **SPDX:** (no standard SPDX identifier; custom academic license)
- **Source:** https://strawdi.github.io/
- **Citation:** Perez-Borrero, I.; Marin-Santos, D.; Gegundez-Arias, M.E.; Cortes-Ancos, E. (2020). A fast and accurate deep learning method for strawberry instance segmentation. Computers and Electronics in Agriculture.
- **Commercial use:** **NOT PERMITTED.** This dataset is restricted to non-commercial academic research.
- **Usage constraints (LOAD-BEARING):**
  - MUST NOT be redistributed as part of a commercial product.
  - MUST NOT be used to train any model weights that ship in the commercial deliverable.
  - MAY be used for ablation studies, research, and internal comparison only.
  - Any detector weights trained with StrawDI data must be clearly segregated from the production weights shipped to the client.

3100 instance-seg masks (train 2800 / val 100 / test 200), single class (strawberry), 1008×756 PNG masks.

**CRITICAL:** Before any commercial deployment, verify that no production model checkpoint has been trained on StrawDI data. If unsure, retrain without StrawDI.
