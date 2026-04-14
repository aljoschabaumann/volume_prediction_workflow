This repisotry stores the code used in the paper "High-Throughput-Screening Workflow for Predicting Volume Changes by Ion Intercalation in Battery Materials" (https://doi.org/10.1021/acsaem.5c03917). If you use any code of this reposetory, please consider citing the article: ACS Appl. Energy Mater. 2026, 9, 7, 3851–3860.

insert_cations_structure.py – Generates all valid crystal structures formed by inserting specified cations into a single pymatgen structure, considering symmetry, Wyckoff positions, and distance constraints.

featurize_structure.py – Generates atom-level and bond-level feature vectors for a single crystal structure, including crystal fingerprints and bond properties, for use in the XGBoost model.

xgboost_model.json - XGBoost model that predicts the bond lengths based on the bond-level features. This model was trained and validated on a dataset of 3d-transition-metal oxide and fluorides and should only be used for applications on this material class.

predict_volume_change.py – Predicts the percentage volume change of a crystal structure during ion intercalation using iterative bond-length-based updates on atomic positions and lattice vectors.
