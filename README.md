This repisotry stores the code used and puplished in the paper "High-Throughput-Screening Workflow for Predicting Volume Changes by Ion Intercalation in Battery Materials".

insert_cations_structure.py – Generates all valid crystal structures formed by inserting specified cations into a single pymatgen structure, considering symmetry, Wyckoff positions, and distance constraints.

featurize_structure.py – Generates atom-level and bond-level feature vectors for a single crystal structure, including crystal fingerprints and bond properties, for use in the XGBoost model.

xgboost_model.json - XGBoost model that predicts the bond lengths based on the bond-level features.

predict_volume_change.py – Predicts the percentage volume change of a crystal structure during ion intercalation using iterative bond-length-based updates on atomic positions and lattice vectors.
