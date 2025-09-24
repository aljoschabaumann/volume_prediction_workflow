import numpy as np
import pandas as pd
import logging
from pymatgen.analysis.local_env import CrystalNN
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Specie

def assign_oxidation_states(structure):
    """
    Assign oxidation states to a pymatgen Structure.
    Uses BVAnalyzer first, falls back to composition guesses if needed.
    Adapted from:
    https://github.com/materialsproject/emmet/blob/4634eac272d321614db4ab5558832ceee19ad130/emmet-core/emmet/core/oxidation_states.py
    """
    structure = structure.copy()
    structure.remove_oxidation_states()
    try:
        bva = BVAnalyzer()
        valences = bva.get_valences(structure)
        structure.add_oxidation_state_by_site(valences)
        return structure
    except:
        try:
            first_guess = structure.composition.oxi_state_guesses(max_sites=-50)[0]
            valences = [first_guess[site.species_string] for site in structure]
            structure.add_oxidation_state_by_site(valences)
            return structure
        except:
            logging.warning("Failed to assign oxidation states.")
            return None


def featurize_structure(structure):
    """
    Featurize a single pymatgen Structure after assigning oxidation states.
    Returns a DataFrame with atom_features, bond_features, bond_indices.
    """
    # Assign oxidation states
    structure = assign_oxidation_states(structure)
    if structure is None:
        return None

    # Initialize CrystalNN and fingerprint
    cnn = CrystalNN(distance_cutoffs=None)
    crystal_finger = CrystalNNFingerprint.from_preset('ops')
    all_labels = crystal_finger.feature_labels()
    keep_indices = [i for i, label in enumerate(all_labels) if not any(q in label for q in ['q2','q4','q6'])]

    # ----------------------
    # Atom featurization
    # ----------------------
    atom_features = []
    for site_idx, site in enumerate(structure):
        feats = [
            site.specie.oxi_state,
            site.specie.Z
        ]
        crystal_feats = crystal_finger.featurize(structure, site_idx)
        filtered_feats = [crystal_feats[i] for i in keep_indices]
        feats.extend(filtered_feats)
        atom_features.append(np.array(feats))

    # ----------------------
    # Bond featurization
    # ----------------------
    bond_features = []
    bond_indices = []

    neigh_info = cnn.get_all_nn_info(structure)
    for site_idx, neighbors in enumerate(neigh_info):
        for neighbor in neighbors:
            neighbor_idx = neighbor['site_index']
            bond_length = structure.get_distance(site_idx, neighbor_idx, jimage=neighbor['image'])
            bond_feat = np.concatenate([
                atom_features[site_idx],
                atom_features[neighbor_idx],
                [bond_length]
            ])
            bond_features.append(bond_feat)
            bond_indices.append((site_idx, neighbor_idx))

    # Build DataFrame
    df = pd.DataFrame({
        'atom_features': [np.array(atom_features)],
        'bond_features': [np.array(bond_features)],
        'bond_indices': [np.array(bond_indices)]
    })

    return df
