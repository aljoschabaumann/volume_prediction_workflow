import numpy as np
import logging
from pymatgen.analysis.local_env import CrystalNN
from matminer.featurizers.structure import CrystalNNFingerprint

def featurize_structure(structure, final_structure=None, mapping=None, final_mapping=None, cnn=None, keep_q_features=False):
    """
    Featurizes a single pymatgen Structure, producing atom and bond features.

    Parameters:
        structure (Structure): Initial pymatgen Structure.
        final_structure (Structure, optional): Final structure (for bond lengths). If None, bond lengths will be skipped.
        mapping (list, optional): Mapping of atom indices from initial to internal ordering.
        final_mapping (list, optional): Mapping from final structure indices to initial structure indices.
        cnn (CrystalNN, optional): CrystalNN object to compute neighbors. If None, a default is used.
        keep_q_features (bool): Whether to keep all crystal fingerprint features. If False, q2/q4/q6 features are removed.

    Returns:
        atom_features (np.ndarray): Array of shape (n_atoms, n_atom_features).
        bond_features (np.ndarray): Array of shape (n_bonds, n_bond_features).
        bond_lengths (np.ndarray): Array of bond lengths corresponding to bond_features.
        bond_indices (list of tuples): Atom index pairs for each bond.
    """
    if cnn is None:
        cnn = CrystalNN(distance_cutoffs=None)

    crystal_finger = CrystalNNFingerprint.from_preset('ops')
    all_labels = crystal_finger.feature_labels()
    if keep_q_features:
        keep_indices = list(range(len(all_labels)))
    else:
        keep_indices = [i for i, label in enumerate(all_labels) if not any(q in label for q in ['q2','q4','q6'])]

    # --- Atom featurization ---
    atom_features = []
    n_atoms = len(structure)
    if mapping is None:
        mapping = list(range(n_atoms))

    for up_site in range(n_atoms):
        site = structure[up_site]
        feats = [
            site.specie.oxi_state,
            site.specie.Z
        ]
        crystal_feats = crystal_finger.featurize(structure, up_site)
        filtered_feats = [crystal_feats[i] for i in keep_indices]
        feats.extend(filtered_feats)
        atom_features.append(np.array(feats))
    atom_features = np.array(atom_features)

    # If final_structure is not provided, return only atom features
    if final_structure is None:
        return atom_features, None, None, None

    if final_mapping is None:
        final_mapping = mapping.copy()

    # --- Bond featurization ---
    neigh_info = cnn.get_all_nn_info(structure)

    bond_lengths = []
    bond_indices = []
    bond_features = []

    for up_site, ini_site in enumerate(mapping):
        fin_site = final_mapping.index(up_site)
        for neighbor in neigh_info[up_site]:
            up_neighbor = neighbor['site_index']
            fin_neigh = final_mapping.index(up_neighbor)

            bond_length_fin = final_structure.get_distance(fin_site, fin_neigh)
            bond_length_ini = structure.get_distance(up_site, up_neighbor, jimage=neighbor['image'])

            bond_feat = np.concatenate([
                atom_features[up_site],         # atom i features
                atom_features[up_neighbor],     # atom j features
                [bond_length_ini]               # bond property
            ])
            bond_features.append(bond_feat)
            bond_lengths.append(bond_length_fin)
            bond_indices.append((up_site, up_neighbor))

    return atom_features, np.array(bond_lengths), np.array(bond_features), bond_indices
