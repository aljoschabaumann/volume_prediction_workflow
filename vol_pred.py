import numpy as np
from pymatgen.core import Structure, Lattice

# Define constants
alpha = 0.001
beta = 0.003
max_iterations = 500
threshold = 1e-3

def predict_volume_change(initial_struc, bond_lengths, bond_indices, cnn):
    """
    Predicts the volume change of a structure based on predicted bond lengths.

    Parameters:
        initial_struc (Structure): Initial pymatgen Structure object.
        bond_lengths (list or array): Predicted bond lengths.
        bond_indices (list of tuples): List of bond index pairs corresponding to bond_lengths.
        cnn: A model or object with method `get_all_nn_info(structure)` returning nearest neighbor info.

    Returns:
        volume_change (float): Predicted percentage volume change (%).
    """
    structure = initial_struc.copy()
    old_positions_frac = structure.frac_coords.copy()
    old_lattice = np.array(structure.lattice.matrix, dtype=np.float64)
    neigh_info = cnn.get_all_nn_info(structure)

    predicted_bond_lengths = {tuple(bond_indices[n]): bond_lengths[n] for n in range(len(bond_lengths))}
    iteration = 0

    old_volume = initial_struc.volume
    while iteration < max_iterations:
        net_forces = np.zeros_like(old_positions_frac, dtype=np.float64)
        total_bond_length_error = 0.0
        direction_errors = np.zeros(3, dtype=np.float64)

        for i in range(len(structure)):
            current_atom_position_frac = structure[i].frac_coords

            # Calculate forces from nearest neighbors
            for neighbor in neigh_info[i]:
                j = neighbor['site_index']
                bond_length_key = (i, j) if (i, j) in predicted_bond_lengths else (j, i)
                if bond_length_key in predicted_bond_lengths:
                    target_length = predicted_bond_lengths[bond_length_key]

                    neighbor_position_frac = structure[j].frac_coords + neighbor['image']
                    current_length = structure.get_distance(i, j, jimage=neighbor['image'])
                    bond_length_error = current_length - target_length

                    direction_lattice = neighbor_position_frac - current_atom_position_frac
                    total_bond_length_error += abs(bond_length_error) / len(structure)
                    net_forces[i] += direction_lattice * bond_length_error
                    direction_errors -= bond_length_error * np.abs(direction_lattice)

        # Positions Update
        new_positions_frac = old_positions_frac + net_forces * alpha
        old_positions_frac = new_positions_frac

        # Lattice update
        new_lattice = np.zeros_like(old_lattice)
        scale_factors = 1.0 + direction_errors * beta
        for i in range(3):
            new_lattice[i] = old_lattice[i] * scale_factors[i]
        old_lattice = new_lattice

        # Update structure with new positions and lattice
        py_new_lattice = Lattice(old_lattice)
        structure = Structure(py_new_lattice, structure.species, new_positions_frac)

        new_volume = structure.volume

        if abs(old_volume - new_volume) < threshold:  # Check for convergence
            break
        old_volume = new_volume
        iteration += 1

    # Final volume change
    volume = abs(np.linalg.det(old_lattice))
    volume_change = ((volume / initial_struc.volume) - 1) * 100
    return volume_change
