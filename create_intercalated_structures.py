import re
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix
from pyxtal.symmetry import Group

# Oxidation states (used for electrons transferred)
oxidation_states = {'Li': 1, 'Na': 1, 'K': 1, 'Ca': 2, 'Mg': 2}

def insert_cations_single_structure(structure, cations):
    """
    Generate valid structures by inserting specified cations into a single pymatgen Structure.

    Parameters:
        structure (Structure): Pymatgen structure to insert cations into.
        cations (list of str): List of cations to insert (e.g., ['Li','Na']).

    Returns:
        results (list of dict): Each dict contains:
            - 'new_struc': pymatgen Structure with cation inserted
            - 'added_element': cation inserted
            - 'wyckoff_added': Wyckoff letter used
            - 'added_ions': number of ions inserted
            - 'electrons_transferred': total electrons from insertion
            - 'attempt': 0 for deterministic, >0 for random attempts
    """
    results = []

    # Symmetry analysis
    sga = SpacegroupAnalyzer(structure)
    sg_num = sga.get_space_group_number()
    dataset = sga.get_symmetry_dataset()
    wyckoffs = dataset.wyckoffs
    elements = [site.species_string for site in structure.sites]
    occupied_letters = set((el, w) for el, w in zip(elements, wyckoffs))

    group = Group(sg_num)
    all_wyckoffs = group.get_wp_list()
    vacant_wyckoffs = [w for w in all_wyckoffs if w not in {w for (_, w) in occupied_letters}]

    # Convert pymatgen to pyxtal
    old_struc = pyxtal()
    old_struc.from_seed(structure)
    old_species = old_struc.species
    old_numIons = old_struc.numIons
    old_lattice = old_struc.lattice
    old_sites = [(w, None) for w in vacant_wyckoffs]

    def parse_number(s):
        from fractions import Fraction
        s = s.strip()
        try:
            return float(s)
        except ValueError:
            return float(Fraction(s))

    def extract_coords_from_wyckoff_letter(group, letter):
        wp = group.get_wp_by_letter(letter)
        lines = str(wp).split('\n')[1:]
        coords = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.replace(' ', '').split(',')
            coords.append(np.array([parse_number(x) for x in parts]))
        return coords if coords else False

    def check_valid(struct):
        dmat = struct.distance_matrix
        np.fill_diagonal(dmat, np.inf)
        return np.all(dmat > 1.0)

    # Loop over cations and vacant Wyckoff positions
    for cat in cations:
        oxidation = oxidation_states.get(cat, 1)
        for w in vacant_wyckoffs:
            match = re.match(r"(\d+)([a-z]+)", w)
            if not match:
                continue
            multiplicity = int(match.group(1))
            species = old_species + [cat]
            numIons = np.append(old_numIons, multiplicity)
            coords = extract_coords_from_wyckoff_letter(group, w)

            # Deterministic insertion
            if coords:
                try:
                    new_sites = old_sites + [(w, c) for c in coords]
                    xtal = pyxtal()
                    xtal.from_random(
                        dim=3,
                        group=sg_num,
                        species=species,
                        numIons=numIons,
                        lattice=old_lattice,
                        sites=new_sites,
                        tm=Tol_matrix.from_single_value(1)
                    )
                    new_pm_struct = xtal.to_pymatgen()
                    if check_valid(new_pm_struct):
                        results.append({
                            "new_struc": new_pm_struct,
                            "added_element": cat,
                            "wyckoff_added": w,
                            "added_ions": multiplicity,
                            "electrons_transferred": oxidation * multiplicity,
                            "attempt": 0
                        })
                        continue
                except Exception:
                    pass

            # Random insertion attempts (up to 10)
            for attempt in range(10):
                try:
                    xtal = pyxtal()
                    xtal.from_random(
                        dim=3,
                        group=sg_num,
                        species=species,
                        numIons=numIons,
                        lattice=old_lattice,
                        sites=old_sites + [(w, None)],
                        tm=Tol_matrix.from_single_value(1)
                    )
                    new_pm_struct = xtal.to_pymatgen()
                    if check_valid(new_pm_struct):
                        results.append({
                            "new_struc": new_pm_struct,
                            "added_element": cat,
                            "wyckoff_added": w,
                            "added_ions": multiplicity,
                            "electrons_transferred": oxidation * multiplicity,
                            "attempt": attempt + 1
                        })
                        break
                except Exception:
                    continue

    return results
