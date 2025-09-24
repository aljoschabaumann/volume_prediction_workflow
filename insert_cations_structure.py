import numpy as np
import pandas as pd
import re
from fractions import Fraction
from pyxtal import pyxtal, Group, Tol_matrix
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def parse_number(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        try:
            return float(Fraction(s))
        except ValueError:
            raise ValueError(f"Cannot parse number: {s}")

def extract_coords_from_wyckoff_letter(group, letter):
    """
    Extract fractional coordinate arrays from Wyckoff letter.
    Returns list of np.array coords or False if parsing fails.
    """
    wp = group.get_wp_by_letter(letter)
    lines = str(wp).split('\n')[1:]  # skip header line
    coords = []
    try:
        for line in lines:
            if not line.strip():
                continue
            parts = line.replace(' ', '').split(',')
            coords.append(np.array([parse_number(x) for x in parts]))
        return coords if coords else False
    except ValueError:
        return False

def get_occupied_wyckoff_letters(structure):
    sga = SpacegroupAnalyzer(structure)
    dataset = sga.get_symmetry_dataset()
    wyckoffs = dataset.wyckoffs
    elements = [site.species_string for site in structure.sites]
    occupied = set((el, w) for el, w in zip(elements, wyckoffs))
    return occupied, sga.get_space_group_number()

def get_vacant_wyckoff_letters(sg_num, occupied_letters):
    group = Group(sg_num)
    all_wyckoffs = group.get_wp_list()  # list of letters e.g. ['a','b','c',...]
    occupied = {w for (_, w) in occupied_letters}
    return [w for w in all_wyckoffs if w not in occupied]

def is_structure_valid(pmg_structure, threshold=1.0):
    """
    Check if all interatomic distances in a pymatgen Structure are above the given threshold.
    """
    dmat = pmg_structure.distance_matrix
    np.fill_diagonal(dmat, np.inf)
    return np.all(dmat > threshold)

def extract_wyckoff_positions(crystal,work_in):
    new_wyckoff_data=[]
    el_list=[]
    # Iterate through each site in the crystal
    for site in crystal.atom_sites:
        el_list.append(site.specie)
        wyckoff_position = f"{site.wp.multiplicity}{site.wp.letter}"
        coord = site.coords[0].tolist()
        new_wyckoff_data.append(tuple([wyckoff_position,coord[0],coord[1],coord[2]]))
    # Initialize a dictionary to store lists of tuples for each element
    element_dict = {element: [] for element in el_list}
    # Loop through each tuple and add it to the corresponding element list
    for tup, element in zip(new_wyckoff_data, el_list):
        if element not in element_dict:
            element_dict[element] = []
        if tup is not None:
            element_dict[element].append(tup)
    indexes = np.unique(el_list, return_index=True)[1]
    new_el_list=[el_list[index] for index in sorted(indexes)]
    # Create the final list of lists maintaining the order of elements_list
    result = [element_dict[element] for element in new_el_list]   
    if not work_in:
        result.append(None)
    return result

def insert_cations(structure, cation="Li"):
    """
    Generate valid structures by inserting a specified cation into a single pymatgen Structure.
    
    Parameters:
        structure (Structure): Pymatgen structure to insert cation into.
        cation (str): Cation to insert (default: "Li").
    
    Returns:
        DataFrame: Each row contains a new structure and insertion info.
    """
    # Convert to pyxtal
    old_struc = pyxtal()
    old_struc.from_seed(structure)

    occupied_letters, sg_num = get_occupied_wyckoff_letters(structure)
    vacant_wyckoffs = get_vacant_wyckoff_letters(sg_num, occupied_letters)

    old_species = old_struc.species.copy()
    old_numIons = old_struc.numIons.copy()
    old_lattice = old_struc.lattice.copy()
    old_sites = extract_wyckoff_positions(old_struc, work_in=False)

    group = Group(sg_num)
    records = []

    for w in vacant_wyckoffs:
        coords = extract_coords_from_wyckoff_letter(group, w)
        match = re.match(r"(\d+)([a-z]+)", w)
        if not match:
            continue
        multiplicity = int(match.group(1))
        new_species = old_species + [cation]
        new_numIons = np.append(old_numIons, multiplicity)

        if coords:
            # Deterministic insertion
            new_sites = old_sites + [(w, c) for c in coords]
            xtal = pyxtal()
            xtal.from_random(
                dim=3, group=sg_num, species=new_species, numIons=new_numIons,
                lattice=old_lattice, sites=new_sites, tm=Tol_matrix.from_single_value(1)
            )
            new_pm_struct = xtal.to_pymatgen()
            records.append({
                "structure": new_pm_struct,
                "wyckoff_added": w,
                "random_attempt": None,
            })
        else:
            # Random insertion attempts
            for attempt in range(10):
                try:
                    xtal = pyxtal()
                    xtal.from_random(
                        dim=3, group=sg_num, species=new_species, numIons=new_numIons,
                        lattice=old_lattice, sites=old_sites + [(w, None)],
                        tm=Tol_matrix.from_single_value(1)
                    )
                    struct = xtal.to_pymatgen()
                    if is_structure_valid(struct):
                        records.append({
                            "structure": struct,
                            "wyckoff_added": w,
                            "random_attempt": attempt + 1,
                        })
                        break
                except Exception:
                    continue

    return pd.DataFrame(records)
