import pandas as pd
import numpy as np
import argparse
import logging
import time
import warnings
from pathlib import Path
from fractions import Fraction
from itertools import product
import re

from pymatgen.core.composition import Composition
from pyxtal.symmetry import Group
from pyxtal import pyxtal

# Suppress warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

# ============================================================================

# CONSTANTS

# ============================================================================

FARADAY = 96485  # C/mol
CONVERSION = 3.6  # to mAh/g
OXIDATION_STATES = {'Li': 1, 'Na': 1, 'K': 1, 'Ca': 2, 'Mg': 2}

TOXIC_RADIOACTIVE_ELEMENTS = {
    'U', 'Th', 'Pu', 'Ra', 'Ac', 'Pa', 'Np', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Tc', 'Po', 'At', 'Rn', 'Fr',
    'As', 'Cd', 'Hg', 'Pb', 'Tl', 'Be', 'Sb', 'Se', 'Ba',
}

# ============================================================================

# LOGGING SETUP

# ============================================================================

def setup_logging(verbose, log_file='intercalation.log'):
    """Setup logging based on verbosity level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    # Also log to console in verbose mode
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

# ============================================================================

# VALIDATION FUNCTIONS

# ============================================================================

def contains_toxic_elements(structure):
    """Check if structure contains toxic or radioactive elements"""
    elements = set([str(element) for element in structure.composition.elements])
    toxic_found = elements.intersection(TOXIC_RADIOACTIVE_ELEMENTS)
    
    if toxic_found:
        return True, sorted(list(toxic_found))
    return False, []

def check_minimum_distance(structure, min_distance=1.0):
    """Check if all interatomic distances are >= min_distance"""
    dmat = structure.distance_matrix
    np.fill_diagonal(dmat, np.inf)
    min_dist = np.min(dmat)
    return np.all(dmat >= min_distance), min_dist

def validate_composition_change(old_structure, new_structure, cation, cation_was_present):
    """Validate composition change - supercells allowed"""
    
    old_comp = old_structure.composition
    old_formula = old_comp.reduced_formula
    
    new_comp = new_structure.composition
    new_formula = new_comp.reduced_formula
    
    old_cation_count = old_comp.get(cation, 0)
    new_cation_count = new_comp.get(cation, 0)
    
    # 1. Cation count must increase
    cation_increase = new_cation_count - old_cation_count
    if cation_increase <= 0:
        return False
    
    # 2. Total atom count must increase
    atom_increase = new_comp.num_atoms - old_comp.num_atoms
    if atom_increase <= 0:
        return False
    
    # 3. IMPORTANT: Reduced formula must change
    if old_formula == new_formula:
        return False
    
    # 4. For new cation: Must appear in new formula
    if not cation_was_present:
        if cation not in new_comp:
            return False
    
    # 5. No elements may disappear
    old_elements = set(str(el) for el in old_comp.elements)
    new_elements = set(str(el) for el in new_comp.elements)
    
    if not old_elements.issubset(new_elements):
        return False
    
    return True

# ============================================================================

# CAPACITY CALCULATION

# ============================================================================

def compute_gravimetric_capacity(structure, electrons):
    """Calculate gravimetric capacity in mAh/g"""
    molar_mass = Composition(structure.composition).weight
    return (electrons * FARADAY) / (CONVERSION * molar_mass)

# ============================================================================

# WYCKOFF POSITION HANDLING

# ============================================================================

def extract_wyckoff_positions(crystal, work_in=False):
    """Extract Wyckoff positions from pyxtal crystal"""
    new_wyckoff_data = []
    el_list = []
    
    for site in crystal.atom_sites:
        el_list.append(site.specie)
        wyckoff_position = f"{site.wp.multiplicity}{site.wp.letter}"
        coord = site.coords[0].tolist()
        new_wyckoff_data.append(tuple([wyckoff_position, coord[0], coord[1], coord[2]]))
    
    element_dict = {element: [] for element in el_list}
    for tup, element in zip(new_wyckoff_data, el_list):
        if element not in element_dict:
            element_dict[element] = []
        if tup is not None:
            element_dict[element].append(tup)
    
    indexes = np.unique(el_list, return_index=True)[1]
    new_el_list = [el_list[index] for index in sorted(indexes)]
    result = [element_dict[element] for element in new_el_list]
    
    if not work_in:
        result.append(None)
    
    return result

def get_occupied_wyckoff_letters_pyxtal(structure):
    """Determine occupied Wyckoff positions from pyxtal"""
    try:
        xtal = pyxtal()
        xtal.from_seed(structure)
        
        occupied = set()
        for site in xtal.atom_sites:
            element = site.specie
            wyckoff_letter = site.wp.letter
            occupied.add((element, wyckoff_letter))
        
        sg_num = xtal.group.number
        logging.debug(f"  Spacegroup: {sg_num}, Occupied Wyckoff: {occupied}")
        
        return occupied, sg_num, xtal
    except Exception as e:
        logging.error(f"Failed to analyze structure with pyxtal: {str(e)}")
        raise

def get_available_wyckoff_positions(sg_num, occupied_letters):
    """
    Determine available Wyckoff positions:
    - Fixed positions: Only if NOT occupied
    - Variable positions: ALWAYS available (even if occupied)

    """
    group = Group(sg_num)
    all_wyckoffs = group.Wyckoff_positions
    occupied_set = {w for (_, w) in occupied_letters}
    
    available = []
    
    for wp in all_wyckoffs:
        letter = wp.letter
        
        # Check if position has variables
        wp_str_repr = str(wp)
        has_variables = any(var in wp_str_repr for var in ['x', 'y', 'z'])
        
        if has_variables:
            # Variable position -> ALWAYS available (even if occupied)
            available.append((letter, True))
            logging.debug(f"    {letter}: VARIABLE -> available (even if occupied)")
        else:
            # Fixed position -> only if not occupied
            if letter not in occupied_set:
                available.append((letter, False))
                logging.debug(f"    {letter}: FIXED & unoccupied -> available")
            else:
                logging.debug(f"    {letter}: FIXED & occupied -> NOT available")
    
    logging.debug(f"  Available Wyckoff positions: {[l for l, _ in available]}")
    return available

def parse_number(s):
    """Parse string to float, including fractions"""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        try:
            return float(Fraction(s))
        except ValueError:
            raise ValueError(f"Cannot parse number: {s}")

def extract_coords_from_wyckoff_letter(group, letter):
    """Extract coordinates from Wyckoff letter"""
    wp = group.get_wp_by_letter(letter)
    lines = str(wp).split('\n')[1:]
    coords = []
    has_variables = False
    coord_strings = []
    
    try:
        for line in lines:
            if not line.strip():
                continue
            coord_strings.append(line.strip())
            if any(var in line for var in ['x', 'y', 'z']):
                has_variables = True
        
        if not has_variables:
            for line in coord_strings:
                parts = line.replace(' ', '').split(',')
                coords.append([parse_number(x) for x in parts])
            return coords, False, None
        else:
            return None, True, coord_strings
    except Exception as e:
        logging.debug(f"    Error parsing Wyckoff letter {letter}: {str(e)}")
        return None, True, None

def create_grid_for_variable(lattice_param):
    """Create grid points based on 1 Angstrom spacing"""
    n_points = int(np.ceil(lattice_param / 1.0))
    if n_points < 2:
        n_points = 2
    grid = np.linspace(0, 1, n_points, endpoint=False)
    logging.debug(f"    Created grid with {n_points} points for lattice param {lattice_param:.2f}")
    return grid

def parse_coord_expression(expr, var_dict):
    """Parse coordinate expression and replace variables"""
    original_expr = expr
    expr = expr.replace(' ', '')
    
    for var, val in var_dict.items():
        expr = expr.replace(var, str(val))
    
    tokens = re.split(r'([+\-*/])', expr)
    
    result = 0.0
    current_op = '+'
    
    for token in tokens:
        token = token.strip()
        if not token:
            continue
            
        if token in ['+', '-', '*', '/']:
            current_op = token
        else:
            try:
                if '/' in token:
                    parts = token.split('/')
                    if len(parts) == 2:
                        num = float(parts[0])
                        denom = float(parts[1])
                        value = num / denom
                    else:
                        value = float(token)
                else:
                    value = float(token)
                
                if current_op == '+':
                    result += value
                elif current_op == '-':
                    result -= value
                elif current_op == '*':
                    result *= value
                elif current_op == '/':
                    result /= value
                    
            except Exception as e:
                logging.warning(f"Failed to parse token '{token}' in expression '{original_expr}': {e}")
                result = 0.0
                break
    
    result = result % 1.0
    return result

def generate_variable_coords(coord_strings, lattice):
    """Generate all possible coordinates for variable Wyckoff positions"""
    used_vars = set()
    for coord_str in coord_strings:
        if 'x' in coord_str:
            used_vars.add('x')
        if 'y' in coord_str:
            used_vars.add('y')
        if 'z' in coord_str:
            used_vars.add('z')
    
    grids = {}
    if 'x' in used_vars:
        grids['x'] = create_grid_for_variable(lattice.a)
    if 'y' in used_vars:
        grids['y'] = create_grid_for_variable(lattice.b)
    if 'z' in used_vars:
        grids['z'] = create_grid_for_variable(lattice.c)
    
    var_names = sorted(grids.keys())
    var_grids = [grids[v] for v in var_names]
    all_combinations = list(product(*var_grids))
    
    logging.debug(f"    Generated {len(all_combinations)} variable combinations")
    return all_combinations, var_names, grids

# ============================================================================

# MAIN PROCESSING FUNCTION

# ============================================================================

def process_structure_with_cation(structure, cation, material_id='unknown', verbose=False):
    """Process a single host structure with one cation"""
    results = []
    
    start_time = time.time()
    formula = structure.composition.reduced_formula
    
    logging.info(f"Processing {material_id} ({formula}) with cation {cation}")
    
    try:
        oxidation = OXIDATION_STATES.get(cation, 1)
        logging.debug(f"  Oxidation state for {cation}: {oxidation}")
        
        # Analyze structure with pyxtal
        occupied_letters, sg_num, host_xtal = get_occupied_wyckoff_letters_pyxtal(structure)
        
        # Get available Wyckoff positions
        available_wyckoffs = get_available_wyckoff_positions(sg_num, occupied_letters)
        
        if not available_wyckoffs:
            logging.warning(f"  No available Wyckoff positions found")
            return results
        
        group = Group(sg_num)
        fixed_positions = []
        variable_positions = []
        
        # Categorize positions
        for letter, is_var_from_check in available_wyckoffs:
            coords, has_vars, coord_strings = extract_coords_from_wyckoff_letter(group, letter)
            
            wp = group.get_wp_by_letter(letter)
            multiplicity = wp.multiplicity
            wp_str = f"{multiplicity}{letter}"
            
            if has_vars:
                variable_positions.append((wp_str, letter, coord_strings, multiplicity))
            else:
                if coords:
                    fixed_positions.append((wp_str, letter, coords, multiplicity))
        
        logging.debug(f"  Found {len(fixed_positions)} fixed and {len(variable_positions)} variable positions")
        
        # Extract old sites
        old_sites = extract_wyckoff_positions(host_xtal, work_in=False)
        if old_sites and old_sites[-1] is None:
            old_sites = old_sites[:-1]
        
        # Check if cation already present
        existing_species = list(host_xtal.species)
        cation_already_present = cation in existing_species
        
        if cation_already_present:
            cation_index = existing_species.index(cation)
            logging.debug(f"  Cation {cation} already present at index {cation_index}")
        
        # Process fixed positions
        for wp_str, letter, coords, multiplicity in fixed_positions:
            if coords is None or len(coords) == 0:
                continue
            
            first_coord = coords[0]
            
            try:
                xtal = pyxtal()
                
                if cation_already_present:
                    species_list = list(host_xtal.species)
                    num_ions_list = list(host_xtal.numIons)
                    
                    new_sites = [list(site) for site in old_sites]
                    new_sites[cation_index] = list(new_sites[cation_index])
                    new_sites[cation_index].append((wp_str, first_coord[0], first_coord[1], first_coord[2]))
                else:
                    species_list = list(host_xtal.species) + [cation]
                    num_ions_list = list(host_xtal.numIons) + [0]
                    
                    new_sites = [list(site) for site in old_sites]
                    cation_sites = [(wp_str, first_coord[0], first_coord[1], first_coord[2])]
                    new_sites.append(cation_sites)
                
                xtal.build(
                    group=sg_num,
                    species=species_list,
                    numIons=num_ions_list,
                    lattice=host_xtal.lattice,
                    sites=new_sites,
                    dim=3
                )
                
                new_structure = xtal.to_pymatgen()
                
                # Validation
                is_valid = validate_composition_change(
                    structure, new_structure, cation, cation_already_present
                )
                
                if not is_valid:
                    logging.debug(f"    Fixed position {wp_str}: Validation failed")
                    continue
                
                # Distance check
                is_valid_dist, min_dist = check_minimum_distance(new_structure, min_distance=1.0)
                
                if not is_valid_dist:
                    logging.debug(f"    Fixed position {wp_str}: Distance check failed (min={min_dist:.3f}A)")
                    continue
                
                electrons_transferred = oxidation * multiplicity
                new_formula = new_structure.composition.reduced_formula
                
                results.append({
                    'parent_id': material_id,
                    'parent_formula': formula,
                    'new_formula': new_formula,
                    'added_element': cation,
                    'wyckoff_added': wp_str,
                    'added_ions': multiplicity,
                    'electrons_transferred': electrons_transferred,
                    'new_struc': new_structure,
                    'old_struc': structure,
                    'attempt': 0,
                    'spacegroup_number': sg_num,
                    'is_variable': False,
                    'cation_was_present': cation_already_present
                })
                
                logging.debug(f"    Fixed position {wp_str}: SUCCESS (min_dist={min_dist:.3f}A, new_formula={new_formula})")
                
            except Exception as e:
                logging.debug(f"    Fixed position {wp_str}: Build failed - {str(e)}")
                continue
        
        # Process variable positions
        for wp_str, letter, coord_strings, multiplicity in variable_positions:
            if coord_strings is None:
                continue
            
            all_combinations, var_names, grids = generate_variable_coords(coord_strings, host_xtal.lattice)
            
            for combo_idx, combo in enumerate(all_combinations):
                var_dict = {var: val for var, val in zip(var_names, combo)}
                
                try:
                    xtal = pyxtal()
                    
                    # Parse first coordinate
                    coord_str = coord_strings[0]
                    parts = coord_str.replace(' ', '').split(',')
                    coord = []
                    for part in parts:
                        coord.append(parse_coord_expression(part, var_dict))
                    
                    if cation_already_present:
                        species_list = list(host_xtal.species)
                        num_ions_list = list(host_xtal.numIons)
                        
                        new_sites = [list(site) for site in old_sites]
                        new_sites[cation_index] = list(new_sites[cation_index])
                        new_sites[cation_index].append((wp_str, coord[0], coord[1], coord[2]))
                    else:
                        species_list = list(host_xtal.species) + [cation]
                        num_ions_list = list(host_xtal.numIons) + [0]
                        
                        new_sites = [list(site) for site in old_sites]
                        cation_sites = [(wp_str, coord[0], coord[1], coord[2])]
                        new_sites.append(cation_sites)
                    
                    xtal.build(
                        group=sg_num,
                        species=species_list,
                        numIons=num_ions_list,
                        lattice=host_xtal.lattice,
                        sites=new_sites,
                        dim=3
                    )
                    
                    new_structure = xtal.to_pymatgen()
                    
                    # Validation
                    is_valid = validate_composition_change(
                        structure, new_structure, cation, cation_already_present
                    )
                    
                    if not is_valid:
                        continue
                    
                    # Distance check
                    is_valid_dist, min_dist = check_minimum_distance(new_structure, min_distance=1.0)
                    
                    if not is_valid_dist:
                        continue
                    
                    electrons_transferred = oxidation * multiplicity
                    new_formula = new_structure.composition.reduced_formula
                    
                    results.append({
                        'parent_id': material_id,
                        'parent_formula': formula,
                        'new_formula': new_formula,
                        'added_element': cation,
                        'wyckoff_added': wp_str,
                        'added_ions': multiplicity,
                        'electrons_transferred': electrons_transferred,
                        'new_struc': new_structure,
                        'old_struc': structure,
                        'attempt': combo_idx + 1,
                        'spacegroup_number': sg_num,
                        'is_variable': True,
                        'variable_values': var_dict,
                        'cation_was_present': cation_already_present
                    })
                    
                except Exception as e:
                    continue
        
        elapsed = time.time() - start_time
        logging.info(f"Completed in {elapsed:.2f}s: {len(results)} structures generated")
    
    except Exception as e:
        logging.error(f"Processing with {cation} FAILED: {str(e)}", exc_info=True)
    
    return results

# ============================================================================

# MAIN INTERCALATION FUNCTION

# ============================================================================

def intercalate_structure(structure, cations=None, material_id='unknown', 
                         filter_toxic=False, verbose=False):
    """
    Main function to intercalate a structure with cations
    
    Parameters:
    -----------
    structure : pymatgen.core.structure.Structure
        The host structure to intercalate
    cations : list of str, optional
        List of cations to test. Default: ['Li', 'Na', 'K', 'Mg', 'Ca']
    material_id : str, optional
        Identifier for the structure
    filter_toxic : bool, optional
        If True, skip structures with toxic/radioactive elements
    verbose : bool, optional
        Enable verbose output
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all generated structures, sorted by gravimetric capacity
    """
    
    if cations is None:
        cations = ['Li', 'Na', 'K', 'Mg', 'Ca']
    
    print("="*70)
    print("STRUCTURE INTERCALATION")
    print("="*70)
    
    formula = structure.composition.reduced_formula
    print(f"\nHost structure: {material_id} ({formula})")
    print(f"Cations to test: {', '.join(cations)}")
    
    # Check for toxic elements
    if filter_toxic:
        is_toxic, toxic_els = contains_toxic_elements(structure)
        if is_toxic:
            print(f"\nWARNING: Structure contains toxic/radioactive elements: {', '.join(toxic_els)}")
            print("Skipping structure due to filter_toxic=True")
            return None
    
    # Process with all cations
    all_results = []
    
    for cation in cations:
        print(f"\nProcessing with cation: {cation}")
        results = process_structure_with_cation(structure, cation, material_id, verbose)
        
        if results:
            # Calculate gravimetric capacity
            for result in results:
                result['gravimetric_capacity_mAh_g'] = compute_gravimetric_capacity(
                    result['new_struc'],
                    result['electrons_transferred']
                )
            
            all_results.extend(results)
            print(f"  -> Generated {len(results)} structures")
        else:
            print(f"  -> No valid structures generated")
    
    if not all_results:
        print("\n" + "="*70)
        print("WARNING: No valid structures generated!")
        print("="*70)
        return None
    
    # Create DataFrame and sort
    df_results = pd.DataFrame(all_results)
    df_sorted = df_results.sort_values(
        by="gravimetric_capacity_mAh_g", 
        ascending=False
    ).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total structures generated: {len(df_sorted)}")
    print(f"\nBy cation:")
    for cation in cations:
        count = len(df_sorted[df_sorted['added_element'] == cation])
        print(f"  {cation}: {count} structures")
    
    print(f"\nTop 5 by gravimetric capacity:")
    print(df_sorted[['parent_formula', 'new_formula', 'added_element', 
                     'wyckoff_added', 'gravimetric_capacity_mAh_g']].head().to_string())
    
    print(f"{'='*70}\n")
    
    return df_sorted

# ============================================================================

# COMMAND LINE INTERFACE

# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Intercalate a pymatgen structure with cations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python intercalate_structure.py --input structure.pkl --output results.pkl
  python intercalate_structure.py --input structure.pkl --cations Li Na --verbose
  python intercalate_structure.py --input structure.pkl --filter-toxic --output filtered_results.pkl
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input pickle file containing pymatgen Structure')
    parser.add_argument('--output', '-o', default='intercalated_structures.pkl',
                       help='Output pickle file for results (default: intercalated_structures.pkl)')
    parser.add_argument('--cations', '-c', nargs='+', 
                       default=['Li', 'Na', 'K', 'Mg', 'Ca'],
                       help='Cations to test (default: Li Na K Mg Ca)')
    parser.add_argument('--material-id', '-m', default='unknown',
                       help='Material ID for the structure')
    parser.add_argument('--filter-toxic', '-f', action='store_true',
                       help='Skip structures with toxic/radioactive elements')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output and detailed logging')
    parser.add_argument('--log-file', default='intercalation.log',
                       help='Log file name (default: intercalation.log)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Load structure
    print(f"Loading structure from: {args.input}")
    try:
        with open(args.input, 'rb') as f:
            import pickle
            structure = pickle.load(f)
        print(f"[OK] Structure loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load structure: {str(e)}")
        return 1
    
    # Process structure
    df_results = intercalate_structure(
        structure=structure,
        cations=args.cations,
        material_id=args.material_id,
        filter_toxic=args.filter_toxic,
        verbose=args.verbose
    )
    
    # Save results
    if df_results is not None and len(df_results) > 0:
        df_results.to_pickle(args.output)
        print(f"[OK] Results saved to: {args.output}")
        print(f"[OK] Log saved to: {args.log_file}")
        return 0
    else:
        print("[WARNING] No results to save")
        return 1

if __name__ == "__main__":
    exit(main())
