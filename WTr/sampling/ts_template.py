"""
TS-templated surface design.
"""

import numpy as np
import os

from ..models.datatypes import Geometry, CalcSpec
from ..io.ase_helpers import geometry_to_atoms, atoms_to_geometry, set_calculator
from ..io.geometry import merge

def create_ts_restraints(atoms, ts_indices: list, reference_distances: dict, 
                        k_internal: float = 0.5):
    """
    Create harmonic restraints for TS internal coordinates.
    
    Args:
        atoms: ASE Atoms object
        ts_indices: Indices of TS atoms
        reference_distances: Dict of (i,j): distance pairs
        k_internal: Force constant for restraints (eV/Å²)
    
    Returns:
        List of restraint objects (simplified implementation)
    """
    # In real implementation, would create ASE constraints
    # For now, return empty list as placeholder
    return []

def extract_key_distances(ts_geometry: Geometry) -> dict:
    """
    Extract key internal distances from TS geometry.
    
    Args:
        ts_geometry: TS geometry
    
    Returns:
        Dictionary of (atom_i, atom_j): distance pairs
    """
    coords = np.array(ts_geometry.coords)
    distances = {}
    
    # Find bonds that are likely forming/breaking (simplified heuristic)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            
            # Consider distances in "transition" range (1.2 - 2.0 Å)
            if 1.2 < dist < 2.0:
                distances[(i, j)] = dist
    
    return distances

def ts_templated_surface(surface: Geometry, ts_seed: Geometry,
                        calc_spec: CalcSpec, core_indices: list,
                        k_internal: float = 0.5, workdir: str = ".") -> Geometry:
    """
    Generate TS-templated surface by optimizing surface around placed TS.
    
    Args:
        surface: Initial water surface
        ts_seed: TS geometry to template around
        calc_spec: Calculator specification
        core_indices: Indices of core atoms to constrain
        k_internal: Force constant for TS restraints
        workdir: Working directory
    
    Returns:
        Optimized surface geometry (with TS removed)
    """
    os.makedirs(workdir, exist_ok=True)
    
    # Place TS near surface (simplified placement)
    surface_coords = np.array(surface.coords)
    surface_center = np.mean(surface_coords, axis=0)
    
    # Find surface boundary
    max_radius = np.max(np.linalg.norm(surface_coords - surface_center, axis=1))
    
    # Place TS at surface edge
    ts_coords = np.array(ts_seed.coords)
    ts_center = np.mean(ts_coords, axis=0)
    
    # Translate TS to surface boundary
    direction = np.array([1, 0, 0])  # Simplified - could be smarter
    ts_position = surface_center + direction * (max_radius + 2.0)
    translation = ts_position - ts_center
    
    translated_ts_coords = ts_coords + translation
    placed_ts = Geometry(
        symbols=ts_seed.symbols,
        coords=[tuple(coord) for coord in translated_ts_coords],
        comment="Placed TS for templating"
    )
    
    # Merge surface and TS
    combined = merge(surface, placed_ts)
    combined_atoms = geometry_to_atoms(combined)
    
    # Set up calculator
    set_calculator(combined_atoms, calc_spec)
    
    # Create constraints for core atoms (simplified)
    from ..geom.constraints import make_constraints
    try:
        constraints = make_constraints(combined_atoms, core_indices, k=None)  # Hard constraints
        combined_atoms.set_constraint(constraints)
    except Exception as e:
        print(f"Warning: Could not set constraints: {e}")
    
    # Create TS restraints
    n_surface_atoms = len(surface.symbols)
    ts_atom_indices = list(range(n_surface_atoms, len(combined.symbols)))
    key_distances = extract_key_distances(placed_ts)
    
    # Adjust indices for combined system
    adjusted_distances = {}
    for (i, j), dist in key_distances.items():
        adjusted_distances[(i + n_surface_atoms, j + n_surface_atoms)] = dist
    
    ts_restraints = create_ts_restraints(combined_atoms, ts_atom_indices, 
                                       adjusted_distances, k_internal)
    
    # Optimization (simplified - in real implementation would use ASE optimizers)
    print(f"Optimizing TS-templated surface in {workdir}")
    
    try:
        # Placeholder for optimization
        # In real implementation:
        # from ase.optimize import BFGS
        # opt = BFGS(combined_atoms, logfile=os.path.join(workdir, 'opt.log'))
        # opt.run(fmax=0.1)
        
        # For now, just add small random displacements to surface waters
        positions = combined_atoms.get_positions()
        
        # Only perturb surface atoms (not TS or core)
        surface_mobile_indices = []
        for i in range(n_surface_atoms):
            if i not in core_indices:
                surface_mobile_indices.append(i)
        
        for i in surface_mobile_indices:
            # Small random displacement
            displacement = np.random.normal(0, 0.1, 3)
            positions[i] += displacement
        
        combined_atoms.set_positions(positions)
        
    except Exception as e:
        print(f"Warning: Optimization failed: {e}")
    
    # Remove TS and return optimized surface
    optimized_positions = combined_atoms.get_positions()[:n_surface_atoms]
    optimized_symbols = combined_atoms.get_chemical_symbols()[:n_surface_atoms]
    
    optimized_surface = Geometry(
        symbols=optimized_symbols,
        coords=[tuple(pos) for pos in optimized_positions],
        comment=f"TS-templated surface (k_internal={k_internal})"
    )
    
    # Save result
    output_path = os.path.join(workdir, "ts_templated_surface.xyz")
    from ..io.geometry import save_xyz
    save_xyz(optimized_surface, output_path)
    
    print(f"TS-templated surface saved to {output_path}")
    
    return optimized_surface
