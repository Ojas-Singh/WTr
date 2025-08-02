"""
Constraints for water cluster optimization.
"""

import numpy as np
from typing import List, Optional
from ase.constraints import FixAtoms, Hookean

from ..models.datatypes import Geometry, SurfaceSpec

def select_core_atoms(cluster: Geometry, spec: SurfaceSpec) -> List[int]:
    """
    Select core atom indices based on distance from cluster centroid.
    
    Args:
        cluster: Water cluster geometry
        spec: Surface specification with core_fraction
    
    Returns:
        List of atom indices for core atoms
    """
    coords = np.array(cluster.coords)
    symbols = cluster.symbols
    
    # Find oxygen atoms (one per water molecule)
    oxygen_indices = [i for i, symbol in enumerate(symbols) if symbol == 'O']
    
    if not oxygen_indices:
        return []
    
    # Calculate centroid of oxygen atoms
    oxygen_coords = coords[oxygen_indices]
    centroid = np.mean(oxygen_coords, axis=0)
    
    # Calculate distances from centroid
    distances = []
    for i, o_idx in enumerate(oxygen_indices):
        dist = np.linalg.norm(oxygen_coords[i] - centroid)
        distances.append((dist, o_idx))
    
    # Sort by distance and select inner fraction
    distances.sort(key=lambda x: x[0])
    n_core_waters = max(1, int(len(oxygen_indices) * spec.core_fraction))
    
    core_water_indices = [distances[i][1] for i in range(n_core_waters)]
    
    # Include all atoms of core water molecules (O + 2H each)
    core_atom_indices = []
    for o_idx in core_water_indices:
        # Add oxygen
        core_atom_indices.append(o_idx)
        
        # Find corresponding hydrogens (next two atoms in typical water ordering)
        # More robust: find H atoms within 1.2 Å of this O
        o_pos = coords[o_idx]
        for i, symbol in enumerate(symbols):
            if symbol == 'H':
                h_pos = coords[i]
                if np.linalg.norm(o_pos - h_pos) < 1.2:  # OH bond length ~ 0.96 Å
                    core_atom_indices.append(i)
    
    return sorted(core_atom_indices)

def make_constraints(atoms, core_indices: List[int], k: Optional[float] = None):
    """
    Create ASE constraints for core atoms.
    
    Args:
        atoms: ASE Atoms object
        core_indices: Indices of core atoms
        k: Harmonic force constant (eV/Å²). If None, use FixAtoms
    
    Returns:
        List of ASE constraint objects
    """
    if k is None:
        # Hard constraints
        return [FixAtoms(indices=core_indices)]
    else:
        # Soft harmonic constraints
        constraints = []
        positions = atoms.get_positions()
        
        for idx in core_indices:
            # Create Hookean constraint for each core atom
            constraint = Hookean(a1=idx, a2=(0, 0, 0), k=k, rt=0.0)
            # Note: This is a simplified version - real implementation would
            # need to properly set up the reference positions
            constraints.append(constraint)
        
        return constraints
