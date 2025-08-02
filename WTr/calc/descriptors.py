"""
Descriptors and physics-based scoring functions.
"""

import numpy as np
from typing import List, Dict, Tuple
import networkx as nx

from ..models.datatypes import DescriptorSet

def compute_electric_field(charges: np.ndarray, positions: np.ndarray, 
                          field_point: np.ndarray) -> np.ndarray:
    """
    Compute electric field at a point due to point charges.
    
    Args:
        charges: Point charges (elementary charge units)
        positions: Charge positions (Å)
        field_point: Point where to compute field (Å)
    
    Returns:
        Electric field vector (V/Å, atomic units with 4πε₀=1)
    """
    field = np.zeros(3)
    
    for i, (q, pos) in enumerate(zip(charges, positions)):
        r_vec = field_point - pos
        r = np.linalg.norm(r_vec)
        
        if r > 1e-6:  # Avoid singularity
            # E = q * r_vec / r³ (atomic units)
            field += q * r_vec / (r**3)
    
    return field

def get_model_charges(symbols: List[str]) -> np.ndarray:
    """
    Get simple model charges for atoms.
    
    Args:
        symbols: List of atomic symbols
    
    Returns:
        Array of partial charges
    """
    charge_dict = {
        'O': -0.8476,  # Water oxygen
        'H': 0.4238,   # Water hydrogen
        'C': 0.0,      # Neutral carbon (placeholder)
        'N': -0.3,     # Approximate nitrogen
    }
    
    charges = []
    for symbol in symbols:
        charges.append(charge_dict.get(symbol, 0.0))
    
    return np.array(charges)

def find_hbond_pairs(positions: np.ndarray, symbols: List[str], 
                    r_cutoff: float = 3.3, angle_cutoff: float = 140.0) -> List[Tuple[int, int]]:
    """
    Find hydrogen bond donor-acceptor pairs.
    
    Args:
        positions: Atomic positions (Å)
        symbols: Atomic symbols
        r_cutoff: O···O distance cutoff (Å)
        angle_cutoff: H-O···O angle cutoff (degrees)
    
    Returns:
        List of (donor_O_idx, acceptor_O_idx) pairs
    """
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    
    hbond_pairs = []
    angle_cutoff_rad = np.radians(angle_cutoff)
    
    for i, o1_idx in enumerate(o_indices):
        for j, o2_idx in enumerate(o_indices):
            if i >= j:  # Avoid double counting and self
                continue
            
            o1_pos = positions[o1_idx]
            o2_pos = positions[o2_idx]
            oo_dist = np.linalg.norm(o1_pos - o2_pos)
            
            if oo_dist > r_cutoff:
                continue
            
            # Check if o1 can donate to o2
            for h_idx in h_indices:
                h_pos = positions[h_idx]
                oh_dist = np.linalg.norm(o1_pos - h_pos)
                
                if oh_dist < 1.2:  # H belongs to o1
                    # Calculate H-O1···O2 angle
                    ho1_vec = o1_pos - h_pos
                    o1o2_vec = o2_pos - o1_pos
                    
                    cos_angle = np.dot(ho1_vec, o1o2_vec) / (np.linalg.norm(ho1_vec) * np.linalg.norm(o1o2_vec))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    
                    if angle >= angle_cutoff_rad:
                        hbond_pairs.append((o1_idx, o2_idx))
                        break
    
    return hbond_pairs

def count_donors_acceptors(positions: np.ndarray, symbols: List[str], 
                          reactive_point: np.ndarray, cutoff: float = 3.2) -> Tuple[int, int]:
    """
    Count H-bond donors and acceptors near reactive site.
    
    Args:
        positions: Atomic positions
        symbols: Atomic symbols  
        reactive_point: Position of reactive site
        cutoff: Distance cutoff for counting
    
    Returns:
        (donor_count, acceptor_count)
    """
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    
    # Find waters near reactive site
    nearby_oxygens = []
    for o_idx in o_indices:
        o_pos = positions[o_idx]
        if np.linalg.norm(o_pos - reactive_point) < cutoff:
            nearby_oxygens.append(o_idx)
    
    # Count donors (waters with available H)
    donor_count = 0
    acceptor_count = 0
    
    for o_idx in nearby_oxygens:
        o_pos = positions[o_idx]
        
        # Count hydrogens on this oxygen
        h_count = 0
        for h_idx in h_indices:
            h_pos = positions[h_idx]
            if np.linalg.norm(o_pos - h_pos) < 1.2:
                h_count += 1
        
        # Each H can act as donor, O can accept 2 H-bonds
        donor_count += h_count
        acceptor_count += max(0, 2 - h_count)  # Remaining lone pairs
    
    return donor_count, acceptor_count

def analyze_proton_wire(positions: np.ndarray, symbols: List[str],
                       start_idx: int, end_idx: int) -> Tuple[float, float]:
    """
    Analyze proton wire path between two points.
    
    Args:
        positions: Atomic positions
        symbols: Atomic symbols
        start_idx: Starting oxygen index
        end_idx: Target oxygen index
    
    Returns:
        (path_length, max_bend_angle)
    """
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']
    
    # Build H-bond graph
    G = nx.DiGraph()
    G.add_nodes_from(o_indices)
    
    hbond_pairs = find_hbond_pairs(positions, symbols)
    for donor, acceptor in hbond_pairs:
        G.add_edge(donor, acceptor)
    
    # Find shortest path
    try:
        path = nx.shortest_path(G, start_idx, end_idx)
    except nx.NetworkXNoPath:
        return float('inf'), float('inf')
    
    if len(path) < 2:
        return 0.0, 0.0
    
    # Calculate path length
    path_length = 0.0
    for i in range(len(path) - 1):
        pos1 = positions[path[i]]
        pos2 = positions[path[i + 1]]
        path_length += np.linalg.norm(pos2 - pos1)
    
    # Calculate maximum bend angle
    max_bend = 0.0
    for i in range(len(path) - 2):
        pos1 = positions[path[i]]
        pos2 = positions[path[i + 1]]
        pos3 = positions[path[i + 2]]
        
        vec1 = pos2 - pos1
        vec2 = pos3 - pos2
        
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        bend_angle = np.pi - angle  # Supplement of the angle
        
        max_bend = max(max_bend, np.degrees(bend_angle))
    
    return path_length, max_bend

def calculate_strain_penalty(positions: np.ndarray, symbols: List[str]) -> float:
    """
    Calculate strain penalty from distorted H-bonds.
    
    Args:
        positions: Atomic positions
        symbols: Atomic symbols
    
    Returns:
        Strain penalty (dimensionless)
    """
    hbond_pairs = find_hbond_pairs(positions, symbols)
    
    strain = 0.0
    d0 = 2.8  # Reference H-bond distance (Å)
    sigma_d = 0.15  # Distance tolerance (Å)
    
    for donor_idx, acceptor_idx in hbond_pairs:
        donor_pos = positions[donor_idx]
        acceptor_pos = positions[acceptor_idx]
        distance = np.linalg.norm(acceptor_pos - donor_pos)
        
        strain += ((distance - d0) / sigma_d) ** 2
    
    return strain

def compute_descriptors(surface_atoms, rc_atoms, reaction_axis: np.ndarray,
                       reactive_point: np.ndarray, charges: np.ndarray) -> DescriptorSet:
    """
    Compute full descriptor set for a surface configuration.
    
    Args:
        surface_atoms: ASE Atoms object for surface
        rc_atoms: ASE Atoms object for reactant complex
        reaction_axis: Unit vector defining reaction direction
        reactive_point: 3D point characterizing reactive region
        charges: Partial charges for all atoms
    
    Returns:
        DescriptorSet with all computed descriptors
    """
    # Get positions and symbols
    all_positions = np.vstack([surface_atoms.get_positions(), rc_atoms.get_positions()])
    all_symbols = surface_atoms.get_chemical_symbols() + rc_atoms.get_chemical_symbols()
    
    # Electric field at reactive point
    efield = compute_electric_field(charges, all_positions, reactive_point)
    efield_proj = np.dot(efield, reaction_axis)
    
    # Donor/acceptor counts
    donor_count, acceptor_count = count_donors_acceptors(all_positions, all_symbols, reactive_point)
    
    # Find suitable atoms for proton wire analysis
    o_indices = [i for i, s in enumerate(all_symbols) if s == 'O']
    if len(o_indices) >= 2:
        # Use first two oxygens as start/end (simplified)
        wire_length, wire_bend_max = analyze_proton_wire(all_positions, all_symbols, 
                                                       o_indices[0], o_indices[-1])
    else:
        wire_length, wire_bend_max = 0.0, 0.0
    
    # Strain penalty
    strain_penalty = calculate_strain_penalty(all_positions, all_symbols)
    
    return DescriptorSet(
        efield=tuple(efield),
        efield_proj=efield_proj,
        donor_count=donor_count,
        acceptor_count=acceptor_count,
        wire_length=wire_length,
        wire_bend_max=wire_bend_max,
        strain_penalty=strain_penalty
    )
