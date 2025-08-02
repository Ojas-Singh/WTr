"""
Geometric transforms for water molecules.
"""

import numpy as np
from typing import Tuple

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create rotation matrix from axis and angle using Rodrigues' formula.
    
    Args:
        axis: Unit vector representing rotation axis
        angle: Rotation angle in radians
    
    Returns:
        3x3 rotation matrix
    """
    axis = axis / np.linalg.norm(axis)  # Ensure unit vector
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    # Cross-product matrix
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    # Rodrigues' formula
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    
    return R

def rotate_water(atoms, water_index: int, axis: np.ndarray, angle: float):
    """
    Rotate a water molecule around given axis through its oxygen.
    
    Args:
        atoms: ASE Atoms object (modified in place)
        water_index: Index of oxygen atom of the water to rotate
        axis: Rotation axis (3D vector)
        angle: Rotation angle in radians
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Find oxygen position
    o_pos = positions[water_index]
    
    # Find hydrogen atoms belonging to this water
    h_indices = []
    for i, symbol in enumerate(symbols):
        if symbol == 'H':
            h_pos = positions[i]
            if np.linalg.norm(o_pos - h_pos) < 1.2:  # OH bond cutoff
                h_indices.append(i)
    
    # Create rotation matrix
    R = rotation_matrix(axis, angle)
    
    # Rotate hydrogen positions around oxygen
    for h_idx in h_indices:
        h_pos = positions[h_idx]
        # Vector from O to H
        oh_vec = h_pos - o_pos
        # Rotate the vector
        oh_vec_rot = R @ oh_vec
        # Update position
        positions[h_idx] = o_pos + oh_vec_rot
    
    atoms.set_positions(positions)

def flip_protons(atoms, water_index: int):
    """
    Flip protons of a water molecule across the lone-pair bisector plane.
    
    Args:
        atoms: ASE Atoms object (modified in place)
        water_index: Index of oxygen atom of the water to flip
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Find oxygen position
    o_pos = positions[water_index]
    
    # Find hydrogen atoms belonging to this water
    h_indices = []
    h_positions = []
    for i, symbol in enumerate(symbols):
        if symbol == 'H':
            h_pos = positions[i]
            if np.linalg.norm(o_pos - h_pos) < 1.2:  # OH bond cutoff
                h_indices.append(i)
                h_positions.append(h_pos)
    
    if len(h_positions) != 2:
        return  # Not a proper water molecule
    
    # Calculate bisector of HOH angle (approximate lone-pair direction)
    oh1 = h_positions[0] - o_pos
    oh2 = h_positions[1] - o_pos
    
    # Normalize OH vectors
    oh1_norm = oh1 / np.linalg.norm(oh1)
    oh2_norm = oh2 / np.linalg.norm(oh2)
    
    # Bisector direction (points between hydrogens)
    bisector = oh1_norm + oh2_norm
    bisector = bisector / np.linalg.norm(bisector)
    
    # Normal to the bisector plane (perpendicular to bisector and OH1xOH2)
    normal = np.cross(oh1_norm, oh2_norm)
    normal = normal / np.linalg.norm(normal)
    
    # Reflect OH vectors across the bisector plane
    oh_bond_length = 0.9572  # Standard OH bond length
    
    # New OH directions (reflected)
    oh1_reflected = oh1_norm - 2 * np.dot(oh1_norm, normal) * normal
    oh2_reflected = oh2_norm - 2 * np.dot(oh2_norm, normal) * normal
    
    # Update hydrogen positions
    positions[h_indices[0]] = o_pos + oh_bond_length * oh1_reflected
    positions[h_indices[1]] = o_pos + oh_bond_length * oh2_reflected
    
    atoms.set_positions(positions)

def rewire_hbond_pair(atoms, water_i: int, water_j: int):
    """
    Rotate two neighboring waters to exchange donor/acceptor roles.
    
    Args:
        atoms: ASE Atoms object (modified in place)
        water_i: Index of first water's oxygen
        water_j: Index of second water's oxygen
    """
    # Small random rotations to change H-bond geometry
    axis_i = np.random.normal(0, 1, 3)
    axis_i = axis_i / np.linalg.norm(axis_i)
    angle_i = np.random.normal(0, 0.2)  # Small angle
    
    axis_j = np.random.normal(0, 1, 3)
    axis_j = axis_j / np.linalg.norm(axis_j)
    angle_j = np.random.normal(0, 0.2)  # Small angle
    
    rotate_water(atoms, water_i, axis_i, angle_i)
    rotate_water(atoms, water_j, axis_j, angle_j)
