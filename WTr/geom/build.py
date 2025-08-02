"""
Water cluster building functionality.
"""

import numpy as np
from typing import List, Tuple
import random

from ..models.datatypes import Geometry, SurfaceSpec

def poisson_disc_sampling(radius: float, min_distance: float, n_attempts: int = 30) -> List[Tuple[float, float, float]]:
    """
    Generate points using Poisson disc sampling within a sphere.
    
    Args:
        radius: Sphere radius
        min_distance: Minimum distance between points
        n_attempts: Number of attempts to place each point
    
    Returns:
        List of 3D coordinates
    """
    points = []
    
    # Start with a random point in the sphere
    while len(points) == 0:
        # Generate random point in sphere
        u = np.random.uniform(0, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        r = radius * (u ** (1/3))  # Uniform distribution in sphere
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        points.append((x, y, z))
    
    # Generate additional points
    active_list = [0]
    
    while active_list:
        # Pick random point from active list
        idx = random.choice(active_list)
        point = points[idx]
        
        found_new = False
        
        for _ in range(n_attempts):
            # Generate candidate point in annulus around current point
            angle = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            distance = np.random.uniform(min_distance, 2 * min_distance)
            
            dx = distance * np.sin(phi) * np.cos(angle)
            dy = distance * np.sin(phi) * np.sin(angle)
            dz = distance * np.cos(phi)
            
            candidate = (point[0] + dx, point[1] + dy, point[2] + dz)
            
            # Check if candidate is within sphere
            if np.linalg.norm(candidate) > radius:
                continue
            
            # Check minimum distance to all existing points
            valid = True
            for existing in points:
                if np.linalg.norm(np.array(candidate) - np.array(existing)) < min_distance:
                    valid = False
                    break
            
            if valid:
                points.append(candidate)
                active_list.append(len(points) - 1)
                found_new = True
                break
        
        if not found_new:
            active_list.remove(idx)
    
    return points

def create_water_molecule(center: Tuple[float, float, float]) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    """
    Create a water molecule at given center with random orientation.
    
    Returns:
        symbols, coordinates for O, H, H
    """
    # Water geometry parameters
    r_OH = 0.9572  # Å
    angle_HOH = 104.5 * np.pi / 180  # radians
    
    # Random orientation
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    psi = np.random.uniform(0, 2 * np.pi)
    
    # H positions relative to O in molecular frame
    h1_local = np.array([r_OH * np.sin(angle_HOH/2), 0, r_OH * np.cos(angle_HOH/2)])
    h2_local = np.array([-r_OH * np.sin(angle_HOH/2), 0, r_OH * np.cos(angle_HOH/2)])
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])
    
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    # Apply rotation and translation
    o_pos = np.array(center)
    h1_pos = o_pos + R @ h1_local
    h2_pos = o_pos + R @ h2_local
    
    symbols = ['O', 'H', 'H']
    coords = [tuple(o_pos), tuple(h1_pos), tuple(h2_pos)]
    
    return symbols, coords

def toy_hbond_energy(coords: np.ndarray, symbols: List[str]) -> float:
    """
    Simple toy potential for H-bond interactions to remove clashes.
    """
    energy = 0.0
    
    # Find O and H indices
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    
    # O-O repulsion
    for i in range(len(o_indices)):
        for j in range(i + 1, len(o_indices)):
            r = np.linalg.norm(coords[o_indices[i]] - coords[o_indices[j]])
            if r < 4.0:  # Only for close contacts
                sigma = 2.8  # Å
                energy += 0.1 * (sigma / r) ** 12
    
    # O-H attraction (simplified)
    for o_idx in o_indices:
        for h_idx in h_indices:
            # Skip if H belongs to this O (crude check by distance)
            o_pos = coords[o_idx]
            h_pos = coords[h_idx]
            r = np.linalg.norm(o_pos - h_pos)
            
            if 1.5 < r < 3.5:  # H-bond range
                energy -= 0.05 * np.exp(-(r - 2.8)**2 / 0.2)
    
    return energy

def build_water_cluster(spec: SurfaceSpec) -> Geometry:
    """
    Build an N-water cluster using Poisson disc sampling and random orientations.
    """
    np.random.seed(spec.random_seed)
    random.seed(spec.random_seed)
    
    # Generate O positions
    min_distance = 2.6  # Minimum O-O distance
    oxygen_positions = poisson_disc_sampling(spec.radius, min_distance)
    
    # Limit to requested number of waters
    if len(oxygen_positions) > spec.waters_n:
        oxygen_positions = oxygen_positions[:spec.waters_n]
    elif len(oxygen_positions) < spec.waters_n:
        # Need more points - add some with relaxed constraints
        while len(oxygen_positions) < spec.waters_n:
            # Try placing in shell
            u = np.random.uniform(0.7, 1.0)  # Prefer outer shell
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            r = spec.radius * u
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            candidate = (x, y, z)
            
            # Check minimum distance (relaxed)
            valid = True
            for existing in oxygen_positions:
                if np.linalg.norm(np.array(candidate) - np.array(existing)) < min_distance * 0.9:
                    valid = False
                    break
            
            if valid:
                oxygen_positions.append(candidate)
    
    # Create water molecules
    all_symbols = []
    all_coords = []
    
    for o_pos in oxygen_positions:
        symbols, coords = create_water_molecule(o_pos)
        all_symbols.extend(symbols)
        all_coords.extend(coords)
    
    # Quick relaxation with toy potential (simplified)
    coords_array = np.array(all_coords)
    
    # Simple steepest descent for a few steps
    for step in range(5):
        forces = np.zeros_like(coords_array)
        
        # Calculate simple forces (gradient of toy potential)
        for i in range(len(all_symbols)):
            if all_symbols[i] == 'O':
                for j in range(len(all_symbols)):
                    if i != j and all_symbols[j] == 'O':
                        r_vec = coords_array[i] - coords_array[j]
                        r = np.linalg.norm(r_vec)
                        if r < 4.0 and r > 0.1:
                            # Repulsive force
                            f_mag = 12 * 0.1 * (2.8 / r) ** 12 / r
                            forces[i] += f_mag * r_vec / r
        
        # Update positions
        coords_array += 0.01 * forces  # Small step
    
    # Convert back to coordinate tuples
    final_coords = [tuple(coord) for coord in coords_array]
    
    return Geometry(
        symbols=all_symbols,
        coords=final_coords,
        comment=f"Water cluster: {len(oxygen_positions)} molecules, R={spec.radius} Å"
    )
