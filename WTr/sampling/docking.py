"""
Docking reactant complex to water surface.
"""

import numpy as np
from typing import List, Tuple
import os

from ..models.datatypes import Geometry, CalcSpec
from ..io.ase_helpers import geometry_to_atoms, atoms_to_geometry, set_calculator
from ..io.geometry import merge

def generate_surface_points(cluster: Geometry, n_points: int = 20, 
                          radius_offset: float = 1.5) -> List[Tuple[float, float, float]]:
    """
    Generate candidate anchor points on the cluster surface.
    
    Args:
        cluster: Water cluster geometry
        n_points: Number of anchor points to generate
        radius_offset: Distance from cluster surface
    
    Returns:
        List of 3D anchor points
    """
    coords = np.array(cluster.coords)
    
    # Find cluster bounds
    center = np.mean(coords, axis=0)
    max_radius = np.max(np.linalg.norm(coords - center, axis=1))
    
    # Generate points on sphere around cluster
    points = []
    for _ in range(n_points):
        # Random point on unit sphere
        u = np.random.normal(0, 1, 3)
        u = u / np.linalg.norm(u)
        
        # Place at cluster surface + offset
        point = center + (max_radius + radius_offset) * u
        points.append(tuple(point))
    
    return points

def calculate_surface_normal(cluster_coords: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Calculate approximate surface normal at a point.
    
    Args:
        cluster_coords: Coordinates of cluster atoms
        point: Point where to calculate normal
    
    Returns:
        Unit normal vector pointing outward
    """
    # Simple approximation: direction from cluster center to point
    center = np.mean(cluster_coords, axis=0)
    normal = point - center
    return normal / np.linalg.norm(normal)

def check_steric_clash(surface_coords: np.ndarray, rc_coords: np.ndarray, 
                      min_distance: float = 2.0) -> bool:
    """
    Check for steric clashes between surface and reactant complex.
    
    Args:
        surface_coords: Surface atom coordinates
        rc_coords: Reactant complex coordinates
        min_distance: Minimum allowed distance
    
    Returns:
        True if clash detected
    """
    for rc_pos in rc_coords:
        for surf_pos in surface_coords:
            if np.linalg.norm(rc_pos - surf_pos) < min_distance:
                return True
    return False

def align_to_normal(rc: Geometry, reaction_axis: Tuple[int, int], 
                   target_normal: np.ndarray) -> Geometry:
    """
    Align reactant complex so reaction axis points along target normal.
    
    Args:
        rc: Reactant complex geometry
        reaction_axis: Indices of atoms defining reaction axis
        target_normal: Target direction for reaction axis
    
    Returns:
        Aligned geometry
    """
    coords = np.array(rc.coords)
    
    # Calculate current reaction axis direction
    axis_vec = coords[reaction_axis[1]] - coords[reaction_axis[0]]
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    
    # Calculate rotation to align with target normal
    target_normal = target_normal / np.linalg.norm(target_normal)
    
    # Rotation axis (cross product)
    rot_axis = np.cross(axis_vec, target_normal)
    rot_axis_norm = np.linalg.norm(rot_axis)
    
    if rot_axis_norm < 1e-6:  # Already aligned or anti-aligned
        if np.dot(axis_vec, target_normal) < 0:
            # Anti-aligned, rotate 180 degrees around any perpendicular axis
            perp = np.array([1, 0, 0])
            if abs(np.dot(axis_vec, perp)) > 0.9:
                perp = np.array([0, 1, 0])
            rot_axis = np.cross(axis_vec, perp)
            angle = np.pi
        else:
            # Already aligned
            angle = 0
    else:
        rot_axis = rot_axis / rot_axis_norm
        angle = np.arccos(np.clip(np.dot(axis_vec, target_normal), -1, 1))
    
    # Apply rotation (simplified - using Rodrigues' formula)
    if angle > 1e-6:
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                      [rot_axis[2], 0, -rot_axis[0]],
                      [-rot_axis[1], rot_axis[0], 0]])
        
        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
        
        # Rotate around center of mass
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        coords_rotated = (R @ coords_centered.T).T + center
    else:
        coords_rotated = coords
    
    new_coords = [tuple(coord) for coord in coords_rotated]
    
    return Geometry(
        symbols=rc.symbols,
        coords=new_coords,
        comment=f"Aligned RC: {rc.comment}"
    )

def dock_reactant(surface: Geometry, rc: Geometry, axis: Tuple[int, int],
                  ntries: int, calc_spec: CalcSpec, workdir: str) -> List[Geometry]:
    """
    Dock reactant complex to surface using heuristic placement.
    
    Args:
        surface: Water surface geometry
        rc: Reactant complex geometry
        axis: Reaction axis atom indices
        ntries: Number of docking attempts
        calc_spec: Calculator specification
        workdir: Working directory for calculations
    
    Returns:
        List of RC-on-surface candidates (merged geometries)
    """
    os.makedirs(workdir, exist_ok=True)
    
    surface_coords = np.array(surface.coords)
    candidates = []
    
    # Generate anchor points
    anchor_points = generate_surface_points(surface, ntries)
    
    for i, anchor in enumerate(anchor_points):
        try:
            # Calculate surface normal at anchor point
            normal = calculate_surface_normal(surface_coords, np.array(anchor))
            
            # Align RC to surface normal
            aligned_rc = align_to_normal(rc, axis, normal)
            
            # Translate RC to anchor point
            rc_coords = np.array(aligned_rc.coords)
            rc_center = np.mean(rc_coords, axis=0)
            translation = np.array(anchor) - rc_center
            
            translated_coords = rc_coords + translation
            placed_rc = Geometry(
                symbols=aligned_rc.symbols,
                coords=[tuple(coord) for coord in translated_coords],
                comment=f"Placed RC {i}"
            )
            
            # Check for steric clashes
            if check_steric_clash(surface_coords, translated_coords):
                continue
            
            # Merge surface and RC
            merged = merge(surface, placed_rc)
            
            # Quick local optimization (simplified)
            atoms = geometry_to_atoms(merged)
            set_calculator(atoms, calc_spec)
            
            # Save candidate
            candidate_path = os.path.join(workdir, f"candidate_{i}.xyz")
            candidate_geom = atoms_to_geometry(atoms, f"Docked candidate {i}")
            
            from ..io.geometry import save_xyz
            save_xyz(candidate_geom, candidate_path)
            
            candidates.append(candidate_geom)
            
            if len(candidates) >= 10:  # Limit number of candidates
                break
                
        except Exception as e:
            print(f"Warning: Docking attempt {i} failed: {e}")
            continue
    
    return candidates
