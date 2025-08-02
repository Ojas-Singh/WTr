"""
Surrogate scoring and Monte Carlo field sampling.
"""

import numpy as np
from typing import List, Dict, Any
import os
import random

from ..models.datatypes import Geometry, CalcSpec, DescriptorSet
from ..io.ase_helpers import geometry_to_atoms, atoms_to_geometry, set_calculator
from ..geom.transforms import rotate_water, flip_protons, rewire_hbond_pair
from ..calc.descriptors import compute_descriptors, get_model_charges

class MCMove:
    """Base class for Monte Carlo moves."""
    pass

class RotateWater(MCMove):
    """Rotate a water molecule."""
    def __init__(self, water_idx: int, sigma: float = 0.2):
        self.water_idx = water_idx
        self.sigma = sigma
    
    def apply(self, atoms):
        axis = np.random.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.normal(0, self.sigma)
        rotate_water(atoms, self.water_idx, axis, angle)

class FlipProtons(MCMove):
    """Flip protons of a water molecule."""
    def __init__(self, water_idx: int):
        self.water_idx = water_idx
    
    def apply(self, atoms):
        flip_protons(atoms, self.water_idx)

class RewirePair(MCMove):
    """Rewire H-bond between two waters."""
    def __init__(self, water_i: int, water_j: int):
        self.water_i = water_i
        self.water_j = water_j
    
    def apply(self, atoms):
        rewire_hbond_pair(atoms, self.water_i, self.water_j)

def calculate_surrogate_score(descriptors: DescriptorSet, 
                            weights: Dict[str, float] = None) -> float:
    """
    Calculate surrogate objective score (lower is better).
    
    Args:
        descriptors: Computed descriptor set
        weights: Weight parameters for scoring
    
    Returns:
        Surrogate score (dimensionless)
    """
    if weights is None:
        weights = {
            'w_E': 1.0,
            'w_D': 0.5, 
            'w_W': 0.7,
            'w_S': 0.3,
            'alpha_D': -1.0,
            'alpha_A': -0.5,
            'a_L': 1.0,
            'a_phi': 1.0
        }
    
    # Normalize descriptors (simplified - should use running statistics)
    # Guard against NaNs/Infs in descriptors by replacing with finite caps
    def safe(val, cap=1e3):
        try:
            if not np.isfinite(val):
                return cap
            return float(np.clip(val, -cap, cap))
        except Exception:
            return cap

    efield_norm = safe(descriptors.efield_proj / 0.1)  # Typical scale V/Å
    donor_norm = safe(descriptors.donor_count / 5.0)   # Typical count
    acceptor_norm = safe(descriptors.acceptor_count / 5.0)
    length_norm = safe(descriptors.wire_length / 10.0)  # Typical length Å
    bend_norm = safe(descriptors.wire_bend_max / 30.0)  # Typical angle degrees
    strain_norm = safe(descriptors.strain_penalty / 2.0)  # Typical strain
    
    # Calculate score components
    field_term = weights['w_E'] * (-efield_norm)  # Prefer large negative field
    donor_term = weights['w_D'] * (weights['alpha_D'] * donor_norm + 
                                 weights['alpha_A'] * acceptor_norm)
    wire_term = weights['w_W'] * (weights['a_L'] * length_norm + 
                                weights['a_phi'] * bend_norm)
    strain_term = weights['w_S'] * strain_norm
    
    score = field_term + donor_term + wire_term + strain_term
    # Final guard to avoid +/-inf or NaN propagating through MC
    if not np.isfinite(score):
        return 1e6
    return float(np.clip(score, -1e6, 1e6))

def generate_mc_moves(surface: Geometry, n_moves: int = 100) -> List[MCMove]:
    """
    Generate a list of random MC moves for the surface.
    
    Args:
        surface: Surface geometry
        n_moves: Number of moves to generate
    
    Returns:
        List of MC moves
    """
    moves = []
    
    # Find water oxygen indices
    o_indices = [i for i, symbol in enumerate(surface.symbols) if symbol == 'O']
    
    for _ in range(n_moves):
        move_type = random.choice(['rotate', 'flip', 'rewire'])
        
        if move_type == 'rotate':
            water_idx = random.choice(o_indices)
            moves.append(RotateWater(water_idx))
        
        elif move_type == 'flip':
            water_idx = random.choice(o_indices)
            moves.append(FlipProtons(water_idx))
        
        elif move_type == 'rewire' and len(o_indices) >= 2:
            water_i, water_j = random.sample(o_indices, 2)
            moves.append(RewirePair(water_i, water_j))
    
    return moves

def metropolis_accept(current_score: float, proposed_score: float, 
                     temperature: float) -> bool:
    """
    Metropolis acceptance criterion with simulated annealing.
    
    Args:
        current_score: Current objective value
        proposed_score: Proposed objective value
        temperature: Annealing temperature
    
    Returns:
        True if move should be accepted
    """
    if proposed_score < current_score:
        return True
    
    if temperature <= 0:
        return False
    
    delta = proposed_score - current_score
    prob = np.exp(-delta / temperature)
    return random.random() < prob

def mc_field_sampler(surface: Geometry, rc: Geometry, reaction_axis: np.ndarray,
                    temps_sa: List[float], moves: List[MCMove], steps: int,
                    checkpoint_every: int, top_k: int, calc_spec: CalcSpec,
                    workdir: str) -> List[Geometry]:
    """
    Monte Carlo field sampling with simulated annealing.
    
    Args:
        surface: Initial surface geometry
        rc: Reactant complex geometry
        reaction_axis: Reaction axis vector
        temps_sa: Annealing temperature schedule
        moves: List of MC moves to choose from
        steps: Number of MC steps
        checkpoint_every: Frequency of ASE optimization checkpoints
        top_k: Number of top surfaces to keep
        calc_spec: Calculator specification
        workdir: Working directory
    
    Returns:
        List of top-scoring surface geometries
    """
    os.makedirs(workdir, exist_ok=True)
    
    # Initialize
    current_atoms = geometry_to_atoms(surface)
    rc_atoms = geometry_to_atoms(rc)
    
    # Get charges for descriptor calculation
    all_symbols = current_atoms.get_chemical_symbols() + rc_atoms.get_chemical_symbols()
    charges = get_model_charges(all_symbols)
    
    # Calculate initial descriptors and score
    reactive_point = np.mean(rc_atoms.get_positions(), axis=0)  # Simplified
    current_descriptors = compute_descriptors(current_atoms, rc_atoms, 
                                            reaction_axis, reactive_point, charges)
    current_score = calculate_surrogate_score(current_descriptors)
    
    # Track best configurations
    best_configs = [(current_score, atoms_to_geometry(current_atoms, f"Initial"))]
    
    accepted_moves = 0
    
    for step in range(steps):
        # Select temperature for this step
        temp_idx = min(step * len(temps_sa) // steps, len(temps_sa) - 1)
        temperature = temps_sa[temp_idx]
        
        # Select and apply random move
        move = random.choice(moves)
        
        # Make a copy for trial move
        trial_atoms = current_atoms.copy()
        move.apply(trial_atoms)
        
        # Calculate new descriptors and score with robust guards
        try:
            trial_descriptors = compute_descriptors(trial_atoms, rc_atoms,
                                                    reaction_axis, reactive_point, charges)
            trial_score = calculate_surrogate_score(trial_descriptors)
            if not np.isfinite(trial_score):
                trial_score = 1e6
        except Exception as e:
            # If descriptor calculation fails, penalize heavily but keep MC running
            print(f"Warning: Descriptor/score failed at step {step}: {e}")
            trial_score = 1e6
        
        # Accept or reject
        if metropolis_accept(current_score, trial_score, temperature):
            current_atoms = trial_atoms
            current_score = trial_score
            current_descriptors = trial_descriptors
            accepted_moves += 1
            
            # Update best configurations
            best_configs.append((current_score, atoms_to_geometry(current_atoms, f"Step {step}")))
            best_configs.sort(key=lambda x: x[0])
            best_configs = best_configs[:top_k]
        
        # Periodic optimization checkpoint
        if (step + 1) % checkpoint_every == 0:
            # More robust logging of score without forcing format on inf/NaN
            score_str = f"{current_score:.3f}" if np.isfinite(current_score) else str(current_score)
            print(f"MC step {step + 1}/{steps}, T={temperature:.3f}, "
                  f"score={score_str}, accepted={accepted_moves}/{step+1}")

            # Quick ASE optimization (simplified - would need proper implementation)
            try:
                set_calculator(current_atoms, calc_spec)
                # In real implementation, would run BFGS optimization here
                # current_atoms = optimize_with_constraints(current_atoms, core_indices)
            except Exception as e:
                print(f"Warning: Optimization failed at step {step}: {e}")
    
    # Return top geometries
    return [config[1] for config in best_configs]
