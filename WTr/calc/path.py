"""
Reaction path calculations using NEB and Dimer methods.
"""

import numpy as np
import os

from ..models.datatypes import CalcSpec
from ..io.ase_helpers import set_calculator

def linear_interpolate_atoms(atoms1, atoms2, n_images: int):
    """
    Create linear interpolation between two atomic configurations.
    
    Args:
        atoms1: Initial ASE Atoms object
        atoms2: Final ASE Atoms object
        n_images: Number of intermediate images
    
    Returns:
        List of interpolated Atoms objects
    """
    pos1 = atoms1.get_positions()
    pos2 = atoms2.get_positions()
    
    images = []
    for i in range(n_images):
        alpha = i / (n_images - 1)
        interp_pos = (1 - alpha) * pos1 + alpha * pos2
        
        image = atoms1.copy()
        image.set_positions(interp_pos)
        images.append(image)
    
    return images

def idpp_interpolation(atoms1, atoms2, n_images: int):
    """
    Image-dependent pair potential (IDPP) interpolation.
    Simplified implementation - in real code would use ASE's IDPP.
    
    Args:
        atoms1: Initial configuration
        atoms2: Final configuration  
        n_images: Number of images
    
    Returns:
        List of IDPP-optimized images
    """
    # For now, just use linear interpolation
    # Real implementation would use ase.neb.idpp_interpolate
    return linear_interpolate_atoms(atoms1, atoms2, n_images)

def run_neb_optimization(images, calc_spec: CalcSpec, fmax: float = 0.05, 
                        steps: int = 100, climbing: bool = True):
    """
    Run NEB optimization on image chain.
    
    Args:
        images: List of ASE Atoms objects
        calc_spec: Calculator specification
        fmax: Force convergence criterion
        steps: Maximum optimization steps
        climbing: Whether to use climbing image
    
    Returns:
        Optimized images and energies
    """
    # Set calculators on all images
    for image in images:
        set_calculator(image, calc_spec)
    
    # Try real ASE NEB with climbing image; fall back to energy evaluation if unavailable
    try:
        from ase.neb import NEB
        from ase.optimize import FIRE

        neb = NEB(images, climb=climbing)
        opt = FIRE(neb, logfile='neb.log')
        opt.run(fmax=fmax, steps=steps)

        energies = []
        for image in images:
            try:
                energies.append(image.get_potential_energy())
            except Exception as e:
                print(f"Warning: Energy calculation failed after NEB: {e}")
                energies.append(np.inf)
        return images, energies
    except Exception as e:
        print(f"Warning: Real NEB failed or unavailable, using placeholder energies: {e}")
        energies = []
        for image in images:
            try:
                energy = image.get_potential_energy()
                # Guard against non-finite energies
                if not np.isfinite(energy):
                    energy = np.inf
                energies.append(energy)
            except Exception as e2:
                print(f"Warning: Energy calculation failed: {e2}")
                energies.append(np.inf)
        return images, energies

def find_ts_image(images, energies):
    """
    Find the transition state image (highest energy).
    
    Args:
        images: List of optimized NEB images
        energies: Corresponding energies
    
    Returns:
        TS image (ASE Atoms object)
    """
    if not energies:
        return images[len(images)//2]  # Middle image as fallback
    
    ts_idx = np.argmax(energies)
    return images[ts_idx]

def dimer_ts_refinement(ts_guess, calc_spec: CalcSpec, fmax: float = 0.01):
    """
    Refine TS using Dimer method.
    
    Args:
        ts_guess: Initial TS guess (ASE Atoms)
        calc_spec: Calculator specification
        fmax: Force convergence criterion
    
    Returns:
        Refined TS structure
    """
    # Set calculator
    set_calculator(ts_guess, calc_spec)
    
    # Attempt a real dimer refinement; fall back to returning guess
    try:
        from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate

        d_control = DimerControl(initial_eigenmode_method='displacement',
                                 displacement_method='vector',
                                 displacement=0.02,  # Å
                                 frot=0.02)
        d_atoms = MinModeAtoms(ts_guess, d_control)
        d_opt = MinModeTranslate(d_atoms, logfile='dimer.log')
        d_opt.run(fmax=fmax)
        return d_atoms.atoms
    except Exception as e:
        print(f"Warning: Dimer TS refinement failed or unavailable, returning TS guess: {e}")
        return ts_guess

def verify_ts_connectivity(ts_atoms, rc_atoms, p_atoms, displacement: float = 0.1):
    """
    Verify TS connects RC and P by displacing along imaginary mode.
    
    Args:
        ts_atoms: TS structure
        rc_atoms: Reactant complex
        p_atoms: Product complex
        displacement: Displacement along imaginary mode
    
    Returns:
        (connects_rc, connects_p) - boolean flags
    """
    # Simplified implementation - would need vibrational analysis
    # to get imaginary mode vector
    
    # Basic sanity: TS should be higher in energy than RC and P
    try:
        e_ts = ts_atoms.get_potential_energy()
        e_rc = rc_atoms.get_potential_energy()
        e_p = p_atoms.get_potential_energy()
        if not (np.isfinite(e_ts) and np.isfinite(e_rc) and np.isfinite(e_p)):
            return False, False
        connects_rc = e_ts > e_rc
        connects_p = e_ts > e_p
        return connects_rc, connects_p
    except Exception as e:
        print(f"Warning: TS connectivity check failed: {e}")
        return False, False

def neb_ts(rc_atoms, p_atoms, calc_spec: CalcSpec, n_images: int = 9, 
          fmax: float = 0.05, workdir: str = "."):
    """
    Find transition state using NEB + Dimer refinement.
    
    Args:
        rc_atoms: Reactant complex (ASE Atoms)
        p_atoms: Product complex (ASE Atoms)
        calc_spec: Calculator specification
        n_images: Number of NEB images
        fmax: Force convergence criterion
        workdir: Working directory
    
    Returns:
        (ts_atoms, images, energies) - TS structure and NEB data
    """
    os.makedirs(workdir, exist_ok=True)
    
    print(f"Running NEB calculation with {n_images} images")
    
    # Create initial interpolation
    initial_images = linear_interpolate_atoms(rc_atoms, p_atoms, n_images)

    # IDPP pre-optimization
    print("IDPP pre-optimization...")
    idpp_images = idpp_interpolation(rc_atoms, p_atoms, n_images)

    # NEB optimization with climbing image enabled
    print("NEB optimization...")
    optimized_images, energies = run_neb_optimization(idpp_images, calc_spec, fmax=fmax, steps=1000, climbing=True)
    
    # Find TS image
    ts_image = find_ts_image(optimized_images, energies)
    
    # Dimer refinement
    print("Dimer TS refinement...")
    ts_refined = dimer_ts_refinement(ts_image, calc_spec, fmax=0.01)
    
    # Verify connectivity
    connects_rc, connects_p = verify_ts_connectivity(ts_refined, rc_atoms, p_atoms)
    
    if not (connects_rc and connects_p):
        print("Warning: TS may not properly connect RC and P")
    
    # Save results
    from ..io.ase_helpers import atoms_to_geometry
    from ..io.geometry import save_xyz
    
    ts_geometry = atoms_to_geometry(ts_refined, "Refined TS")
    ts_path = os.path.join(workdir, "ts_refined.xyz")
    save_xyz(ts_geometry, ts_path)
    
    print(f"TS saved to {ts_path}")
    # Barrier relative to RC (image 0). Guard for non-finite energies.
    try:
        e0 = energies[0] if energies and np.isfinite(energies[0]) else np.nan
        emax = np.nanmax(energies) if len(energies) else np.nan
        barrier = emax - e0 if np.isfinite(e0) and np.isfinite(emax) else np.nan
        barrier_str = f"{barrier:.3f}" if np.isfinite(barrier) else "nan"
    except Exception:
        barrier_str = "nan"
    print(f"Barrier height: {barrier_str} eV")
    
    return ts_refined, optimized_images, energies
