"""
Main pipeline orchestration for surface evaluation and search.
"""

import os
import json
import numpy as np
from typing import List

from ..models.datatypes import (
    Geometry, ReactionSpec, SurfaceSpec, CalcSpec, EvalResult, DescriptorSet
)
from ..geom.build import build_water_cluster
from ..geom.constraints import select_core_atoms
from ..sampling.docking import dock_reactant
from ..sampling.ts_template import ts_templated_surface
from ..sampling.mc_field import mc_field_sampler, generate_mc_moves
from ..calc.path import neb_ts
from ..calc.vibthermo import vibrational_analysis
from ..calc.kinetics import calculate_rate_constants, print_kinetics_summary
from ..calc.descriptors import compute_descriptors, get_model_charges
from ..io.ase_helpers import geometry_to_atoms, atoms_to_geometry, set_calculator
from ..io.geometry import save_xyz

def evaluate_surface(surface: Geometry, reaction: ReactionSpec, calc_spec: CalcSpec,
                    core_indices: List[int], temps: List[float], workdir: str) -> EvalResult:
    """
    Evaluate a single surface configuration for the given reaction.
    
    Args:
        surface: Water surface geometry
        reaction: Reaction specification
        calc_spec: Calculator specification
        core_indices: Indices of core atoms to constrain
        temps: List of temperatures for evaluation
        workdir: Working directory for calculations
    
    Returns:
        EvalResult with computed properties
    """
    os.makedirs(workdir, exist_ok=True)
    
    surface_id = os.path.basename(workdir)
    
    print(f"\n{'='*60}")
    print(f"EVALUATING SURFACE: {surface_id}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Dock reactant complex to surface
        print("Step 1: Docking reactant complex...")
        rc_candidates = dock_reactant(
            surface, reaction.reactant, reaction.reaction_axis_atoms,
            ntries=10, calc_spec=calc_spec, workdir=os.path.join(workdir, "docking")
        )
        
        if not rc_candidates:
            raise RuntimeError("No successful docking configurations found")
        
        # Use best docking candidate (first one for now)
        rc_on_surface = rc_candidates[0]
        
        # Convert to ASE atoms and optimize
        rc_atoms = geometry_to_atoms(rc_on_surface)
        
        # Apply constraints for core atoms
        from ..geom.constraints import make_constraints
        constraints = make_constraints(rc_atoms, core_indices, k=None)
        rc_atoms.set_constraint(constraints)
        
        # Local optimization (simplified)
        set_calculator(rc_atoms, calc_spec)
        print("Optimizing RC-on-surface...")
        
        # Save optimized RC
        rc_optimized = atoms_to_geometry(rc_atoms, f"Optimized RC on {surface_id}")
        rc_path = os.path.join(workdir, "rc_optimized.xyz")
        save_xyz(rc_optimized, rc_path)
        
        # Step 2: Generate product configuration
        print("Step 2: Generating product configuration...")
        # Build product-on-surface with identical atom count/order as RC-on-surface.
        # Start from RC atoms and replace the adsorbate/reactant-complex portion with the product fragment.
        p_fragment_atoms = geometry_to_atoms(reaction.product)
        set_calculator(p_fragment_atoms, calc_spec)

        # Split RC atoms into surface and adsorbate parts
        n_surface_atoms = len(surface.symbols)
        rc_adsorbate_atoms = rc_atoms[n_surface_atoms:]

        # Sanity check: product fragment must match the size of the adsorbate portion
        if len(p_fragment_atoms) != len(rc_adsorbate_atoms):
            raise ValueError(
                f"Product fragment atom count ({len(p_fragment_atoms)}) does not match "
                f"reactant-complex fragment on surface ({len(rc_adsorbate_atoms)}). "
                "Ensure product.xyz contains only the reactive complex with the same atom ordering."
            )

        # Construct full product-on-surface Atoms by copying RC and replacing adsorbate positions
        p_atoms = rc_atoms.copy()
        p_positions = p_atoms.get_positions()
        p_positions[n_surface_atoms:] = p_fragment_atoms.get_positions()
        p_atoms.set_positions(p_positions)
        set_calculator(p_atoms, calc_spec)

        # Save the full product-on-surface structure
        p_geometry = atoms_to_geometry(p_atoms, f"Product on {surface_id}")
        p_path = os.path.join(workdir, "product.xyz")
        save_xyz(p_geometry, p_path)
        
        # Step 3: Find transition state using NEB
        print("Step 3: Finding transition state...")
        ts_atoms, neb_images, neb_energies = neb_ts(
            rc_atoms, p_atoms, calc_spec, n_images=9, fmax=0.05,
            workdir=os.path.join(workdir, "neb")
        )
        
        ts_geometry = atoms_to_geometry(ts_atoms, f"TS for {surface_id}")
        ts_path = os.path.join(workdir, "ts_optimized.xyz")
        save_xyz(ts_geometry, ts_path)
        
        # Step 4: Vibrational analysis and thermochemistry
        print("Step 4: Vibrational analysis...")
        vib_results = vibrational_analysis(
            rc_atoms, ts_atoms, p_atoms, calc_spec, temps, nu_cutoff_cm1=100.0
        )
        
        # Step 5: Kinetics with tunneling
        print("Step 5: Calculating kinetics...")
        imaginary_freq = vib_results['ts'].get('imaginary_freq', 0.0)
        
        rate_data = calculate_rate_constants(
            vib_results['barriers'], temps, imaginary_freq, method="wigner"
        )
        
        # Print summary
        print_kinetics_summary(rate_data, vib_results['barriers'])
        
        # Step 6: Compute descriptors
        print("Step 6: Computing descriptors...")
        
        # Split atoms back to surface and RC parts
        n_surface_atoms = len(surface.symbols)
        surface_atoms_only = rc_atoms[:n_surface_atoms]
        rc_atoms_only = rc_atoms[n_surface_atoms:]
        
        # Get charges and reaction axis
        all_symbols = rc_atoms.get_chemical_symbols()
        charges = get_model_charges(all_symbols)
        
        # Reaction axis vector
        rc_positions = rc_atoms_only.get_positions()
        if len(rc_positions) >= 2:
            axis_vec = rc_positions[1] - rc_positions[0]  # Simplified
            axis_vec = axis_vec / max(np.linalg.norm(axis_vec), 1e-6)
        else:
            axis_vec = np.array([1, 0, 0])
        
        reactive_point = np.mean(rc_positions, axis=0)
        
        descriptors = compute_descriptors(
            surface_atoms_only, rc_atoms_only, axis_vec, reactive_point, charges
        )
        
        # Step 7: Create result object
        deltaE_dagger = vib_results['barriers']['deltaE']
        
        # Get temperature-specific values
        deltaG_10K = vib_results['barriers']['deltaG'].get(10.0, deltaE_dagger)
        deltaG_20K = vib_results['barriers']['deltaG'].get(20.0, deltaE_dagger)
        
        rate_10K = rate_data['rates'].get(10.0, 0.0)
        rate_20K = rate_data['rates'].get(20.0, 0.0)
        
        result = EvalResult(
            surface_id=surface_id,
            deltaE_dagger=deltaE_dagger,
            deltaG_dagger_10K=deltaG_10K,
            deltaG_dagger_20K=deltaG_20K,
            rate_10K=rate_10K,
            rate_20K=rate_20K,
            descriptors=descriptors,
            paths={
                'rc': rc_path,
                'ts': ts_path,
                'p': p_path,
                'workdir': workdir
            }
        )
        
        # Save result as JSON
        result_path = os.path.join(workdir, "result.json")
        with open(result_path, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        
        print(f"Evaluation complete. Results saved to {result_path}")
        
        return result
        
    except Exception as e:
        print(f"Error evaluating surface {surface_id}: {e}")
        
        # Return dummy result with high barrier
        dummy_descriptors = DescriptorSet(
            efield=(0.0, 0.0, 0.0), efield_proj=0.0,
            donor_count=0, acceptor_count=0,
            wire_length=float('inf'), wire_bend_max=180.0,
            strain_penalty=100.0
        )
        
        return EvalResult(
            surface_id=surface_id,
            deltaE_dagger=10.0,  # High barrier
            deltaG_dagger_10K=10.0,
            deltaG_dagger_20K=10.0,
            rate_10K=0.0,
            rate_20K=0.0,
            descriptors=dummy_descriptors,
            paths={'workdir': workdir}
        )

def search_best_surfaces(reaction: ReactionSpec, surface_spec: SurfaceSpec,
                        calc_spec: CalcSpec, temps: List[float],
                        n_ts_templates: int, n_mc_rounds: int, max_evals: int,
                        workdir: str) -> List[EvalResult]:
    """
    Search for the best water surface configurations for a reaction.
    
    Args:
        reaction: Reaction specification
        surface_spec: Surface specification
        calc_spec: Calculator specification
        temps: Temperatures for evaluation
        n_ts_templates: Number of TS-templated surfaces to generate
        n_mc_rounds: Number of MC refinement rounds per template
        max_evals: Maximum number of surfaces to evaluate fully
        workdir: Working directory
    
    Returns:
        List of EvalResult objects, sorted by barrier height
    """
    os.makedirs(workdir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"WATER SURFACE SEARCH FOR REACTION: {reaction.name}")
    print(f"{'='*80}")
    
    # Step A: Build base water cluster
    print("Step A: Building base water cluster...")
    base_cluster = build_water_cluster(surface_spec)
    
    cluster_path = os.path.join(workdir, "base_cluster.xyz")
    save_xyz(base_cluster, cluster_path)
    
    core_indices = select_core_atoms(base_cluster, surface_spec)
    print(f"Selected {len(core_indices)} core atoms from {len(base_cluster.symbols)} total")
    
    # Step B: Generate TS-templated surfaces
    print(f"Step B: Generating {n_ts_templates} TS-templated surfaces...")
    
    ts_surfaces = []
    
    if reaction.ts_seed is not None:
        for i in range(n_ts_templates):
            print(f"  Generating TS-templated surface {i+1}/{n_ts_templates}")
            
            try:
                ts_surface = ts_templated_surface(
                    base_cluster, reaction.ts_seed, calc_spec, core_indices,
                    k_internal=0.5, workdir=os.path.join(workdir, f"ts_template_{i}")
                )
                ts_surfaces.append(ts_surface)
                
            except Exception as e:
                print(f"    Warning: TS template {i} failed: {e}")
                # Use base cluster as fallback
                ts_surfaces.append(base_cluster)
    else:
        print("  No TS seed provided, using base cluster only")
        ts_surfaces = [base_cluster] * min(n_ts_templates, 5)
    
    # Step C: MC field sampling refinement
    print(f"Step C: MC refinement with {n_mc_rounds} rounds per surface...")
    
    refined_surfaces = []
    
    for i, ts_surface in enumerate(ts_surfaces[:10]):  # Limit to avoid too many
        print(f"  MC refinement of surface {i+1}")
        
        try:
            # Generate MC moves
            moves = generate_mc_moves(ts_surface, n_moves=200)
            
            # Run MC sampling
            reaction_axis = np.array([1, 0, 0])  # Simplified
            
            mc_results = mc_field_sampler(
                ts_surface, reaction.reactant, reaction_axis,
                temps_sa=[1.0, 0.9, 0.8, 0.7], moves=moves, steps=n_mc_rounds,
                checkpoint_every=50, top_k=5, calc_spec=calc_spec,
                workdir=os.path.join(workdir, f"mc_refine_{i}")
            )
            
            refined_surfaces.extend(mc_results)
            
        except Exception as e:
            print(f"    Warning: MC refinement {i} failed: {e}")
            refined_surfaces.append(ts_surface)
    
    # Limit number of candidates
    candidates = refined_surfaces[:max_evals]
    print(f"Selected {len(candidates)} candidates for full evaluation")
    
    # Step D: Full evaluation of candidates
    print(f"Step D: Full evaluation of {len(candidates)} candidates...")
    
    results = []
    
    for i, candidate in enumerate(candidates):
        print(f"\nEvaluating candidate {i+1}/{len(candidates)}")
        
        eval_workdir = os.path.join(workdir, f"eval_{i:03d}")
        
        try:
            result = evaluate_surface(
                candidate, reaction, calc_spec, core_indices, temps, eval_workdir
            )
            results.append(result)
            
        except Exception as e:
            print(f"Error evaluating candidate {i}: {e}")
            continue
    
    # Step E: Rank and return results
    print("Step E: Ranking results...")
    
    # Sort by 10K barrier height (primary criterion)
    results.sort(key=lambda r: r.deltaG_dagger_10K)
    
    # Save summary
    summary_path = os.path.join(workdir, "search_summary.json")
    summary_data = {
        'reaction': reaction.dict(),
        'surface_spec': surface_spec.dict(),
        'n_evaluated': len(results),
        'best_barriers': [r.deltaG_dagger_10K for r in results[:10]]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSearch complete! Evaluated {len(results)} surfaces.")
    print(f"Best barrier: {results[0].deltaG_dagger_10K:.3f} eV" if results else "No successful evaluations")
    print(f"Summary saved to {summary_path}")
    
    return results
