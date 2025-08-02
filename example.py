#!/usr/bin/env python3
"""
Example usage of WTr package for water-templated reaction analysis.

This script demonstrates the basic workflow:
1. Create a simple water cluster
2. Define a model reaction
3. Run surface search
4. Analyze results
"""

import numpy as np
import os
from pathlib import Path

# Import WTr components
from WTr.models.datatypes import Geometry, ReactionSpec, SurfaceSpec, CalcSpec
from WTr.geom.build import build_water_cluster
from WTr.geom.constraints import select_core_atoms
from WTr.io.geometry import save_xyz, load_xyz
from WTr.orchestrators.pipeline import search_best_surfaces

def create_example_geometries():
    """Create example reactant and product geometries."""
    
    # Simple water molecule as reactant
    reactant = Geometry(
        symbols=['O', 'H', 'H'],
        coords=[
            (0.000, 0.000, 0.000),
            (0.757, 0.586, 0.000),
            (-0.757, 0.586, 0.000)
        ],
        comment="Reactant: Water molecule"
    )
    
    # Slightly rotated water as product (simple model reaction)
    product = Geometry(
        symbols=['O', 'H', 'H'],
        coords=[
            (0.000, 0.000, 0.000),
            (0.957, 0.000, 0.000),
            (0.000, 0.957, 0.000)
        ],
        comment="Product: Rotated water molecule"
    )
    
    # Save to files
    save_xyz(reactant, "reactant.xyz")
    save_xyz(product, "product.xyz")
    
    return reactant, product

def main():
    """Run example WTr calculation."""
    
    print("WTr Example: Water-Templated Reaction Analysis")
    print("=" * 50)
    
    # Create example directory
    example_dir = Path("wtr_example")
    example_dir.mkdir(exist_ok=True)
    os.chdir(example_dir)
    
    # Step 1: Create example geometries
    print("1. Creating example geometries...")
    reactant, product = create_example_geometries()
    print(f"   Reactant: {len(reactant.symbols)} atoms")
    print(f"   Product: {len(product.symbols)} atoms")
    
    # Step 2: Define reaction
    print("\n2. Defining reaction specification...")
    reaction = ReactionSpec(
        name="water_rotation_example",
        reactant=reactant,
        product=product,
        reactive_indices={
            "rc": [0, 1, 2],  # All atoms involved
            "p": [0, 1, 2],
            "ts": [0, 1, 2]
        },
        reaction_axis_atoms=(0, 1)  # O-H axis
    )
    print(f"   Reaction: {reaction.name}")
    
    # Step 3: Define surface and calculator
    print("\n3. Setting up surface and calculator...")
    surface_spec = SurfaceSpec(
        waters_n=12,        # Small cluster for example
        radius=6.0,         # Compact cluster
        core_fraction=0.6,  # More core atoms for stability
        random_seed=42      # Reproducible
    )
    
    calc_spec = CalcSpec(
        ase_calculator="xtb",  # Note: requires xtb-python installation
        calc_kwargs={
            "method": "GFN2-xTB",
            "accuracy": 1.0
        },
        charge=0,
        spin_multiplicity=1
    )
    
    print(f"   Surface: {surface_spec.waters_n} waters, R={surface_spec.radius} Å")
    print(f"   Calculator: {calc_spec.ase_calculator}")
    
    # Step 4: Build test cluster
    print("\n4. Building water cluster...")
    cluster = build_water_cluster(surface_spec)
    core_indices = select_core_atoms(cluster, surface_spec)
    
    save_xyz(cluster, "test_cluster.xyz")
    print(f"   Built cluster: {len(cluster.symbols)} atoms")
    print(f"   Core atoms: {len(core_indices)}")
    
    # Step 5: Run simplified search (with limited evaluations for example)
    print("\n5. Running WTr search...")
    print("   Note: This is a minimal example with small parameters")
    
    try:
        results = search_best_surfaces(
            reaction=reaction,
            surface_spec=surface_spec,
            calc_spec=calc_spec,
            temps=[10.0, 20.0],
            n_ts_templates=2,    # Reduced for example
            n_mc_rounds=50,      # Reduced for example
            max_evals=3,         # Reduced for example
            workdir="wtr_search"
        )
        
        # Step 6: Analyze results
        print(f"\n6. Results summary:")
        print(f"   Evaluated {len(results)} surface configurations")
        
        if results:
            best = results[0]
            print(f"   Best barrier (10K): {best.deltaG_dagger_10K:.3f} eV")
            print(f"   Best rate (10K): {best.rate_10K:.2e} s⁻¹")
            print(f"   Surface ID: {best.surface_id}")
            
            # Show top 3 results
            print(f"\n   Top configurations:")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. {result.surface_id}: "
                      f"ΔG‡={result.deltaG_dagger_10K:.3f} eV, "
                      f"k={result.rate_10K:.2e} s⁻¹")
        
        print(f"\n✓ WTr search completed successfully!")
        print(f"   Results saved in: {os.getcwd()}/wtr_search/")
        
    except Exception as e:
        print(f"\n⚠ Search failed (this is expected without proper calculator setup):")
        print(f"   Error: {e}")
        print(f"\n   This example demonstrates the workflow structure.")
        print(f"   For real calculations, ensure ASE calculators are properly installed.")

if __name__ == "__main__":
    main()
