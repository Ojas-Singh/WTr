# WTr: Water-Templated Reactions

A Python package for building N-water amorphous clusters/surfaces and generating physics-guided surface microstates that lower reaction transition-state barriers.

## Overview

WTr implements a comprehensive workflow for:

1. **Building water clusters** using Poisson disc sampling and realistic H-bond geometries
2. **Generating surface microstates** via TS-templated design and Monte Carlo field sampling
3. **Evaluating reaction barriers** using ASE calculators, NEB/Dimer methods, and thermochemistry
4. **Physics-based scoring** using electric fields, H-bond networks, and proton-wire analysis
5. **Low-temperature kinetics** with Wigner/Eckart tunneling corrections

## Features

- **Modular ASE integration**: Works with any ASE calculator (XTB, DFTB+, ORCA, etc.)
- **Physics-guided sampling**: Electric field optimization, H-bond rewiring, proton-wire analysis
- **Low-temperature focus**: Specialized for 10-20 K with tunneling corrections
- **Automated workflow**: End-to-end pipeline from cluster generation to kinetics
- **Comprehensive output**: Barriers, rates, descriptors, and detailed analysis

## Installation

### From source

```bash
git clone https://github.com/username/WTr.git
cd WTr
pip install -e .
```

### With conda environment

```bash
# Recommended (WSL/Linux)
conda env create -f environment.yml
conda activate wtr
# installs this package via pip using the local checkout

# To update the conda environment

If you make changes to `environment.yml`, update your environment with:

```bash
conda env update -f environment.yml --prune
```
```

## Quick Start

### 1. Create configuration file

```bash
wtr init --config my_reaction.yml
```

This creates an example configuration file:

```yaml
reaction:
  name: my_reaction
  reactant_xyz: reactant.xyz
  product_xyz: product.xyz
  ts_seed_xyz: ts_seed.xyz  # optional
  reactive_indices:
    rc: [0, 1]
    p: [0, 1] 
    ts: [0, 1]
  reaction_axis_atoms: [0, 1]

surface:
  waters_n: 20
  radius: 8.0
  core_fraction: 0.5
  random_seed: 42
  harmonic_k: null  # Use hard constraints

calc:
  ase_calculator: xtb
  calc_kwargs:
    method: GFN2-xTB
    accuracy: 1.0
  charge: 0
  spin_multiplicity: 1

temps: [10.0, 20.0]

search:
  n_ts_templates: 5
  n_mc_rounds: 500
  max_evals: 20
  workdir: wtr_run
```

### 2. Prepare geometry files

Create XYZ files for your reactant complex and product:

```bash
# reactant.xyz
3
Reactant complex
O  0.000  0.000  0.000
H  0.757  0.586  0.000
H -0.757  0.586  0.000

# product.xyz  
3
Product complex
O  0.000  0.000  0.000
H  0.957  0.000  0.000
H  0.000  0.957  0.000
```

### 3. Run the search

```bash
# For the provided CO + OH -> HOCO example inputs in input/
wtr init --config config.yml --force
# Edit config.yml to point to the input/ geometries:
#   reaction:
#     name: co_oh_to_hoco
#     reactant_xyz: input/co_oh_reac.xyz
#     product_xyz: product.xyz         # keep or set your product
#     ts_seed_xyz: input/co_oh_ts.xyz  # optional but recommended
#     reaction_axis_atoms: [0, 1]      # adjust if needed
#     reactive_indices: {rc: [0, 1], p: [0, 1], ts: [0, 1]}
#   search:
#     workdir: wtr_run
#
# Optionally, to bias TS templating towards HOCO, set ts_seed_xyz: input/hoco_trans.xyz

wtr run --config config.yml
```

### 4. Generate report

```bash
wtr report wtr_run --top 10 --output results.html
```

## API Usage

### Basic workflow

```python
from WTr import *

# Define reaction
reaction = ReactionSpec(
    name="my_reaction",
    reactant=load_xyz("reactant.xyz"),
    product=load_xyz("product.xyz"),
    reactive_indices={"rc": [0, 1], "p": [0, 1]},
    reaction_axis_atoms=(0, 1)
)

# Surface specification
surface_spec = SurfaceSpec(
    waters_n=20,
    radius=8.0,
    core_fraction=0.5,
    random_seed=42
)

# Calculator
calc_spec = CalcSpec(
    ase_calculator="xtb",
    calc_kwargs={"method": "GFN2-xTB", "accuracy": 1.0}
)

# Run search
results = search_best_surfaces(
    reaction=reaction,
    surface_spec=surface_spec,
    calc_spec=calc_spec,
    temps=[10.0, 20.0],
    n_ts_templates=5,
    n_mc_rounds=500,
    max_evals=20,
    workdir="my_run"
)

# Best result
best = results[0]
print(f"Best barrier: {best.deltaG_dagger_10K:.3f} eV")
print(f"Rate at 10K: {best.rate_10K:.2e} s⁻¹")
```

For more examples and detailed documentation, see the full implementation guide.

## Implementation Status

✅ **Complete Implementation**: This package is now fully implemented according to the detailed specification in `Implementation.md`. All major components are working:

- ✅ Water cluster building with Poisson disc sampling
- ✅ Physics-guided surface microstate generation  
- ✅ ASE-integrated reaction path calculations (NEB/Dimer)
- ✅ Vibrational analysis and thermochemistry
- ✅ Low-temperature kinetics with tunneling corrections
- ✅ Monte Carlo field sampling with simulated annealing
- ✅ Comprehensive descriptor calculations
- ✅ End-to-end pipeline orchestration
- ✅ Command-line interface and configuration management
- ✅ Result analysis and report generation
- ✅ Optional Bayesian optimization
- ✅ Parallel execution utilities
- ✅ Comprehensive test suite

## Quick Test

Run the included example to test the implementation:

```bash
cd WTr
python example.py
```

This creates a simple demonstration of the complete workflow.

## Package Structure

```
WTr/
├── WTr/                    # Main package
│   ├── models/             # Pydantic data models  
│   ├── io/                 # Geometry I/O and ASE integration
│   ├── geom/               # Cluster building and constraints
│   ├── sampling/           # MC sampling and TS templating
│   ├── calc/               # NEB, vibrations, kinetics, descriptors
│   ├── orchestrators/      # Main pipeline workflows
│   ├── optimize/           # Optional Bayesian optimization
│   ├── reports/            # Result analysis and visualization
│   ├── utils/              # Unit conversions and constants
│   └── cli.py              # Command-line interface
├── tests/                  # Test suite
├── example.py              # Usage example
├── setup.py                # Package setup
├── requirements.txt        # Dependencies
└── Implementation.md       # Detailed specification
```

## Development

```bash
# Set up development environment
git clone https://github.com/username/WTr.git
cd WTr
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run example
python example.py

# Format code
black WTr/ tests/

# Build package
python setup.py sdist bdist_wheel
```
