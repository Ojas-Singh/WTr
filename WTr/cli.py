"""
Command-line interface for WTr package.
"""

import typer
import yaml
import os
from pathlib import Path
from typing import Optional
import traceback

from .models.datatypes import ReactionSpec, SurfaceSpec, CalcSpec, Geometry
from .io.geometry import load_xyz
from .orchestrators.pipeline import search_best_surfaces
from .reports.summarize import generate_report

app = typer.Typer(help="WTr: Water-Templated Reactions")

@app.command()
def init(
    config: str = typer.Option("config.yml", help="Configuration file to create"),
    force: bool = typer.Option(False, help="Overwrite existing config")
):
    """Initialize a new WTr configuration file."""
    
    config_path = Path(config)
    
    if config_path.exists() and not force:
        typer.echo(f"Configuration file {config} already exists. Use --force to overwrite.")
        raise typer.Exit(1)
    
    # Create example configuration
    example_config = {
        'reaction': {
            'name': 'example_reaction',
            'reactant_xyz': 'reactant.xyz',
            'product_xyz': 'product.xyz',
            'ts_seed_xyz': 'ts_seed.xyz',  # optional
            'reactive_indices': {
                'rc': [0, 1],
                'p': [0, 1],
                'ts': [0, 1]
            },
            'reaction_axis_atoms': [0, 1]
        },
        'surface': {
            'waters_n': 20,
            'radius': 8.0,
            'core_fraction': 0.5,
            'random_seed': 42,
            'harmonic_k': None  # None for hard constraints, float for soft
        },
        'calc': {
            'ase_calculator': 'xtb',
            'calc_kwargs': {
                'method': 'GFN2-xTB',
                'accuracy': 1.0
            },
            'charge': 0,
            'spin_multiplicity': 1
        },
        'temps': [10.0, 20.0],
        'search': {
            'n_ts_templates': 5,
            'n_mc_rounds': 500,
            'max_evals': 20,
            'workdir': 'wtr_run'
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)
    
    typer.echo(f"Created configuration file: {config}")
    typer.echo("Edit this file with your reaction details and run 'wtr run' to start.")

@app.command()
def run(
    config: str = typer.Option("config.yml", help="Configuration file"),
    workdir: Optional[str] = typer.Option(None, help="Override working directory")
):
    """Run WTr surface search with given configuration."""
    
    config_path = Path(config)
    
    if not config_path.exists():
        typer.echo(f"Configuration file {config} not found.")
        typer.echo("Run 'wtr init' to create an example configuration.")
        raise typer.Exit(1)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    typer.echo(f"Loading configuration from {config}")
    
    try:
        # Parse reaction specification
        reaction_config = config_data['reaction']
        
        # Load geometries
        reactant = load_xyz(reaction_config['reactant_xyz'])
        product = load_xyz(reaction_config['product_xyz'])
        
        ts_seed = None
        if 'ts_seed_xyz' in reaction_config and reaction_config['ts_seed_xyz']:
            if os.path.exists(reaction_config['ts_seed_xyz']):
                ts_seed = load_xyz(reaction_config['ts_seed_xyz'])
        
        reaction = ReactionSpec(
            name=reaction_config['name'],
            reactant=reactant,
            product=product,
            ts_seed=ts_seed,
            reactive_indices=reaction_config['reactive_indices'],
            reaction_axis_atoms=tuple(reaction_config['reaction_axis_atoms'])
        )
        
        # Parse surface specification
        surface = SurfaceSpec(**config_data['surface'])
        
        # Parse calculator specification
        calc = CalcSpec(**config_data['calc'])
        
        # Get temperatures
        temps = config_data['temps']
        
        # Get search parameters
        search_config = config_data['search']
        run_workdir = workdir or search_config['workdir']
        
        typer.echo(f"Starting WTr search in {run_workdir}")
        
        # Run the search
        results = search_best_surfaces(
            reaction=reaction,
            surface_spec=surface,
            calc_spec=calc,
            temps=temps,
            n_ts_templates=search_config['n_ts_templates'],
            n_mc_rounds=search_config['n_mc_rounds'],
            max_evals=search_config['max_evals'],
            workdir=run_workdir
        )
        
        typer.echo(f"\nSearch completed! Found {len(results)} results.")
        
        if results:
            best = results[0]
            typer.echo(f"Best surface: {best.surface_id}")
            typer.echo(f"Barrier (10K): {best.deltaG_dagger_10K:.3f} eV")
            typer.echo(f"Rate (10K): {best.rate_10K:.2e} s⁻¹")
        
    except Exception as e:
        typer.echo(f"Error running WTr: {e}")
        traceback.print_exc()
        raise typer.Exit(1)

@app.command()
def report(
    run_dir: str = typer.Argument(..., help="WTr run directory"),
    top: int = typer.Option(10, help="Number of top results to include"),
    output: str = typer.Option("report.html", help="Output report file")
):
    """Generate a report from WTr results."""
    
    run_path = Path(run_dir)
    
    if not run_path.exists():
        typer.echo(f"Run directory {run_dir} not found.")
        raise typer.Exit(1)
    
    typer.echo(f"Generating report for {run_dir}")
    
    try:
        generate_report(str(run_path), top, output)
        typer.echo(f"Report saved to {output}")
        
    except Exception as e:
        typer.echo(f"Error generating report: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
