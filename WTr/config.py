"""
Configuration management for WTr package.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
import os

@dataclass
class WTrConfig:
    """Central configuration class for WTr package."""
    
    # Reaction settings
    reaction_name: str = "default_reaction"
    reactant_xyz: str = "reactant.xyz"
    product_xyz: str = "product.xyz"
    ts_seed_xyz: Optional[str] = None
    reactive_indices: Dict[str, list] = None
    reaction_axis_atoms: tuple = (0, 1)
    
    # Surface settings
    waters_n: int = 20
    radius: float = 8.0
    core_fraction: float = 0.5
    random_seed: int = 42
    harmonic_k: Optional[float] = None
    
    # Calculator settings
    ase_calculator: str = "xtb"
    calc_kwargs: Dict[str, Any] = None
    charge: int = 0
    spin_multiplicity: int = 1
    
    # Temperature settings
    temperatures: list = None
    
    # Search settings
    n_ts_templates: int = 5
    n_mc_rounds: int = 500
    max_evals: int = 20
    workdir: str = "wtr_run"
    
    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.reactive_indices is None:
            self.reactive_indices = {"rc": [0, 1], "p": [0, 1], "ts": [0, 1]}
        
        if self.calc_kwargs is None:
            self.calc_kwargs = {"method": "GFN2-xTB", "accuracy": 1.0}
        
        if self.temperatures is None:
            self.temperatures = [10.0, 20.0]
    
    @classmethod
    def from_yaml(cls, config_file: str) -> 'WTrConfig':
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten nested structure
        config_args = {}
        
        # Reaction section
        if 'reaction' in data:
            reaction = data['reaction']
            config_args.update({
                'reaction_name': reaction.get('name', 'default_reaction'),
                'reactant_xyz': reaction.get('reactant_xyz', 'reactant.xyz'),
                'product_xyz': reaction.get('product_xyz', 'product.xyz'),
                'ts_seed_xyz': reaction.get('ts_seed_xyz'),
                'reactive_indices': reaction.get('reactive_indices'),
                'reaction_axis_atoms': tuple(reaction.get('reaction_axis_atoms', [0, 1]))
            })
        
        # Surface section
        if 'surface' in data:
            surface = data['surface']
            config_args.update({
                'waters_n': surface.get('waters_n', 20),
                'radius': surface.get('radius', 8.0),
                'core_fraction': surface.get('core_fraction', 0.5),
                'random_seed': surface.get('random_seed', 42),
                'harmonic_k': surface.get('harmonic_k')
            })
        
        # Calculator section
        if 'calc' in data:
            calc = data['calc']
            config_args.update({
                'ase_calculator': calc.get('ase_calculator', 'xtb'),
                'calc_kwargs': calc.get('calc_kwargs'),
                'charge': calc.get('charge', 0),
                'spin_multiplicity': calc.get('spin_multiplicity', 1)
            })
        
        # Other sections
        config_args['temperatures'] = data.get('temps', [10.0, 20.0])
        
        if 'search' in data:
            search = data['search']
            config_args.update({
                'n_ts_templates': search.get('n_ts_templates', 5),
                'n_mc_rounds': search.get('n_mc_rounds', 500),
                'max_evals': search.get('max_evals', 20),
                'workdir': search.get('workdir', 'wtr_run')
            })
        
        return cls(**config_args)
    
    def to_yaml(self, config_file: str):
        """Save configuration to YAML file."""
        data = {
            'reaction': {
                'name': self.reaction_name,
                'reactant_xyz': self.reactant_xyz,
                'product_xyz': self.product_xyz,
                'ts_seed_xyz': self.ts_seed_xyz,
                'reactive_indices': self.reactive_indices,
                'reaction_axis_atoms': list(self.reaction_axis_atoms)
            },
            'surface': {
                'waters_n': self.waters_n,
                'radius': self.radius,
                'core_fraction': self.core_fraction,
                'random_seed': self.random_seed,
                'harmonic_k': self.harmonic_k
            },
            'calc': {
                'ase_calculator': self.ase_calculator,
                'calc_kwargs': self.calc_kwargs,
                'charge': self.charge,
                'spin_multiplicity': self.spin_multiplicity
            },
            'temps': self.temperatures,
            'search': {
                'n_ts_templates': self.n_ts_templates,
                'n_mc_rounds': self.n_mc_rounds,
                'max_evals': self.max_evals,
                'workdir': self.workdir
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Check required files exist
        if not os.path.exists(self.reactant_xyz):
            errors.append(f"Reactant file not found: {self.reactant_xyz}")
        
        if not os.path.exists(self.product_xyz):
            errors.append(f"Product file not found: {self.product_xyz}")
        
        if self.ts_seed_xyz and not os.path.exists(self.ts_seed_xyz):
            errors.append(f"TS seed file not found: {self.ts_seed_xyz}")
        
        # Check parameter ranges
        if self.waters_n <= 0:
            errors.append("Number of waters must be positive")
        
        if self.radius <= 0:
            errors.append("Cluster radius must be positive")
        
        if not 0 < self.core_fraction < 1:
            errors.append("Core fraction must be between 0 and 1")
        
        if any(T <= 0 for T in self.temperatures):
            errors.append("All temperatures must be positive")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

def load_config(config_file: str) -> WTrConfig:
    """Load and validate WTr configuration."""
    config = WTrConfig.from_yaml(config_file)
    
    if not config.validate():
        raise ValueError("Configuration validation failed")
    
    return config
