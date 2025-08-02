"""
Pydantic data models for WTr package.
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any

class Geometry(BaseModel):
    """Molecular geometry representation."""
    symbols: List[str]
    coords: List[Tuple[float, float, float]]  # Å
    comment: str = ""

class ReactionSpec(BaseModel):
    """Specification for a chemical reaction."""
    name: str
    reactant: Geometry              # RC complex
    product: Geometry               # P
    ts_seed: Optional[Geometry] = None
    reactive_indices: Dict[str, List[int]]  # {"rc":[iA,iB,...], "p":[...], "ts":[...]}
    reaction_axis_atoms: Tuple[int, int]     # (donor_idx, acceptor_idx) or (A,B) that define axis

class SurfaceSpec(BaseModel):
    """Specification for water surface/cluster."""
    waters_n: int
    radius: float                   # spherical envelope radius (Å)
    core_fraction: float = 0.5      # fraction of waters in "core"
    random_seed: int = 0
    harmonic_k: Optional[float] = None  # eV/Å^2 for soft restraint (None => hard FixAtoms)

class CalcSpec(BaseModel):
    """Calculator specification."""
    ase_calculator: str = "xtb"     # "xtb", "dftb", "orca", etc.
    calc_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"method": "GFN2-xTB", "accuracy": 1.0})
    charge: int = 0
    spin_multiplicity: int = 1

class DescriptorSet(BaseModel):
    """Set of descriptors characterizing a surface configuration."""
    efield: Tuple[float, float, float]
    efield_proj: float
    donor_count: int
    acceptor_count: int
    wire_length: float
    wire_bend_max: float
    strain_penalty: float

class EvalResult(BaseModel):
    """Results from evaluating a surface configuration."""
    surface_id: str
    deltaE_dagger: float            # eV
    deltaG_dagger_10K: float        # eV
    deltaG_dagger_20K: float        # eV
    rate_10K: float                 # s^-1
    rate_20K: float                 # s^-1
    descriptors: DescriptorSet
    paths: Dict[str, str]            # filepaths for RC/TS/P, vib, etc.
