"""
WTr: Water-Templated Reactions

A Python package for building N-water amorphous clusters/surfaces and generating 
physics-guided surface microstates that lower reaction transition-state barriers.
"""

__version__ = "0.1.0"
__author__ = "WTr Development Team"

from .models.datatypes import (
    Geometry, ReactionSpec, SurfaceSpec, CalcSpec, 
    DescriptorSet, EvalResult
)

from .geom.build import build_water_cluster
from .geom.constraints import select_core_atoms, make_constraints
from .orchestrators.pipeline import evaluate_surface, search_best_surfaces

__all__ = [
    "Geometry", "ReactionSpec", "SurfaceSpec", "CalcSpec", 
    "DescriptorSet", "EvalResult",
    "build_water_cluster", "select_core_atoms", "make_constraints",
    "evaluate_surface", "search_best_surfaces"
]
