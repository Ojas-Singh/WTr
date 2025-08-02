"""
ASE integration helpers and calculator factory.
"""

import numpy as np
from ase import Atoms
from typing import Optional
from ase.calculators.emt import EMT
from importlib import import_module

from ..models.datatypes import Geometry, CalcSpec

# IMPORTANT:
# Do NOT import xTB here. We now use tblite.ase.TBLite instead of ase.calculators.xtb
# All calculator import attempts are deferred into make_calculator() so the package can load without dependencies.

def geometry_to_atoms(g: Geometry) -> Atoms:
    """Convert Geometry to ASE Atoms object."""
    positions = np.array(g.coords)
    atoms = Atoms(symbols=g.symbols, positions=positions)
    return atoms

def atoms_to_geometry(atoms: Atoms, comment: str = "") -> Geometry:
    """Convert ASE Atoms to Geometry."""
    symbols = list(atoms.get_chemical_symbols())
    coords = [tuple(pos) for pos in atoms.get_positions()]
    return Geometry(symbols=symbols, coords=coords, comment=comment)

def make_calculator(spec: CalcSpec):
    """
    Return an ASE calculator instance.

    Selection order:
      1) If ase_calculator == "xtb": try tblite.ase.TBLite (modern xTB interface)
      2) If ase_calculator == "emt": use EMT.
      3) For any other name, try to import ase.calculators.<name>.<Name> calculator class dynamically.
      4) Final fallback: EMT.

    This uses tblite.ase.TBLite for xTB calculations which is the recommended modern interface.
    """
    name = (spec.ase_calculator or "emt").lower()

    # Helper: Try dynamic import
    def try_dynamic_calc(mod_name: str, class_name: str):
        try:
            m = import_module(mod_name)
            cls = getattr(m, class_name, None)
            if cls is None:
                return None
            return cls
        except Exception:
            return None

    if name == "xtb":
        # Try modern tblite interface first (recommended)
        TBLite = try_dynamic_calc("tblite.ase", "TBLite")
        if TBLite is not None:
            uhf = max(spec.spin_multiplicity - 1, 0)
            calc_kwargs = dict(spec.calc_kwargs or {})
            
            # Map common kwargs to tblite format
            method = calc_kwargs.get("method", "GFN2-xTB")
            
            try:
                return TBLite(
                    method=method,
                    charge=spec.charge,
                    uhf=uhf,
                    **{k: v for k, v in calc_kwargs.items() if k not in ["method"]}
                )
            except Exception:
                pass
        
        # Fallback: try legacy ASE xTB locations
        XTB = try_dynamic_calc("ase.calculators.external.xtb", "XTB")
        if XTB is None:
            XTB = try_dynamic_calc("ase.calculators.xtb", "XTB")
        if XTB is not None:
            uhf = max(spec.spin_multiplicity - 1, 0)
            calc_kwargs = dict(spec.calc_kwargs or {})
            calc_kwargs.update({"charge": spec.charge, "uhf": uhf})
            try:
                return XTB(**calc_kwargs)
            except Exception:
                pass

    if name == "emt":
        return EMT()

    # Generic attempt: map common names to modules/classes in ASE
    generic_map = {
        "morse": ("ase.calculators.morse", "MorsePotential"),
        "lennardjones": ("ase.calculators.lj", "LennardJones"),
        "lj": ("ase.calculators.lj", "LennardJones"),
        "tblite": ("tblite.ase", "TBLite"),  # Allow direct tblite access
    }
    mod_cls = generic_map.get(name)
    if mod_cls:
        cls = try_dynamic_calc(*mod_cls)
        if cls is not None:
            try:
                if name == "tblite":
                    # Special handling for tblite
                    method = (spec.calc_kwargs or {}).get("method", "GFN2-xTB")
                    uhf = max(spec.spin_multiplicity - 1, 0)
                    return cls(method=method, charge=spec.charge, uhf=uhf)
                else:
                    return cls(**(spec.calc_kwargs or {}))
            except Exception:
                pass

    # Final safe fallback
    return EMT()

def set_calculator(atoms: Atoms, calc_spec: CalcSpec) -> None:
    """Set calculator on atoms object."""
    calc = make_calculator(calc_spec)
    # Use modern assignment to avoid FutureWarning: Please use atoms.calc = calc
    atoms.calc = calc
