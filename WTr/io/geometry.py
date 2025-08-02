"""
Geometry I/O and manipulation functions.
"""

import numpy as np
from typing import Tuple

from ..models.datatypes import Geometry

def load_xyz(path: str) -> Geometry:
    """Load geometry from XYZ file.

    Tolerates leading blank lines and UTF-8 BOM.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Strip UTF-8 BOM if present
    if raw.startswith('\ufeff'):
        raw = raw.lstrip('\ufeff')

    # Split lines and skip leading blanks
    lines_all = raw.splitlines()
    i = 0
    while i < len(lines_all) and lines_all[i].strip() == "":
        i += 1
    if i >= len(lines_all):
        raise ValueError(f"Empty XYZ file: {path}")

    # Parse header
    try:
        n_atoms = int(lines_all[i].strip())
    except Exception as e:
        raise ValueError(f"Invalid XYZ header in {path!r}: expected integer atom count on first non-blank line, got {lines_all[i]!r}") from e

    if i + 1 >= len(lines_all):
        raise ValueError(f"Missing XYZ comment line in {path!r}")
    comment = lines_all[i + 1].strip()

    # Parse coordinates
    start = i + 2
    end = start + n_atoms
    if end > len(lines_all):
        raise ValueError(f"XYZ file {path!r} truncated: need {n_atoms} atom lines, have {len(lines_all) - start}")
    symbols = []
    coords = []
    for j in range(start, end):
        line = lines_all[j].strip()
        if not line:
            raise ValueError(f"Blank atom line at index {j - start + 1} in {path!r}")
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line {j - start + 1} in {path!r}: {line!r}")
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

    return Geometry(symbols=symbols, coords=coords, comment=comment)

def save_xyz(g: Geometry, path: str) -> None:
    """Save geometry to XYZ file."""
    with open(path, 'w') as f:
        f.write(f"{len(g.symbols)}\n")
        f.write(f"{g.comment}\n")
        for symbol, coord in zip(g.symbols, g.coords):
            f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

def merge(ga: Geometry, gb: Geometry) -> Geometry:
    """Merge two geometries into one."""
    symbols = ga.symbols + gb.symbols
    coords = ga.coords + gb.coords
    comment = f"Merged: {ga.comment} + {gb.comment}"
    return Geometry(symbols=symbols, coords=coords, comment=comment)

def rigid_place(fragment: Geometry, origin: Tuple[float, float, float],
                axis: Tuple[float, float, float], angle: float) -> Geometry:
    """
    Place fragment at origin with specified axis orientation and rotation angle.
    """
    # Convert to numpy arrays
    coords = np.array(fragment.coords)
    
    # Translate to origin
    coords_centered = coords - np.mean(coords, axis=0)
    
    # Apply rotation (simplified - would need full quaternion rotation)
    # For now, just translate to origin
    coords_placed = coords_centered + np.array(origin)
    
    new_coords = [tuple(coord) for coord in coords_placed]
    
    return Geometry(
        symbols=fragment.symbols,
        coords=new_coords,
        comment=f"Placed at {origin} with axis {axis}"
    )
