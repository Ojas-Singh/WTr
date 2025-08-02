"""
Example test for WTr package functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from WTr.models.datatypes import Geometry, SurfaceSpec, CalcSpec
from WTr.geom.build import build_water_cluster
from WTr.geom.constraints import select_core_atoms
from WTr.geom.transforms import rotation_matrix
from WTr.utils.units import cm1_to_angular_freq, eV_to_J
from WTr.calc.kinetics import wigner_kappa, eyring_rate

class TestGeometry:
    """Test geometry handling."""
    
    def test_geometry_creation(self):
        """Test creating a geometry object."""
        symbols = ['O', 'H', 'H']
        coords = [(0.0, 0.0, 0.0), (0.757, 0.586, 0.0), (-0.757, 0.586, 0.0)]
        
        geom = Geometry(symbols=symbols, coords=coords, comment="Water molecule")
        
        assert len(geom.symbols) == 3
        assert len(geom.coords) == 3
        assert geom.symbols[0] == 'O'
        assert geom.comment == "Water molecule"
    
    def test_xyz_io(self):
        """Test XYZ file I/O."""
        from WTr.io.geometry import save_xyz, load_xyz
        
        # Create test geometry
        symbols = ['O', 'H', 'H']
        coords = [(0.0, 0.0, 0.0), (0.757, 0.586, 0.0), (-0.757, 0.586, 0.0)]
        geom = Geometry(symbols=symbols, coords=coords, comment="Test water")
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            save_xyz(geom, temp_path)
            loaded_geom = load_xyz(temp_path)
            
            assert loaded_geom.symbols == geom.symbols
            assert len(loaded_geom.coords) == len(geom.coords)
            
            # Check coordinates are close
            for orig, loaded in zip(geom.coords, loaded_geom.coords):
                assert abs(orig[0] - loaded[0]) < 1e-6
                assert abs(orig[1] - loaded[1]) < 1e-6
                assert abs(orig[2] - loaded[2]) < 1e-6
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestWaterCluster:
    """Test water cluster building."""
    
    def test_build_small_cluster(self):
        """Test building a small water cluster."""
        spec = SurfaceSpec(
            waters_n=5,
            radius=6.0,
            core_fraction=0.6,
            random_seed=42
        )
        
        cluster = build_water_cluster(spec)
        
        # Check we got the right number of atoms (5 waters = 15 atoms)
        assert len(cluster.symbols) == 15
        
        # Check we have the right atom types
        o_count = sum(1 for s in cluster.symbols if s == 'O')
        h_count = sum(1 for s in cluster.symbols if s == 'H')
        
        assert o_count == 5
        assert h_count == 10
        
        # Check coordinates are within radius
        coords = np.array(cluster.coords)
        center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        
        assert np.max(distances) <= spec.radius + 2.0  # Allow some tolerance
    
    def test_core_selection(self):
        """Test core atom selection."""
        spec = SurfaceSpec(
            waters_n=6,
            radius=5.0,
            core_fraction=0.5,
            random_seed=123
        )
        
        cluster = build_water_cluster(spec)
        core_indices = select_core_atoms(cluster, spec)
        
        # Should select 3 water molecules (50% of 6) = 9 atoms total
        assert len(core_indices) >= 6  # At least 2 waters worth
        assert len(core_indices) <= 12  # At most 4 waters worth

class TestTransforms:
    """Test geometric transformations."""
    
    def test_rotation_matrix(self):
        """Test rotation matrix generation."""
        # Test 90-degree rotation around z-axis
        axis = np.array([0, 0, 1])
        angle = np.pi / 2
        
        R = rotation_matrix(axis, angle)
        
        # Rotate point (1, 0, 0) -> should become (0, 1, 0)
        point = np.array([1, 0, 0])
        rotated = R @ point
        
        expected = np.array([0, 1, 0])
        
        assert np.allclose(rotated, expected, atol=1e-10)
    
    def test_identity_rotation(self):
        """Test zero-angle rotation gives identity."""
        axis = np.array([1, 1, 1])  # Arbitrary axis
        angle = 0.0
        
        R = rotation_matrix(axis, angle)
        
        assert np.allclose(R, np.eye(3), atol=1e-10)

class TestUnits:
    """Test unit conversions."""
    
    def test_frequency_conversion(self):
        """Test cm⁻¹ to angular frequency conversion."""
        # 1000 cm⁻¹ is a typical vibrational frequency
        nu_cm1 = 1000.0
        omega = cm1_to_angular_freq(nu_cm1)
        
        # Should be positive and reasonable magnitude
        assert omega > 0
        assert omega > 1e11  # rad/s
        assert omega < 1e15  # rad/s
    
    def test_energy_conversion(self):
        """Test eV to Joules conversion."""
        energy_eV = 1.0
        energy_J = eV_to_J(energy_eV)
        
        # Should match known conversion factor
        expected = 1.602176634e-19  # Joules
        assert abs(energy_J - expected) < 1e-25

class TestKinetics:
    """Test kinetics calculations."""
    
    def test_wigner_tunneling(self):
        """Test Wigner tunneling correction."""
        # At very low temperature, tunneling should be significant
        nu_imag = 1000.0  # cm⁻¹
        temperature = 10.0  # K
        
        kappa = wigner_kappa(nu_imag, temperature)
        
        # Should be greater than 1 (enhancement)
        assert kappa > 1.0
        assert kappa < 10.0  # Should be reasonable
    
    def test_eyring_rate(self):
        """Test Eyring rate calculation."""
        deltaG = 0.5  # eV
        temperature = 300.0  # K
        kappa = 1.0  # No tunneling
        
        rate = eyring_rate(deltaG, temperature, kappa)
        
        # Should be positive
        assert rate > 0
        
        # Should be reasonable order of magnitude
        assert rate > 1e-10  # s⁻¹
        assert rate < 1e20   # s⁻¹
    
    def test_temperature_dependence(self):
        """Test that rate increases with temperature."""
        deltaG = 0.3  # eV
        
        rate_low = eyring_rate(deltaG, 100.0, 1.0)
        rate_high = eyring_rate(deltaG, 300.0, 1.0)
        
        assert rate_high > rate_low

class TestDataTypes:
    """Test pydantic data models."""
    
    def test_surface_spec(self):
        """Test SurfaceSpec validation."""
        spec = SurfaceSpec(
            waters_n=10,
            radius=7.0,
            core_fraction=0.4
        )
        
        assert spec.waters_n == 10
        assert spec.radius == 7.0
        assert spec.core_fraction == 0.4
        assert spec.random_seed == 0  # Default value
    
    def test_calc_spec(self):
        """Test CalcSpec with defaults."""
        spec = CalcSpec()
        
        assert spec.ase_calculator == "xtb"
        assert spec.calc_kwargs["method"] == "GFN2-xTB"
        assert spec.charge == 0
        assert spec.spin_multiplicity == 1

# Test fixtures for integration tests
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def water_molecule():
    """Create a simple water molecule geometry."""
    symbols = ['O', 'H', 'H']
    coords = [(0.0, 0.0, 0.0), (0.757, 0.586, 0.0), (-0.757, 0.586, 0.0)]
    return Geometry(symbols=symbols, coords=coords, comment="Test water")

class TestIntegration:
    """Integration tests using fixtures."""
    
    def test_cluster_to_file(self, temp_dir, water_molecule):
        """Test saving cluster to file."""
        from WTr.io.geometry import save_xyz
        
        output_path = os.path.join(temp_dir, "test_water.xyz")
        save_xyz(water_molecule, output_path)
        
        assert os.path.exists(output_path)
        
        # Check file content
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        assert lines[0].strip() == "3"  # Number of atoms
        assert "O" in lines[2]  # Oxygen in first atom line

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
