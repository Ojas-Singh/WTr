"""
Physical constants and unit conversions for WTr package.
"""

import numpy as np

# Physical constants (CODATA 2018)
EV_TO_J = 1.602176634e-19  # eV to Joules
BOLTZMANN_EV_K = 8.617333262e-5  # eV/K
PLANCK_EV_S = 4.135667696e-15  # eV·s
HBAR_EV_S = PLANCK_EV_S / (2 * np.pi)  # eV·s
SPEED_OF_LIGHT_CM_S = 2.99792458e10  # cm/s

# Atomic units
BOHR_TO_ANG = 0.5291772109  # Bohr to Angstroms
HARTREE_TO_EV = 27.211386245988  # Hartree to eV
AU_TO_DEBYE = 2.541746451  # atomic units to Debye

# Conversion functions
def cm1_to_angular_freq(nu_cm1):
    """Convert wavenumber (cm⁻¹) to angular frequency (rad/s)."""
    return 2 * np.pi * SPEED_OF_LIGHT_CM_S * nu_cm1

def angular_freq_to_cm1(omega_rad_s):
    """Convert angular frequency (rad/s) to wavenumber (cm⁻¹)."""
    return omega_rad_s / (2 * np.pi * SPEED_OF_LIGHT_CM_S)

def eV_to_J(energy_eV):
    """Convert energy from eV to Joules."""
    return energy_eV * EV_TO_J

def J_to_eV(energy_J):
    """Convert energy from Joules to eV."""
    return energy_J / EV_TO_J

def kelvin_to_eV(temp_K):
    """Convert temperature from Kelvin to eV."""
    return temp_K * BOLTZMANN_EV_K

def eV_to_kelvin(energy_eV):
    """Convert energy from eV to Kelvin."""
    return energy_eV / BOLTZMANN_EV_K
