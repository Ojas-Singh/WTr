"""
Kinetics calculations including tunneling corrections.
"""

import numpy as np
from typing import Dict

from ..utils.units import BOLTZMANN_EV_K, PLANCK_EV_S, HBAR_EV_S, cm1_to_angular_freq

def eyring_rate(deltaG_eV: float, temperature: float, kappa: float = 1.0) -> float:
    """
    Calculate reaction rate using Eyring transition state theory.
    
    Args:
        deltaG_eV: Activation free energy (eV)
        temperature: Temperature (K)
        kappa: Tunneling correction factor
    
    Returns:
        Rate constant (s⁻¹)
    """
    if temperature <= 0:
        return 0.0
    
    # Eyring equation: k = κ * (kB*T/h) * exp(-ΔG‡/kB*T)
    prefactor = kappa * BOLTZMANN_EV_K * temperature / PLANCK_EV_S
    exponential = np.exp(-deltaG_eV / (BOLTZMANN_EV_K * temperature))
    
    return prefactor * exponential

def wigner_kappa(nu_imag_cm1: float, temperature: float) -> float:
    """
    Calculate Wigner tunneling correction factor.
    
    Args:
        nu_imag_cm1: Magnitude of imaginary frequency (cm⁻¹)
        temperature: Temperature (K)
    
    Returns:
        Wigner tunneling correction κ
    """
    if temperature <= 0 or nu_imag_cm1 <= 0:
        return 1.0
    
    # Convert to angular frequency
    omega_imag = cm1_to_angular_freq(abs(nu_imag_cm1))
    
    # Wigner correction: κ = 1 + (1/24) * (ℏω/kBT)²
    hbar_omega_over_kBT = (HBAR_EV_S * omega_imag) / (BOLTZMANN_EV_K * temperature)
    
    kappa = 1.0 + (1.0/24.0) * hbar_omega_over_kBT**2
    
    return kappa

def eckart_transmission_probability(energy: float, V_f: float, V_b: float, 
                                  nu_imag_cm1: float) -> float:
    """
    Calculate transmission probability for Eckart barrier.
    
    Args:
        energy: Kinetic energy (eV)
        V_f: Forward barrier height (eV)
        V_b: Backward barrier height (eV)
        nu_imag_cm1: Imaginary frequency (cm⁻¹)
    
    Returns:
        Transmission probability P(E)
    """
    if energy < 0:
        return 0.0
    
    if energy > max(V_f, V_b):
        return 1.0  # Classical transmission
    
    # Eckart barrier parameters
    # Simplified implementation - full version requires more complex formulas
    
    # Convert frequency to energy scale
    omega_imag = cm1_to_angular_freq(abs(nu_imag_cm1))
    hbar_omega = HBAR_EV_S * omega_imag
    
    # Approximate barrier width from frequency
    # More rigorous: solve Eckart potential parameters from V_f, V_b, ω
    barrier_width = 2.0 * np.sqrt(2.0 * hbar_omega / max(V_f, V_b))  # Rough estimate
    
    # WKB approximation for tunneling
    if energy < min(V_f, V_b):
        # Pure tunneling
        kappa_wkb = np.sqrt(2.0 * (min(V_f, V_b) - energy) / hbar_omega)
        transmission = np.exp(-2.0 * kappa_wkb * barrier_width)
    else:
        # Partial tunneling (simplified)
        transmission = energy / max(V_f, V_b)
    
    return min(transmission, 1.0)

def eckart_kappa(deltaE_fwd_eV: float, deltaE_rev_eV: float, 
                nu_imag_cm1: float, temperature: float) -> float:
    """
    Calculate Eckart tunneling correction factor.
    
    Args:
        deltaE_fwd_eV: Forward barrier height (eV)
        deltaE_rev_eV: Reverse barrier height (eV)  
        nu_imag_cm1: Imaginary frequency magnitude (cm⁻¹)
        temperature: Temperature (K)
    
    Returns:
        Eckart tunneling correction κ
    """
    if temperature <= 0:
        return 1.0
    
    # Thermal average of transmission probability
    # κ = ∫₀^∞ P(E) exp(-E/kBT) dE / ∫₀^∞ exp(-E/kBT) dE
    
    kBT = BOLTZMANN_EV_K * temperature
    
    # Numerical integration (simplified)
    E_max = max(deltaE_fwd_eV, deltaE_rev_eV) + 5 * kBT
    n_points = 100
    dE = E_max / n_points
    
    numerator = 0.0
    denominator = 0.0
    
    for i in range(n_points):
        E = i * dE
        weight = np.exp(-E / kBT)
        
        P_E = eckart_transmission_probability(E, deltaE_fwd_eV, deltaE_rev_eV, nu_imag_cm1)
        
        numerator += P_E * weight * dE
        denominator += weight * dE
    
    if denominator > 0:
        kappa = numerator / denominator
    else:
        kappa = 1.0
    
    return max(kappa, 1.0)  # κ should be ≥ 1

def calculate_rate_constants(barriers: Dict, temperatures: list, 
                           imaginary_freq: float, method: str = "wigner") -> Dict:
    """
    Calculate temperature-dependent rate constants with tunneling.
    
    Args:
        barriers: Dictionary with barrier heights
        temperatures: List of temperatures (K)
        imaginary_freq: Imaginary frequency (cm⁻¹, can be negative)
        method: Tunneling method ("wigner" or "eckart")
    
    Returns:
        Dictionary with rate constants and tunneling factors
    """
    results = {
        'temperatures': temperatures,
        'rates': {},
        'kappa': {},
        'method': method
    }
    
    # Get barrier heights
    deltaE = barriers.get('deltaE', 0.0)
    deltaG_dict = barriers.get('deltaG', {})
    
    for T in temperatures:
        deltaG = deltaG_dict.get(T, deltaE)  # Fallback to electronic barrier
        
        # Calculate tunneling correction
        if method.lower() == "wigner":
            kappa = wigner_kappa(abs(imaginary_freq), T)
        elif method.lower() == "eckart":
            # For Eckart, need forward and reverse barriers
            # Simplified: assume symmetric barrier
            V_f = deltaE
            V_b = deltaE  # Would need product barrier for asymmetric case
            kappa = eckart_kappa(V_f, V_b, abs(imaginary_freq), T)
        else:
            kappa = 1.0  # No tunneling correction
        
        # Calculate rate
        rate = eyring_rate(deltaG, T, kappa)
        
        results['rates'][T] = rate
        results['kappa'][T] = kappa
    
    return results

def print_kinetics_summary(rate_data: Dict, barriers: Dict):
    """
    Print summary of kinetics calculations.
    
    Args:
        rate_data: Rate constant data
        barriers: Barrier height data
    """
    print("\n" + "="*50)
    print("KINETICS SUMMARY")
    print("="*50)
    
    print(f"Tunneling method: {rate_data['method']}")
    print(f"Electronic barrier: {barriers.get('deltaE', 0.0):.3f} eV")
    
    print("\nTemperature-dependent results:")
    print("T (K)    ΔG‡ (eV)    κ        k (s⁻¹)")
    print("-" * 40)
    
    deltaG_dict = barriers.get('deltaG', {})
    
    for T in rate_data['temperatures']:
        deltaG = deltaG_dict.get(T, barriers.get('deltaE', 0.0))
        kappa = rate_data['kappa'][T]
        rate = rate_data['rates'][T]
        
        print(f"{T:4.0f}    {deltaG:7.3f}    {kappa:6.2f}   {rate:.2e}")
    
    print("-" * 40)
    
    # Highlight low-temperature tunneling
    low_T_rates = [(T, rate_data['rates'][T]) for T in rate_data['temperatures'] if T <= 20]
    if low_T_rates:
        print(f"\nLow-temperature rates (tunneling important):")
        for T, rate in low_T_rates:
            kappa = rate_data['kappa'][T]
            enhancement = kappa - 1.0
            print(f"  {T} K: {rate:.2e} s⁻¹ (κ={kappa:.2f}, {enhancement*100:.1f}% enhancement)")
    
    print("="*50)
