import numpy as np

# Laser parameters 
class GreenLaser():
    def __init__(self, delta=50, omega=60, phi=-np.pi / 4, detuning=0, phi_right=0):
        self.lambda_green = 556e-9 # wavelength = 556 nm
        self.w0 = 2 * np.pi * 3e8 / self.lambda_green  # Atomic resonance frequency omega_0 = 2 * pi * c / 556e-9
        self.hbar = 6.63e-34 / (2 * np.pi) # Planck constant / 2 * pi
        self.k = 2 * np.pi / self.lambda_green  # wavenumber
        self.gamma = 2 * np.pi * 182e3  # Spontaneous decay rate
        self.Delta = delta * self.gamma  # Bichromatic detuning
        self.Omega = omega * self.gamma  # Rabi frequency
        self.tau = 1 / self.gamma # Transition lifetime
        self.detuning = detuning  # laser detuning from atomic resonance w0
        self.phi_left = phi  # phase of laser beams moving towards the atom (-pi/4 by default)
        self.phi_right = phi_right  # phase of laser beams moving in the same direction as the atom (zero by default)
        self.atom_mass = 174 * 1.67e-27 # Yb-174 atom's mass in kg
        self.beam_radius = 0.004  # laser beam radius = 4 mm by default


class GreenLaserSquare():
    def __init__(self, delta=50, omega=60, phi=-np.pi / 4, detuning=0, phi_right=0):
        self.lambda_green = 556e-9 # wavelength = 556 nm
        self.w0 = 2 * np.pi * 3e8 / self.lambda_green  # Atomic resonance frequency omega_0 = 2 * pi * c / 556e-9
        self.hbar = 6.63e-34 / (2 * np.pi) # Planck constant / 2 * pi
        self.k = 2 * np.pi / self.lambda_green  # wavenumber
        self.gamma = 2 * np.pi * 182e3  # Spontaneous decay rate
        self.Delta = delta * self.gamma  # Modulation frequency
        self.Omega = omega * self.gamma  # Rabi frequency
        self.k_Rabi = self.Delta / 3e8  # wave number for Rabi freq modulation
        self.tau = 1 / self.gamma # Transition lifetime
        self.detuning = detuning  # laser detuning from atomic resonance w0
        self.phi_left = phi  # phase of laser beams moving towards the atom (-pi/4 by default)
        self.phi_right = phi_right  # phase of laser beams moving in the same direction as the atom (zero by default)
        self.phasemod_left = 0 # PM left: laser beams moving towards the atom (0: phase modulation off, 1: phase modulation on)
        self.phasemod_right = 0 # PM right: laser beams moving in the same direction as the atom (0: phase modulation off, 1: phase modulation on)
        self.atom_mass = 174 * 1.67e-27 # Yb-174 atom's mass in kg
        self.beam_radius = 0.004  # laser beam radius = 4 mm by default
