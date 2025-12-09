from dataclasses import dataclass
import numpy as np

@dataclass
class Pickleball:
    """Pickleball properties"""
    mass_g: float  # grams
    radius_in: float  # inches
    
    @property
    def mass_lb(self):
        return self.mass_g / 453.592
    
    @property
    def area_ft2(self):
        return np.pi * (self.radius_in / 12)**2


@dataclass
class Environment:
    """Environmental conditions"""
    g_ft_s2: float = 32.174  # gravitational acceleration
    rho_lb_ft3: float = 0.0765  # air density at sea level, STP