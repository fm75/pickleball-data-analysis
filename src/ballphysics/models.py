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

    @classmethod
    def from_conditions(cls, temp_c=20.0, pressure_inhg=29.92):
        """
        Create Environment from temperature and pressure.
        
        Uses ideal gas law: ρ = PM/(RT)
        
        Parameters:
        - temp_c: temperature in Celsius (default 20°C ≈ 68°F)
        - pressure_inhg: pressure in inches of mercury (default 29.92 inHg = sea level)
        
        Returns:
        - Environment with calculated air density in lb/ft³
        """
        # Convert temperature to Kelvin
        temp_k = temp_c + 273.15
        
        # Convert pressure from inHg to Pascals
        # 1 inHg = 3386.39 Pa
        pressure_pa = pressure_inhg * 3386.39
        
        # Ideal gas law for air density
        # R = 287.05 J/(kg·K) for dry air
        # M = molar mass not needed when using specific gas constant
        rho_kg_m3 = pressure_pa / (287.05 * temp_k)
        
        # Convert from kg/m³ to lb/ft³
        # 1 kg/m³ = 0.062428 lb/ft³
        rho_lb_ft3 = rho_kg_m3 * 0.062428
        
        return cls(rho_lb_ft3=rho_lb_ft3)