import numpy as np
from scipy.integrate import solve_ivp
from numpy.typing import NDArray
from ballphysics.models import Environment, Pickleball


def simulate_trajectory_2d(
    t_span: tuple[float, float], 
    r0: NDArray[np.float64] | tuple[float, float],
    v0: NDArray[np.float64] | tuple[float, float],
    env: 'Environment',
    ball: 'Pickleball',
    Cd: float = 0
) -> 'OdeSolution':
    """
    Simulate 2D pickleball trajectory with air resistance.
    
    Parameters:
    - t_span: (t_start, t_end) in seconds
    - r0: initial position [x0, y0] in feet
    - v0: initial velocity [vx0, vy0] in ft/s
    - m: mass in grams
    - r_ball: radius in inches
    - Cd: coefficient of drag (default 0 for free fall)
    
    Returns: solution object from solve_ivp
    """
    from scipy.integrate import solve_ivp
    
    # Constants
    g = np.array([0, env.g_ft_s2])
    rho = env.rho_lb_ft3
    m_lb = ball.mass_lb
    A = ball.area_ft2
    
    # Drag coefficient term
    k = 0.5 * Cd * rho * A / m_lb if Cd > 0 else 0
    
    def derivatives(t, state):
        # state = [x, y, vx, vy]
        r = state[:2]
        v = state[2:]
        
        # Calculate acceleration
        v_mag = np.linalg.norm(v)
        a_drag = -k * v * v_mag if v_mag > 0 else np.array([0, 0])
        a = g + a_drag
        
        return np.concatenate([v, a])
    
    # Initial state: [x0, y0, vx0, vy0]
    state0 = np.concatenate([r0, v0])
    
    # Solve ODE
    sol = solve_ivp(derivatives, t_span, state0, 
                    dense_output=True, max_step=0.01)
    
    return sol