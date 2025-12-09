import numpy as np
import pandas as pd
from numpy.typing import NDArray
from ballphysics.models import Environment, Pickleball
from ballphysics.analysis.tracking import simulate_trajectory_2d


def cost_function_2d(
    Cd: NDArray[np.float64],
    df: pd.DataFrame,
    r0: tuple[float, float],
    v0: tuple[float, float],
    env: 'Environment',
    ball: 'Pickleball',
    t0: float
) -> float:
    """
    Calculate how well a given Cd matches the 2D experimental data.
    Returns the sum of squared errors for both x and y positions.
    """
    t_end = df['time'].max() / 1000
    
    # Run 2D simulation with this Cd value
    sol = simulate_trajectory_2d((t0, t_end), r0, v0, env, ball, Cd=Cd[0])
    
    # Get predicted positions at experimental time points
    t_exp = df['time'].values / 1000
    x_pred = sol.sol(t_exp)[0]
    y_pred = sol.sol(t_exp)[1]
    
    # Calculate sum of squared errors for both dimensions
    x_exp = df['x'].values
    y_exp = df['y'].values
    error = np.sum((x_pred - x_exp)**2 + (y_pred - y_exp)**2)
    
    return error