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


def calculate_fit_statistics(df, sol, t0):
    """Calculate RMSE and R² for the fit."""
    # Get predicted positions at experimental time points
    t_exp = df['time'].values / 1000
    x_pred = sol.sol(t_exp)[0]
    y_pred = sol.sol(t_exp)[1]
    
    # Experimental values
    x_exp = df['x'].values
    y_exp = df['y'].values
    
    # Calculate residuals
    x_residuals = x_pred - x_exp
    y_residuals = y_pred - y_exp
    
    # RMSE (combined and per-dimension)
    rmse_x = np.sqrt(np.mean(x_residuals**2))
    rmse_y = np.sqrt(np.mean(y_residuals**2))
    rmse_total = np.sqrt(np.mean(x_residuals**2 + y_residuals**2))
    
    # R² (per dimension)
    ss_res_x = np.sum(x_residuals**2)
    ss_tot_x = np.sum((x_exp - np.mean(x_exp))**2)
    r2_x = 1 - (ss_res_x / ss_tot_x)
    
    ss_res_y = np.sum(y_residuals**2)
    ss_tot_y = np.sum((y_exp - np.mean(y_exp))**2)
    r2_y = 1 - (ss_res_y / ss_tot_y)
    
    return {
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_total': rmse_total,
        'r2_x': r2_x,
        'r2_y': r2_y,
        'n_points': len(df)
    }