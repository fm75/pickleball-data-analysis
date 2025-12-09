import click
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from ballphysics.models import Pickleball, Environment
from ballphysics.analysis.tracking import simulate_trajectory_2d
from ballphysics.analysis.regression import cost_function_2d


@click.command()
@click.option('--folder', type=click.Path(exists=True), required=True, help='Data folder path')
@click.option('--filename', default='freefall.csv', help='Trajectory data file')
@click.option('--prefix', type=click.Choice(['none', 'new_', 'old_']), default='none', 
              help='File prefix, e.g. new_ is shortcut for new_freefall.csv and should have data for a new ball')
# @click.option('--ballspec', default='ballspecs.csv', help='Ball specifications file')
@click.option('--ballspec', help='Ball specifications file')
@click.option('--ball-mass', '--mass', '-m', type=float, default=None, help='Ball mass in grams (overrides ballspec)')
@click.option('--ball-diameter', '--diameter', '-d', type=float, default=None, help='Ball diameter in inches (overrides ballspec)')
@click.option('--temp-f', type=float, default=None, help='Temperature in Fahrenheit')
@click.option('--temp-c', type=float, default=20, help='Temperature in Celsius')
@click.option('--pressure', '-p', type=float, default=14.696, help='Pressure in psi')
@click.option('--curve', type=click.Choice(['1d', '2d', 'both']), default='2d', help='Fit type')
@click.option('--figure', type=click.Choice(['none', 'show', 'save', 'both']), default='show', help='Figure output, save writes to an image file')
def fit_drag_coefficient(folder, 
    filename, 
    prefix, 
    ballspec,
    ball_mass,
    ball_diameter, 
    temp_f, 
    temp_c, 
    pressure, 
    curve, 
    figure):
    """Fit coefficient of drag from pickleball drop test data."""
    
    # Handle prefix
    prefix_str = '' if prefix == 'none' else prefix
    
    print(f"ballspec {ballspec}")
    # Validate ball specification options
    if ballspec and (ball_mass or ball_diameter):
        raise click.UsageError("Cannot use --ball-mass or --ball-diameter with --ballspec")
    
    if not ballspec and not (ball_mass and ball_diameter):
        raise click.UsageError("Must provide either --ballspec OR both --ball-mass and --ball-diameter")

    # Extract ball properties (single row expected)
    if ball_spec:
        ball_df = pd.read_csv(folder_path / f'{prefix_str}{ballspec}')
        ball_row = ball_df.iloc[0]
        ball = Pickleball(mass_g=ball_row['mass_g'], radius_in=ball_row['diameter']/2)        
    else:
        ball = Pickleball(mass_g=ball_mass, radius_in=ball_diameter/2)        

    # Load data files
    folder_path = Path(folder)
    df = pd.read_csv(folder_path / f'{prefix_str}{filename}')
    
    
    # Create environment
    if temp_c and temp_f:
        raise click.UsageError("Use --temp-c, temp-f or neither, not both.")
    if temp_f: temp_c = 5 * (temp_f - 32) / 9
    
    env = Environment()  # TODO: adjust for temp/pressure
    
    # Calculate initial conditions using centered difference at index 1
    # ... (rest of the logic)
    
    click.echo(f"Fitted Cd = {best_Cd:.4f}")


if __name__ == '__main__':
    fit_drag_coefficient
