import numpy as np

def rotate_points(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Apply 2D rotation transform to points.
    
    Args:
        points: Array of shape (N, 2) with (x, y) coordinates
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        Rotated points as array of shape (N, 2)
    
    Note:
        Used to correct for camera tilt when analyzing ball trajectories.
        Apply camera angle to transform observed motion to true vertical.
    """
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return points @ rotation_matrix.T