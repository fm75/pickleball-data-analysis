import numpy as np

def extract_vertical_slice(frame: np.ndarray, x_start: int, x_end: int) -> np.ndarray:
    """Extract a vertical slice from frame between x_start and x_end."""
    return frame[:, x_start:x_end]

def extract_horizontal_slice(frame: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
    """Extract a horizontal slice from frame between y_start and y_end."""
    return frame[y_start:y_end, :]

def extract_region(frame: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int) -> np.ndarray:
    """Extract a rectangular region from frame."""
    return frame[y_start:y_end, x_start:x_end]
    