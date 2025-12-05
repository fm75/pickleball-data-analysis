import numpy as np
from sklearn.cluster import KMeans

@dataclass
class SpacingStats:
    filtered_count: int
    rejected_count: int
    mode_estimate: float
    filtered_mean: float
    rejected_min: float
    rejected_max: float
    rejected_mean: float


@dataclass
class PegboardCalibration:
    pixels_per_inch: float
    pegboard_angle: float
    hole_count: int
    avg_hole_radius: float
    spacing_stats: SpacingStats
    status: str
    messages: list[str]
def calibrate_pegboard(frame: np.ndarray,
                       slice_x_start: int = 800,
                       slice_x_end: int = 1200,
                       n_clusters: int = 6,
                       angle_tolerance: float = 1.0,
                       hole_radius_range: tuple[float, float] = (0.10, 0.15),
                       **hough_params) -> PegboardCalibration:
    """
    Calibrate spatial measurements using pegboard backdrop.
    
    Returns calibration metrics and validation status.
    """
    # Extract slice and detect holes
    slice_frame = extract_vertical_slice(frame, slice_x_start, slice_x_end)
    circles = detect_holes(slice_frame, **hough_params)
    
    if circles is None:
        return PegboardCalibration(0, 0, 0, 0, 'fail', ['No holes detected'])
    
    # Cluster holes by column
    labels = cluster_holes(circles, n_clusters)
    
    # Calculate metrics
    angles = _fit_lines_to_clusters(circles, labels)
    spacings = _calculate_hole_spacings(circles, labels)
    pixels_per_inch = np.mean(spacings)
    pegboard_angle = np.mean(angles)
    avg_radius = np.mean(circles[:, 2])
    
    # Validate
    angle_status, angle_msgs = _validate_pegboard_angle(angles, angle_tolerance)
    dim_status, dim_msgs = _validate_physical_dimensions(avg_radius, pixels_per_inch, hole_radius_range)
    
    # Combine statuses: fail > warning > pass
    if angle_status == 'fail' or dim_status == 'fail':
        status = 'fail'
    elif angle_status == 'warning' or dim_status == 'warning':
        status = 'warning'
    else:
        status = 'pass'
    
    messages = angle_msgs + dim_msgs
    
    return PegboardCalibration(pixels_per_inch, pegboard_angle, 
                               len(circles), avg_radius, status, messages)


def cluster_holes(circles: np.ndarray, n_clusters: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster detected holes by column (or row) using K-means.
    
    Args:
        circles: Array of shape (N, 3) with (x, y, radius) from detect_holes
        n_clusters: Number of columns/rows to cluster into
    
    Returns:
        tuple of (labels, cluster_centers) where labels is array of cluster 
        assignments for each circle, cluster_centers is (n_clusters, 2) array
    
    Note:
        For vertical pegboard columns, cluster on x-coordinates.
        For horizontal rows, cluster on y-coordinates.
        n_clusters doesn't need to match actual columns exactly - even with
        fewer clusters, linear fitting to each cluster can work well.
    """
    # Extract x coordinates for clustering (use y for horizontal rows)
    X = circles[:, 0].reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    return labels


def _validate_physical_dimensions(avg_radius: float, 
                                  pixels_per_inch: float, 
                                  hole_radius_range: tuple[float, float]) -> tuple[str, list[str]]:
    """
    Validate that hole size makes physical sense.
    
    Returns (status, messages) where status is 'pass', 'warning', or 'fail'.
    """
    radius_inches = avg_radius / pixels_per_inch
    min_radius, max_radius = hole_radius_range
    messages = []
    
    if radius_inches < 0.5 * min_radius or radius_inches > 2.0 * max_radius:  # Hard fail
        status = 'fail'
        messages.append(f'Hole radius {radius_inches:.3f}" is unreasonable (expected {min_radius:.2f}-{max_radius:.2f}")')
    elif radius_inches < min_radius or radius_inches > max_radius:  # Warning
        status = 'warning'
        messages.append(f'Hole radius {radius_inches:.3f}" outside expected range ({min_radius:.2f}-{max_radius:.2f}")')
    else:
        status = 'pass'
        messages.append(f'Hole radius {radius_inches:.3f}" within expected range')
    
    return status, messages


def _validate_pegboard_angle(angles: np.ndarray, tolerance: float) -> tuple[str, list[str]]:
    """
    Validate that pegboard angle is within tolerance of vertical.
    
    Returns (status, messages) where status is 'pass', 'warning', or 'fail'.
    """
    max_angle = np.max(np.abs(angles))
    mean_angle = np.mean(np.abs(angles))
    messages = []
    
    if max_angle > 3 * tolerance:  # Hard fail
        status = 'fail'
        messages.append(f'Pegboard angle {max_angle:.2f}° exceeds hard limit ({3*tolerance:.1f}°)')
    elif max_angle > tolerance:  # Warning
        status = 'warning'
        messages.append(f'Pegboard angle {max_angle:.2f}° exceeds tolerance ({tolerance:.1f}°)')
    else:
        status = 'pass'
        messages.append(f'Pegboard angle {mean_angle:.2f}° within tolerance')
    
    return status, messages


def _fit_lines_to_clusters(circles: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Fit a line to each cluster of holes and return the angle from vertical.
    
    Returns array of angles in degrees for each cluster.
    """
    angles = []
    
    for cluster_id in np.unique(labels):
        cluster_circles = circles[labels == cluster_id]
        positions = cluster_circles[:, :2]
        
        # Linear fit: y = mx + b
        x = positions[:, 0]
        y = positions[:, 1]
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Angle from vertical (vertical line has infinite slope)
        # tan(angle) = 1/slope for angle from vertical
        angle = np.degrees(np.arctan(1 / slope)) if slope != 0 else 0
        angles.append(angle)
    
    return np.array(angles)