import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class BallDetection:
    cx: int | None
    cy: int | None
    radius: float | None


def detect_ball(frame: np.ndarray, 
                h_range: tuple[int, int] = (30, 80), 
                s_range: tuple[int, int] = (40, 255), 
                v_range: tuple[int, int] = (40, 255)) -> BallDetection:
    """
    Detect ball position and size using HSV color thresholding.
    
    Args:
        frame: BGR image from cv2
        h_range: Hue range (default for yellow pickleball)
        s_range: Saturation range
        v_range: Value range
    
    Returns:
        BallDetection with cx, cy, radius (None values if not detected)
    
    Note:
        Default HSV ranges were empirically determined for yellow pickleballs
        under indoor lighting. For other ball colors or lighting conditions,
        see docs/hsv_calibration.md for threshold selection guidance.
    """
    # Convert to HSV and create mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, 
                       np.array([h_range[0], s_range[0], v_range[0]]),
                       np.array([h_range[1], s_range[1], v_range[1]]))
    
    # Find centroid using moments
    moments = cv2.moments(mask)
    if moments['m00'] == 0:
        return BallDetection(None, None, None)
    
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    # Find radius using contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BallDetection(cx, cy, None)
    
    largest_contour = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(largest_contour)
    
    return BallDetection(cx, cy, radius)


def detect_holes(frame: np.ndarray, 
                 min_dist: int = 20,
                 param2: int = 7,
                 min_radius: int = 5,
                 max_radius: int = 8) -> np.ndarray | None:
    """
    Detect circular holes (e.g., pegboard) using Hough Circle Transform.
    
    Args:
        frame: Grayscale or BGR image
        min_dist: Minimum distance between detected circle centers
        param2: Accumulator threshold for circle detection (lower = more circles)
        min_radius: Minimum circle radius in pixels
        max_radius: Maximum circle radius in pixels
    
    Returns:
        Array of shape (N, 3) with (x, y, radius) for each detected circle,
        or None if no circles detected
    
    Note:
        Uses cv2.HoughCircles with HOUGH_GRADIENT method. For pegboard calibration,
        apply to a vertical or horizontal slice for best results. Detected holes
        can be clustered by column/row for spatial calibration (see cluster_holes).
        Default parameters tuned for 1/4" pegboard holes at typical camera distances.
        See docs/hough_calibration.md for parameter tuning guidance.
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=min_dist, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    
    if circles is None:
        return None
    
    return circles[0]  # Returns (N, 3) array

