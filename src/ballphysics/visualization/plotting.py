import cv2
import numpy as np

def draw_ball(frame: np.ndarray, cx: int | None, cy: int | None, radius: float | None, 
              color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw detected ball on frame.
    
    Args:
        frame: BGR image
        cx, cy: Ball center coordinates (None if not detected)
        radius: Ball radius (None if not detected)
        color: BGR color tuple for drawing
        thickness: Line thickness for circle
    
    Returns:
        Frame with ball drawn (original frame if ball not detected)
    """
    result = frame.copy()
    
    if cx is not None and cy is not None:
        cv2.circle(result, (cx, cy), 5, color, -1)  # Center point
        if radius is not None:
            cv2.circle(result, (cx, cy), int(radius), color, thickness)
    
    return result
def draw_holes(frame: np.ndarray, circles: np.ndarray, labels: np.ndarray | None = None,
               colors: list[tuple[int, int, int]] | None = None, thickness: int = 2) -> np.ndarray:
    """
    Draw detected holes on frame, optionally color-coded by cluster.
    
    Args:
        frame: BGR image
        circles: Array of shape (N, 3) with (x, y, radius)
        labels: Optional cluster labels for color-coding
        colors: List of BGR colors for each cluster (auto-generated if None)
        thickness: Line thickness for circles
    
    Returns:
        Frame with holes drawn
    """
    result = frame.copy()
    
    if labels is None:
        # Draw all holes in same color
        color = (0, 255, 0)
        for x, y, r in circles:
            cv2.circle(result, (int(x), int(y)), int(r), color, thickness)
    else:
        # Color-code by cluster
        n_clusters = len(np.unique(labels))
        if colors is None:
            # Generate distinct colors
            colors = [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(n_clusters)]
        
        for i, (x, y, r) in enumerate(circles):
            color = colors[labels[i]]
            cv2.circle(result, (int(x), int(y)), int(r), color, thickness)
    
    return result


import matplotlib.pyplot as plt

def view_frame(frame: np.ndarray, h: int = 10, w: int = 12) -> None:
    """
    Display a single frame with axis ticks on top and bottom.
    
    Args:
        frame: BGR image from cv2
        h: Figure height in inches
        w: Figure width in inches
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(h, w))
    plt.imshow(frame_rgb)
    plt.gca().xaxis.set_ticks_position('both')
    plt.tick_params(top=True, labeltop=True)
    plt.show()


def show_frames_side_by_side(frame1: np.ndarray, frame2: np.ndarray, 
                             title1: str = 'Frame 1', title2: str = 'Frame 2') -> None:
    """
    Display two frames side-by-side for comparison.
    
    Args:
        frame1: First BGR image from cv2
        frame2: Second BGR image from cv2
        title1: Title for first frame
        title2: Title for second frame
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(frame1_rgb)
    ax1.set_title(title1)
    ax1.axis('on')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(top=True, labeltop=True)

    ax2.imshow(frame2_rgb)
    ax2.set_title(title2)
    ax2.axis('on')
    ax2.xaxis.set_ticks_position('both')
    ax2.tick_params(top=True, labeltop=True)
    
    plt.tight_layout()
    plt.show()


