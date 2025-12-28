
# Detailed Code Breakdown: From Pixels to Physics with Homography

This document outlines the process of using a **Homography** to transform pixel coordinates from a 2D image into real-world physical coordinates on a plane. This method is a powerful step up from a simple `pixels_per_inch` calculation because it correctly accounts for perspective distortion (i.e., objects farther away appearing smaller) and lens distortion.

### What is a Homography?

In computer vision, a homography is a 3x3 transformation matrix, often denoted as $H$, that maps points from one plane to another. In our case, we want to map points from the **image plane** (measured in pixels) to the **world plane** (the physical board where the ball is dropping, measured in inches or millimeters).

To calculate this matrix, we need a set of at least four points for which we know the coordinates in *both* planes. This is why calibration targets like ArUco markers are so useful.

### The Workflow

The process can be broken down into three main stages:

1.  **Calibration Data Gathering:**
    *   Physically place at least four non-collinear markers (e.g., ArUco markers) on the plane of motion (your backboard).
    *   Measure the precise physical locations of these markers. This is your "ground truth." For example, you could place them at the corners of a 20-inch by 60-inch rectangle. These are your **world coordinates**.
    *   In your code, detect these same markers in a video frame and record their center points. These are your **pixel coordinates**.

2.  **Calculating the Homography Matrix ($H$):**
    *   Use OpenCV's `cv2.findHomography()` function. This function takes your list of world coordinates and the corresponding list of pixel coordinates.
    *   It computes the optimal $H$ matrix that describes the projective transformation between these two sets of points.

3.  **Applying the Transformation:**
    *   Once you have $H$, you can use it to transform any pixel coordinate (like the detected center of the ball) into its corresponding real-world coordinate on the board.
    *   **Important:** The matrix $H$ calculated in the previous step maps `world -> image`. To go from `image -> world`, we need to calculate and use its inverse, $H^{-1}$.

---

### Step-by-Step Code Implementation

Here is a full example demonstrating the process.

```python
import numpy as np
import cv2

# --- Step 1: Define Calibration Points ---

# Define the real-world coordinates of your 4 ArUco markers.
# Let's assume you placed them at the corners of a 2-foot by 5-foot rectangle.
# We'll use millimeters for precision to avoid small floating point numbers.
# (0,0) is the top-left corner.
# 2 feet = 609.6 mm
# 5 feet = 1524.0 mm
world_coords = np.array([
    [0.0, 0.0],         # Top-left marker
    [609.6, 0.0],        # Top-right marker
    [609.6, 1524.0],     # Bottom-right marker
    [0.0, 1524.0]        # Bottom-left marker
], dtype="float32")

# Define the corresponding pixel coordinates where these markers were detected in the image.
# You would get these values from your ArUco detection function.
# NOTE: These are example values and must be replaced with your actual detected coordinates.
pixel_coords = np.array([
    [986, 437],          # Top-left pixel coord
    [1614, 442],         # Top-right pixel coord
    [1624, 1890],        # Bottom-right pixel coord
    [978, 1883]          # Bottom-left pixel coord
], dtype="float32")


# --- Step 2: Calculate the Homography Matrix ---

# cv2.findHomography calculates the matrix that maps points from the first
# set (world_coords) to the second set (pixel_coords).
# So, H maps WORLD -> PIXEL.
H, status = cv2.findHomography(world_coords, pixel_coords)

# To map from PIXEL -> WORLD, we need the inverse of H.
try:
    H_inv = np.linalg.inv(H)
    print("Successfully calculated inverse homography matrix H_inv:")
    print(H_inv)
except np.linalg.LinAlgError:
    print("Error: Could not compute the inverse of H. Check if your points are collinear.")
    H_inv = None


# --- Step 3: Apply the Transformation ---


# --- Example Usage ---

# Let's say your ball detection algorithm found the ball at pixel (1250, 980)
ball_pixel_location = (1250, 980)

# Use the function to get its real-world physical location
ball_world_location = transform_pixel_to_world(ball_pixel_location, H_inv)

if ball_world_location:
    print(f"\nBall at pixel {ball_pixel_location} corresponds to world location:")
    # We print with formatting to make it readable
    print(f"  X: {ball_world_location[0]:.2f} mm")
    print(f"  Y: {ball_world_location[1]:.2f} mm")

# You would run this `transform_pixel_to_world` function on the ball's
# coordinates for every frame to get a time series of its physical position.
# From this, you can accurately calculate physical velocity and acceleration.
