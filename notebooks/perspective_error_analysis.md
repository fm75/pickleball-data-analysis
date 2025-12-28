
# Analysis of Perspective Error in Photogrammetry

This document details the estimation of measurement errors arising from perspective distortion when an object's motion is not perfectly coplanar with the calibration reference plane.

### The Scenario: The Depth Problem

This error occurs because the object being tracked (a falling ball) is not moving on the exact same plane as the calibration markers (the ArUco markers on the board). The calculated `pixels_per_inch` ratio is the ground truth *at the surface of the board*, but the ball is closer to the camera, making it appear slightly larger.

We can model this effect using the principles of similar triangles, which form the basis of pinhole camera models and perspective projection.

#### Assumptions for the Model

*   **Camera Distance to Board (`D_board`):** Approximately 18 feet, which is `18 * 12 = 216` inches.
*   **Ball-to-Board Distance (`d`):** A maximum of 3 inches. This is the "depth" of the measurement volume.
*   **Camera Position:** Assumed to be reasonably centered on the board's area of interest.

#### The Logic

The apparent size of an object in an image is inversely proportional to its distance from the camera's optical center. This means that our `pixels_per_inch` conversion factor is not constant throughout the 3D space; it changes with depth.

1.  **Calibration Plane Distance:** The ArUco markers are at a distance of $D_{\text{board}} = 216$ inches from the camera sensor. The `pixels_per_inch` value calculated from these markers is valid only for objects on this specific plane. Let's call this value $\text{PPI}_{	ext{board}}$.

2.  **Ball's Plane Distance:** The ball, being up to 3 inches in front of the board, is closer to the camera. Its distance is $D_{\text{ball}} = 216 - 3 = 213$ inches from the sensor.

3.  **The Error Factor:** The ratio of the *true* `pixels_per_inch` at the ball's location ($\text{PPI}_{\text{ball}}$) to the one measured at the board ($\text{PPI}_{\text{board}}$) is equal to the inverse ratio of their distances from the camera:

    $$ \frac{\text{PPI}_{\text{ball}}}{\text{PPI}_{\text{board}}} = \frac{D_{\text{board}}}{D_{\text{ball}}} $$

    Plugging in our assumed values:

    $$ \text{Error Factor} = \frac{216 \text{ inches}}{213 \text{ inches}} \approx 1.01408 $$

#### Estimated Correction Size

The result of `1.014` means that because the ball is 3 inches closer to the camera, it appears **1.4% larger** in the image than it would if it were moving on the surface of the board.

*   **Impact on Measurement:** When you use your $\text{PPI}_{\text{board}}$ value to convert the ball's movement from pixels to inches, you are using a conversion factor that is about 1.4% too small. This will cause you to **overestimate the physical distance the ball travels by approximately 1.4%**.

*   **Example Calculation:**
    *   Let's assume the ball traveled **3300 pixels** vertically.
    *   Let's assume your measured `pixels_per_inch` from the board was **42.0**.
    *   **Your current calculation:** `3300 / 42.0 = 78.57` inches.
    *   **Corrected calculation:** `3300 / (42.0 * 1.014) = 3300 / 42.59 = 77.48` inches.
    *   The difference is **1.09 inches** over a ~78-inch fall.

This systematic 1.4% overestimation would apply to all your derived physical quantities, including velocity and acceleration. While the *shape* of the data curve would be correct, the absolute physical scale would be off by this factor.
