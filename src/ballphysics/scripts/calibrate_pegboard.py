#!/usr/bin/env python3
"""Calibrate spatial measurements using pegboard backdrop."""

import click
import cv2
import numpy as np
from pathlib import Path

from ballphysics.vision.calibration import calibrate_pegboard
from ballphysics.vision.detection import detect_holes
from ballphysics.vision.calibration import cluster_holes
from ballphysics.visualization.plotting import draw_holes, view_frame, show_frames_side_by_side
from ballphysics.vision.utils import extract_vertical_slice


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--frame-number', type=int, default=0, 
              help='Frame number to use (for video input, default: 0)')
@click.option('--slice-x-start', type=int, default=800,
              help='Starting x coordinate for vertical slice (default: 800)')
@click.option('--slice-x-end', type=int, default=1200,
              help='Ending x coordinate for vertical slice (default: 1200)')
@click.option('--n-clusters', type=int, default=6,
              help='Number of hole columns to detect (default: 6)')
@click.option('--angle-tolerance', type=float, default=1.0,
              help='Maximum acceptable pegboard angle in degrees (default: 1.0)')
@click.option('--visualize', is_flag=True,
              help='Show frame with detected holes')
@click.option('--visualize-detailed', is_flag=True,
              help='Show detailed visualization with original and slice')
def calibrate_pegboard_cli(input_path, frame_number, slice_x_start, slice_x_end, 
                           n_clusters, angle_tolerance, visualize, visualize_detailed):
    """
    Calibrate spatial measurements using pegboard backdrop.
    
    Accepts either image files or video files as INPUT_PATH.
    For video files, specify which frame to analyze with --frame-number.
    """
    input_path = Path(input_path)
    
    # Determine if input is image or video
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    is_image = input_path.suffix.lower() in image_extensions
    
    # Load frame
    if is_image:
        frame = cv2.imread(str(input_path))
        if frame is None:
            click.echo(f"Error: Could not load image {input_path}", err=True)
            return 1
        click.echo(f"Loaded image: {input_path}")
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            click.echo(f"Error: Could not open video {input_path}", err=True)
            return 1
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number >= total_frames:
            click.echo(f"Error: Frame {frame_number} exceeds video length {total_frames}", err=True)
            return 1
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            click.echo(f"Error: Could not read frame {frame_number}", err=True)
            return 1
        
        click.echo(f"Loaded frame {frame_number} from video: {input_path}")
    
    # Run calibration
    click.echo(f"\nRunning calibration with slice x=[{slice_x_start}:{slice_x_end}], {n_clusters} clusters...")
    
    result = calibrate_pegboard(
        frame,
        slice_x_start=slice_x_start,
        slice_x_end=slice_x_end,
        n_clusters=n_clusters,
        angle_tolerance=angle_tolerance
    )
    
    # Display results
    click.echo("\n=== Pegboard Calibration Results ===")
    click.echo(f"Status: {result.status.upper()}")
    click.echo(f"\nPixels per inch: {result.pixels_per_inch:.2f}")
    click.echo(f"Pegboard angle: {result.pegboard_angle:.2f}°")
    click.echo(f"Holes detected: {result.hole_count}")
    click.echo(f"Average hole radius: {result.avg_hole_radius:.2f} pixels")
    click.echo(f"  ({result.avg_hole_radius / result.pixels_per_inch:.3f} inches)")
    
    click.echo("\nValidation messages:")
    for msg in result.messages:
        click.echo(f"  • {msg}")
    
    if result.status == 'fail':
        click.echo("\n❌ Calibration FAILED", err=True)
        return 1
    elif result.status == 'warning':
        click.echo("\n⚠️  Calibration completed with WARNINGS")
    else:
        click.echo("\n✓ Calibration PASSED")
    
    # Visualization
    if visualize or visualize_detailed:
        # Re-run detection for visualization
        slice_frame = extract_vertical_slice(frame, slice_x_start, slice_x_end)
        circles = detect_holes(slice_frame)
        
        if circles is not None:
            labels = cluster_holes(circles, n_clusters)
            annotated_slice = draw_holes(slice_frame, circles, labels)
            
            if visualize_detailed:
                # Show original and annotated side-by-side
                show_frames_side_by_side(slice_frame, annotated_slice, 
                                        'Original Slice', 'Detected Holes')
            else:
                # Just show annotated
                view_frame(annotated_slice)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(calibrate_pegboard_cli())
