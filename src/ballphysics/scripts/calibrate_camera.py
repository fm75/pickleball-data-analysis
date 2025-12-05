#!/usr/bin/env python3
"""Calculate camera tilt angle from ball free fall trajectory."""

import click
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ballphysics.vision.detection import detect_ball


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--start-frame', type=int, help='Starting frame number for ball detection')
@click.option('--end-frame', type=int, help='Ending frame number for ball detection')
def calibrate_camera(video_path, start_frame, end_frame):
    """
    Calculate camera tilt angle from ball trajectory.
    
    Analyzes ball position in two frames to determine horizontal drift
    relative to vertical fall, indicating camera tilt from true vertical.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        click.echo(f"Error: Could not open video file {video_path}", err=True)
        return 1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Auto-select frames if not provided
    if start_frame is None:
        start_frame = 11  # Default from your data
        click.echo(f"Using default start frame: {start_frame}")
    
    if end_frame is None:
        end_frame = min(63, total_frames - 1)  # Default from your data
        click.echo(f"Using default end frame: {end_frame}")
    
    # Validate frame numbers
    if start_frame >= end_frame:
        click.echo("Error: start_frame must be less than end_frame", err=True)
        return 1
    
    if end_frame >= total_frames:
        click.echo(f"Error: end_frame {end_frame} exceeds video length {total_frames}", err=True)
        return 1
    
    # Detect ball in start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame1 = cap.read()
    if not ret:
        click.echo(f"Error: Could not read frame {start_frame}", err=True)
        return 1
    
    detection1 = detect_ball(frame1)
    if detection1.cx is None or detection1.cy is None:
        click.echo(f"Error: Ball not detected in frame {start_frame}", err=True)
        return 1
    
    # Detect ball in end frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
    ret, frame2 = cap.read()
    if not ret:
        click.echo(f"Error: Could not read frame {end_frame}", err=True)
        return 1
    
    detection2 = detect_ball(frame2)
    if detection2.cx is None or detection2.cy is None:
        click.echo(f"Error: Ball not detected in frame {end_frame}", err=True)
        return 1
    
    cap.release()
    
    # Calculate camera angle
    horizontal_drift = detection2.cx - detection1.cx
    vertical_fall = detection2.cy - detection1.cy
    
    if vertical_fall <= 0:
        click.echo("Error: Ball did not fall (vertical_fall <= 0)", err=True)
        return 1
    
    angle_rad = np.arctan(horizontal_drift / vertical_fall)
    angle_deg = np.degrees(angle_rad)
    
    # Report results
    click.echo("\n=== Camera Calibration Results ===")
    click.echo(f"Video: {video_path}")
    click.echo(f"Frames analyzed: {start_frame} to {end_frame}")
    click.echo(f"\nBall position at frame {start_frame}: ({detection1.cx}, {detection1.cy})")
    click.echo(f"Ball position at frame {end_frame}: ({detection2.cx}, {detection2.cy})")
    click.echo(f"\nHorizontal drift: {horizontal_drift:.1f} pixels")
    click.echo(f"Vertical fall: {vertical_fall:.1f} pixels")
    click.echo(f"\nCamera tilt angle: {angle_deg:.2f}°")
    
    if abs(angle_deg) > 3.0:
        click.echo("\n⚠️  WARNING: Camera angle exceeds 3° - consider releveling", err=True)
    
    return 0


if __name__ == '__main__':
    sys.exit(calibrate_camera())
