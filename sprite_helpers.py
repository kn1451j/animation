#!/usr/bin/env python3
"""
Affine Transform Calculator for 2.5D Sprite Walker

This script computes the exact screen-space affine transforms
for sprite positioning based on Blender camera and motion data.

Usage:
    python3 compute_transforms.py

Output:
    - Screen positions for each step
    - Affine transform matrices for each step count
    - C++ code snippet for the transforms
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# Configuration - UPDATE THESE WITH YOUR BLENDER DATA
# ============================================================================

@dataclass
class CameraConfig:
    """Camera parameters from Blender"""
    position: np.ndarray = None
    euler_xyz_deg: np.ndarray = None  # Rotation in degrees (XYZ order)
    focal_length_mm: float = 50.0
    sensor_width_mm: float = 36.0     # Full frame
    sensor_height_mm: float = 20.25
    image_width: int = 1920
    image_height: int = 1080
    
    def __post_init__(self):
        if self.position is None:
            # Your camera position
            self.position = np.array([-13.923, -10.286, 2.4484])
        if self.euler_xyz_deg is None:
            # Your camera rotation
            self.euler_xyz_deg = np.array([85.574, 2.1905, -64.732])

@dataclass
class StepConfig:
    """Step motion data from Blender"""
    # World coordinates for walking RIGHT (sprite moves right on screen)
    start_pos: np.ndarray = None
    left_foot_end: np.ndarray = None
    right_foot_end: np.ndarray = None
    
    def __post_init__(self):
        if self.start_pos is None:
            self.start_pos = np.array([0.0, 0.0, 0.0])
        if self.left_foot_end is None:
            # After left foot step
            self.left_foot_end = np.array([-0.040788, 0.170629, -1.21896])
        if self.right_foot_end is None:
            # After right foot step (one full cycle)
            self.right_foot_end = np.array([-0.072024, 0.174482, -1.98306])

# ============================================================================
# Math Functions
# ============================================================================

def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix around X axis"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """Rotation matrix around Y axis"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix around Z axis"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def euler_to_rotation_matrix(euler_xyz_deg: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (XYZ order, degrees) to rotation matrix.
    This matches Blender's default Euler rotation order.
    """
    rad = np.deg2rad(euler_xyz_deg)
    Rx = rotation_matrix_x(rad[0])
    Ry = rotation_matrix_y(rad[1])
    Rz = rotation_matrix_z(rad[2])
    # XYZ order: first X, then Y, then Z
    return Rz @ Ry @ Rx

class CameraProjector:
    """Projects 3D world points to 2D screen coordinates"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        
        # Compute rotation matrix (world to camera)
        R_cam = euler_to_rotation_matrix(config.euler_xyz_deg)
        self.R_world_to_cam = R_cam.T  # Transpose = inverse for rotation
        
        # Focal lengths in pixels
        self.fx = config.focal_length_mm * config.image_width / config.sensor_width_mm
        self.fy = config.focal_length_mm * config.image_height / config.sensor_height_mm
        
    def project(self, world_point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Project a 3D world point to normalized screen coordinates.
        
        Returns:
            (screen_xy, depth) where screen_xy is in [0,1] range
        """
        # Transform to camera space
        p_cam = self.R_world_to_cam @ (world_point - self.config.position)
        
        # Depth (Blender camera looks down -Z)
        depth = -p_cam[2]
        if depth <= 0.001:
            depth = 0.001  # Avoid division by zero
        
        # Perspective projection to image plane
        x_img = (self.fx * p_cam[0]) / depth
        y_img = (self.fy * p_cam[1]) / depth
        
        # Convert to normalized screen coordinates
        # Origin at bottom-left, Y up (OpenGL convention)
        x_screen = (x_img + self.config.image_width / 2) / self.config.image_width
        y_screen = 1.0 - (y_img + self.config.image_height / 2) / self.config.image_height
        
        return np.array([x_screen, y_screen]), depth

# ============================================================================
# Transform Computation
# ============================================================================

@dataclass
class AffineTransform2D:
    """2D Affine transformation matrix"""
    a: float = 1.0   # scale_x
    b: float = 0.0   # shear_x
    c: float = 0.0   # shear_y
    d: float = 1.0   # scale_y
    tx: float = 0.0  # translate_x
    ty: float = 0.0  # translate_y
    
    def __str__(self):
        return f"[{self.a:.6f}, {self.b:.6f}, {self.tx:.6f}]\n" \
               f"[{self.c:.6f}, {self.d:.6f}, {self.ty:.6f}]\n" \
               f"[0, 0, 1]"
    
    def to_cpp_code(self, var_name: str = "transform") -> str:
        return f"AffineTransform2D {var_name}({self.a}f, {self.b}f, {self.c}f, {self.d}f, {self.tx}f, {self.ty}f);"
    
    def compose(self, other: 'AffineTransform2D') -> 'AffineTransform2D':
        """Compose this transform with another (this * other)"""
        return AffineTransform2D(
            a = self.a * other.a + self.b * other.c,
            b = self.a * other.b + self.b * other.d,
            c = self.c * other.a + self.d * other.c,
            d = self.c * other.b + self.d * other.d,
            tx = self.a * other.tx + self.b * other.ty + self.tx,
            ty = self.c * other.tx + self.d * other.ty + self.ty
        )

def compute_step_transforms(cam_config: CameraConfig, step_config: StepConfig):
    """Compute all the affine transforms for the step animation system"""
    
    projector = CameraProjector(cam_config)
    
    # Project key positions
    start_screen, start_depth = projector.project(step_config.start_pos)
    left_screen, left_depth = projector.project(step_config.left_foot_end)
    right_screen, right_depth = projector.project(step_config.right_foot_end)
    
    print("=" * 60)
    print("SCREEN POSITIONS (normalized 0-1)")
    print("=" * 60)
    print(f"Start position:      ({start_screen[0]:.6f}, {start_screen[1]:.6f}), depth: {start_depth:.4f}")
    print(f"After left foot:     ({left_screen[0]:.6f}, {left_screen[1]:.6f}), depth: {left_depth:.4f}")
    print(f"After right foot:    ({right_screen[0]:.6f}, {right_screen[1]:.6f}), depth: {right_depth:.4f}")
    print()
    
    # Compute deltas
    left_step_delta = left_screen - start_screen
    right_step_delta = right_screen - left_screen
    full_cycle_delta = right_screen - start_screen
    
    print("SCREEN DELTAS")
    print("-" * 40)
    print(f"Left foot step:  dx={left_step_delta[0]:.6f}, dy={left_step_delta[1]:.6f}")
    print(f"Right foot step: dx={right_step_delta[0]:.6f}, dy={right_step_delta[1]:.6f}")
    print(f"Full cycle:      dx={full_cycle_delta[0]:.6f}, dy={full_cycle_delta[1]:.6f}")
    print()
    
    # Compute scale changes due to perspective
    left_scale = start_depth / left_depth
    right_scale = left_depth / right_depth
    cycle_scale = start_depth / right_depth
    
    print("PERSPECTIVE SCALE CHANGES")
    print("-" * 40)
    print(f"After left foot:  scale = {left_scale:.6f} (closer = larger)")
    print(f"After right foot: scale = {right_scale:.6f}")
    print(f"After full cycle: scale = {cycle_scale:.6f}")
    print()
    
    # Create transforms
    print("=" * 60)
    print("AFFINE TRANSFORMS")
    print("=" * 60)
    
    # Transform after left foot step only (half cycle)
    T_left = AffineTransform2D(
        a=left_scale, d=left_scale,
        tx=left_step_delta[0], ty=left_step_delta[1]
    )
    print("\nAfter LEFT foot step (walking right):")
    print(T_left)
    
    # Transform for one complete cycle (left + right foot)
    T_cycle = AffineTransform2D(
        a=cycle_scale, d=cycle_scale,
        tx=full_cycle_delta[0], ty=full_cycle_delta[1]
    )
    print("\nAfter FULL CYCLE (left + right foot):")
    print(T_cycle)
    
    # Compute transforms for multiple cycles
    print("\n" + "=" * 60)
    print("TRANSFORMS BY STEP COUNT (walking right)")
    print("=" * 60)
    
    transforms = [AffineTransform2D()]  # Step 0 = identity
    current = AffineTransform2D()
    
    for step in range(1, 7):  # 6 steps max
        if step % 2 == 1:
            # Odd step = left foot
            # Apply left foot delta and scale
            half_cycle = AffineTransform2D(
                a=left_scale, d=left_scale,
                tx=left_step_delta[0], ty=left_step_delta[1]
            )
            current = half_cycle.compose(current)
        else:
            # Even step = right foot, completing a cycle
            # Apply right foot delta and scale
            right_only = AffineTransform2D(
                a=right_scale, d=right_scale,
                tx=right_step_delta[0], ty=right_step_delta[1]
            )
            current = right_only.compose(current)
        
        transforms.append(current)
        print(f"\nStep {step}:")
        print(current)
    
    # Generate C++ code
    print("\n" + "=" * 60)
    print("C++ CODE")
    print("=" * 60)
    print()
    print("// Step transforms for walking right")
    print("// Apply these to the sprite's base transform based on steps completed")
    print()
    print(f"const float LEFT_STEP_DX = {left_step_delta[0]:.6f}f;")
    print(f"const float LEFT_STEP_DY = {left_step_delta[1]:.6f}f;")
    print(f"const float LEFT_STEP_SCALE = {left_scale:.6f}f;")
    print()
    print(f"const float RIGHT_STEP_DX = {right_step_delta[0]:.6f}f;")
    print(f"const float RIGHT_STEP_DY = {right_step_delta[1]:.6f}f;")
    print(f"const float RIGHT_STEP_SCALE = {right_scale:.6f}f;")
    print()
    print(f"const float CYCLE_DX = {full_cycle_delta[0]:.6f}f;")
    print(f"const float CYCLE_DY = {full_cycle_delta[1]:.6f}f;")
    print(f"const float CYCLE_SCALE = {cycle_scale:.6f}f;")
    print()
    print("// Pre-computed transforms for each step count")
    print("AffineTransform2D stepTransformsRight[] = {")
    for i, t in enumerate(transforms):
        print(f"    /* Step {i} */ AffineTransform2D({t.a:.6f}f, {t.b:.6f}f, {t.c:.6f}f, {t.d:.6f}f, {t.tx:.6f}f, {t.ty:.6f}f),")
    print("};")
    
    return transforms

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("AFFINE TRANSFORM CALCULATOR FOR 2.5D SPRITE WALKER")
    print("=" * 60)
    print()
    
    # Initialize configs with your data
    cam_config = CameraConfig()
    step_config = StepConfig()
    
    print("CAMERA CONFIGURATION")
    print("-" * 40)
    print(f"Position: {cam_config.position}")
    print(f"Rotation (deg): {cam_config.euler_xyz_deg}")
    print(f"Focal length: {cam_config.focal_length_mm}mm")
    print(f"Resolution: {cam_config.image_width}x{cam_config.image_height}")
    print()
    
    print("STEP MOTION DATA")
    print("-" * 40)
    print(f"Start:           {step_config.start_pos}")
    print(f"After left foot: {step_config.left_foot_end}")
    print(f"After right foot:{step_config.right_foot_end}")
    print()
    
    # Compute transforms
    transforms = compute_step_transforms(cam_config, step_config)
    
    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print()
    print("To verify these values:")
    print("1. Render frame 1 and frame 87 from Blender")
    print("2. Overlay them in an image editor")
    print("3. Measure the pixel offset of the sprite's anchor point")
    print("4. Compare with: dx * image_width, dy * image_height")
    print()
    
    # Convert to pixels for easy comparison
    w, h = cam_config.image_width, cam_config.image_height
    t1 = transforms[1]  # After first step
    print(f"Expected pixel offset after step 1:")
    print(f"  X: {t1.tx * w:.1f} pixels")
    print(f"  Y: {t1.ty * h:.1f} pixels")
    print(f"  Scale: {t1.a:.4f}")

if __name__ == "__main__":
    main()