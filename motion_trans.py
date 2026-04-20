import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# 1. INTRINSIC MATRIX K
# ============================================================
# Blender defaults
focal_length_mm = 50.0
sensor_width_mm = 36.0
res_x = 1920
res_y = 1080

# Focal length in pixels
f_x = focal_length_mm * res_x / sensor_width_mm  # 50 * 1920 / 36 = 2666.667
f_y = f_x  # square pixels

# Principal point (center of image)
c_x = res_x / 2.0  # 960
c_y = res_y / 2.0  # 540

K = np.array([
    [f_x,  0,   c_x],
    [0,    f_y, c_y],
    [0,    0,   1  ]
])

print("=" * 60)
print("INTRINSIC MATRIX K")
print("=" * 60)
print(f"Focal length (mm): {focal_length_mm}")
print(f"Sensor width (mm): {sensor_width_mm}")
print(f"Resolution: {res_x} x {res_y}")
print(f"f_x = f_y = {f_x:.4f} px")
print(f"c_x = {c_x}, c_y = {c_y}")
print()
print("K =")
print(K)
print()

# ============================================================
# 2. EXTRINSIC MATRIX [R | t]
# ============================================================
# Blender convention: XYZ Euler (intrinsic) = ZYX extrinsic
# Given Euler angles in degrees
rx_deg = 84.574
ry_deg = -2.1905
rz_deg = 64.732

rx = np.radians(rx_deg)
ry = np.radians(ry_deg)
rz = np.radians(rz_deg)

# Individual rotation matrices
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(rx), -np.sin(rx)],
    [0, np.sin(rx),  np.cos(rx)]
])

Ry = np.array([
    [ np.cos(ry), 0, np.sin(ry)],
    [ 0,          1, 0         ],
    [-np.sin(ry), 0, np.cos(ry)]
])

Rz = np.array([
    [np.cos(rz), -np.sin(rz), 0],
    [np.sin(rz),  np.cos(rz), 0],
    [0,           0,           1]
])

# Blender XYZ intrinsic: R = Rx * Ry * Rz
R_blender = Rx @ Ry @ Rz

# Camera position in world coordinates
cam_pos = np.array([13.923, -10.286, 2.4484])

# ---- Blender to OpenCV conversion ----
# Blender camera looks down -Z_local with Y_local up
# OpenCV camera looks down +Z_local with Y_local down
# Conversion: flip Y and Z axes of the camera frame
C_b2cv = np.diag([1, -1, -1])

R_cv = C_b2cv @ R_blender  # rotation: world -> OpenCV camera frame

# Translation: t = -R * C (camera center)
t_cv = -R_cv @ cam_pos

print("=" * 60)
print("EXTRINSIC PARAMETERS")
print("=" * 60)
print(f"Euler angles (deg): rx={rx_deg}, ry={ry_deg}, rz={rz_deg}")
print(f"Camera position (world): {cam_pos}")
print()
print("R_blender (world->blender cam) =")
print(R_blender)
print()
print("R_cv (world->OpenCV cam) =")
print(R_cv)
print()
print("t_cv =")
print(t_cv)
print()

# Extrinsic matrix [R|t]  (3x4)
Rt = np.hstack([R_cv, t_cv.reshape(3, 1)])
print("Extrinsic [R|t] (3x4) =")
print(Rt)
print()

# ============================================================
# 3. FULL PROJECTION MATRIX P = K [R|t]
# ============================================================
P = K @ Rt

print("=" * 60)
print("PROJECTION MATRIX P = K [R|t]  (3x4)")
print("=" * 60)
print(P)
print()

# ============================================================
# 4. PROJECT SPRITE CENTERS
# ============================================================
def project_point(P, X_world):
    """Project a 3D world point to pixel coordinates."""
    X_h = np.append(X_world, 1.0)  # homogeneous
    p = P @ X_h
    return p[0] / p[2], p[1] / p[2]

# Frame A: sprite center
center_A = np.array([-0.0185, 0.003, 0.088])
# Frame C: sprite center after -2m along Y
center_C = np.array([-0.0185, 0.003 - 2.0, 0.088])

px_A = project_point(P, center_A)
px_C = project_point(P, center_C)

print("=" * 60)
print("PROJECTED SPRITE CENTERS")
print("=" * 60)
print(f"Sprite center A (world): {center_A}")
print(f"  -> Pixel: ({px_A[0]:.2f}, {px_A[1]:.2f})")
print()
print(f"Sprite center C (world): {center_C}")
print(f"  -> Pixel: ({px_C[0]:.2f}, {px_C[1]:.2f})")
print()

# ============================================================
# 5. HOMOGRAPHY: Frame A -> Frame C
# ============================================================
# The sprite is a planar quad in the XZ plane (normal along Y).
# In frame A, the plane equation is: Y = 0.003
# In frame C, the plane equation is: Y = -1.997
#
# For a single camera viewing a plane, the homography between
# two views of the plane (at two different Y positions) is:
#
# The sprite surface points in frame A:
#   X_A = (x, 0.003, z) for x in [center_x ± w/2], z in [center_z ± h/2]
#
# The sprite surface points in frame C:
#   X_C = (x, -1.997, z) — same x, z but different y
#
# So the 3D transformation from A-surface to C-surface is:
#   X_C = X_A + (0, -2, 0)
#
# Homography derivation:
# For points on the sprite plane in frame A: n^T * X = d
# The plane (in world coords) has normal n = (0, 1, 0), d = 0.003
#
# The 3D displacement is pure translation: T = (0, -2, 0)
# H = K * (R - t_offset * n^T / d_plane) * K^{-1}  -- but this is
# for inter-frame homographies. Here we have ONE camera, TWO plane positions.
#
# Simpler approach: compute H from 4 corresponding point pairs.

# Sprite corners in world coordinates
w = 0.686
h = 2.87

def sprite_corners(center):
    """Return 4 corners of the sprite (XZ plane quad)."""
    cx, cy, cz = center
    return np.array([
        [cx - w/2, cy, cz - h/2],  # bottom-left
        [cx + w/2, cy, cz - h/2],  # bottom-right
        [cx + w/2, cy, cz + h/2],  # top-right
        [cx - w/2, cy, cz + h/2],  # top-left
    ])

corners_A_3d = sprite_corners(center_A)
corners_C_3d = sprite_corners(center_C)

# Project corners to pixel coordinates
corners_A_px = np.array([project_point(P, c) for c in corners_A_3d])
corners_C_px = np.array([project_point(P, c) for c in corners_C_3d])

print("=" * 60)
print("SPRITE CORNERS (Pixels)")
print("=" * 60)
print("Frame A corners (BL, BR, TR, TL):")
for i, (a, c) in enumerate(zip(corners_A_px, corners_C_px)):
    print(f"  Corner {i}: A=({a[0]:.2f}, {a[1]:.2f})  C=({c[0]:.2f}, {c[1]:.2f})")
print()

# Compute homography using DLT (Direct Linear Transform)
def compute_homography(src_pts, dst_pts):
    """Compute 3x3 homography H such that dst ~ H * src (homogeneous coords)."""
    assert src_pts.shape == dst_pts.shape == (4, 2)
    A = []
    for i in range(4):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]  # normalize
    return H

H = compute_homography(corners_A_px, corners_C_px)

print("=" * 60)
print("HOMOGRAPHY H (maps Frame A pixels -> Frame C pixels)")
print("=" * 60)
print("H =")
print(H)
print()

# ---- Verification ----
print("=" * 60)
print("VERIFICATION")
print("=" * 60)
print("Mapping A corners through H and comparing to C corners:")
for i in range(4):
    src_h = np.array([corners_A_px[i][0], corners_A_px[i][1], 1.0])
    dst_h = H @ src_h
    dst = dst_h[:2] / dst_h[2]
    expected = corners_C_px[i]
    err = np.linalg.norm(dst - expected)
    print(f"  Corner {i}: H(A)=({dst[0]:.2f}, {dst[1]:.2f})  "
          f"C=({expected[0]:.2f}, {expected[1]:.2f})  error={err:.4f} px")
print()

# Verify center point mapping
center_A_h = np.array([px_A[0], px_A[1], 1.0])
center_C_mapped = H @ center_A_h
center_C_mapped = center_C_mapped[:2] / center_C_mapped[2]
print(f"Center A pixel: ({px_A[0]:.2f}, {px_A[1]:.2f})")
print(f"H(center A):    ({center_C_mapped[0]:.2f}, {center_C_mapped[1]:.2f})")
print(f"Center C pixel: ({px_C[0]:.2f}, {px_C[1]:.2f})")
print(f"Error: {np.linalg.norm(center_C_mapped - np.array(px_C)):.4f} px")
print()

# ---- Check if H is affine ----
print("=" * 60)
print("AFFINITY CHECK")
print("=" * 60)
print(f"Last row of H: [{H[2,0]:.8f}, {H[2,1]:.8f}, {H[2,2]:.8f}]")
print(f"If H is affine, last row should be [0, 0, 1] (or proportional).")
print(f"H[2,0] / H[2,2] = {H[2,0]/H[2,2]:.8f}")
print(f"H[2,1] / H[2,2] = {H[2,1]/H[2,2]:.8f}")
if abs(H[2,0]) < 1e-4 and abs(H[2,1]) < 1e-4:
    print("-> H is approximately AFFINE.")
else:
    print("-> H is a general PROJECTIVE homography (not purely affine).")
    print("   This is expected: a pure translation in 3D on a plane viewed")
    print("   in perspective generally yields a projective homography,")
    print("   not strictly affine, unless the plane is fronto-parallel.")