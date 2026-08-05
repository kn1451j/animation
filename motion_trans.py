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
rx_deg_left = 84.574
ry_deg_left = -2.1905
rz_deg_left = 64.732

rx_deg_right = 84.574
ry_deg_right = -2.1905
rz_deg_right = -64.732


def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
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
    return Rx @ Ry @ Rz


# Camera position in world coordinates
cam_pos_left = np.array([13.923, -10.286, 2.4484])
cam_pos_right = np.array([-13.923, -10.286, 2.4484])

# ---- Blender to OpenCV conversion ----
# Blender camera looks down -Z_local with Y_local up
# OpenCV camera looks down +Z_local with Y_local down
# Conversion: flip Y and Z axes of the camera frame
C_b2cv = np.diag([1, -1, -1])

R_blender_left = get_rotation_matrix(rx_deg_left, ry_deg_left, rz_deg_left)
R_blender_right = get_rotation_matrix(rx_deg_right, ry_deg_right, rz_deg_right)
R_cv_left = C_b2cv @ R_blender_left  # rotation: world -> OpenCV camera frame
R_cv_right = C_b2cv @ R_blender_right  # rotation: world -> OpenCV camera frame

# Translation: t = -R * C (camera center)
t_cv_left = -R_cv_left @ cam_pos_left
t_cv_right = -R_cv_right @ cam_pos_right

print("=" * 60)
print("EXTRINSIC PARAMETERS")
print("=" * 60)
print(f"Euler angles (deg): rx_left={rx_deg_left}, ry_left={ry_deg_left}, rz_left={rz_deg_left}")
print(f"Euler angles (deg): rx_right={rx_deg_right}, ry_right={ry_deg_right}, rz_right={rz_deg_right}")
print(f"Camera position (world): {cam_pos_left}")
print(f"Camera position (world): {cam_pos_right}")
print()
print("R_blender (world->blender cam) =")
print(R_blender_left)
print("R_blender_right =")
print(R_blender_right)
print()
print("R_cv (world->OpenCV cam) =")
print(R_cv_left)
print("R_cv_right =")
print(R_cv_right)
print()
print("t_cv =")
print(t_cv_left)
print("t_cv_right =")
print(t_cv_right)
print()

# Extrinsic matrix [R|t]  (3x4)
Rt_left = np.hstack([R_cv_left, t_cv_left.reshape(3, 1)])
Rt_right = np.hstack([R_cv_right, t_cv_right.reshape(3, 1)])
print("Extrinsic [R|t] (3x4) =")
print(Rt_left)
print("Rt_right =")
print(Rt_right)
print()

# ============================================================
# 3. FULL PROJECTION MATRIX P = K [R|t]
# ============================================================
P_left = K @ Rt_left
P_right = K @ Rt_right

print("=" * 60)
print("PROJECTION MATRIX P = K [R|t]  (3x4)")
print("=" * 60)
print(P_left)
print("P_right =")
print(P_right)
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
# center_A = np.array([-0.0185, 0.003, 0.088])
# Frame C: sprite center after -2m along Y
# center_C = np.array([-0.0185, 0.003 - 2.0, 0.088])

center_Z = np.array([0.011302, 0.368549, -1.2308])
center_A = np.array([0.003031, 0.311474, -1.28304])
center_C = np.array([0.01994, -1.55504, -1.34684])

center_A_left = center_A
center_C_left = center_C

px_A_right = project_point(P_right, center_A)
px_C_right = project_point(P_right, center_C)
px_A_left = project_point(P_left, center_A_left)

print("=" * 60)
print("PROJECTED SPRITE CENTERS")
print("=" * 60)
print(f"Sprite center A (world): {center_A}")
print(f"  -> Pixel: ({px_A_right[0]:.2f}, {px_A_right[1]:.2f})")
print()
print(f"Sprite center C (world): {center_C}")
print(f"  -> Pixel: ({px_C_right[0]:.2f}, {px_C_right[1]:.2f})")
print()
print(f"Sprite center A (world): {center_A_left}")
print(f"  -> Pixel: ({px_A_left[0]:.2f}, {px_A_left[1]:.2f})")
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
w = 0.760055
h = 2.97738


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
corners_A_left_3d = sprite_corners(center_A_left)
corners_C_left_3d = sprite_corners(center_C_left)

# Project corners to pixel coordinates
corners_A_right_px = np.array([project_point(P_right, c) for c in corners_A_3d])
corners_C_right_px = np.array([project_point(P_right, c) for c in corners_C_3d])
corners_A_left_px = np.array([project_point(P_left, c) for c in corners_A_left_3d])
corners_C_left_px = np.array([project_point(P_left, c) for c in corners_C_left_3d])

print("=" * 60)
print("SPRITE CORNERS (Pixels)")
print("=" * 60)
print("Frame A corners (BL, BR, TR, TL):")
for i, (a, c) in enumerate(zip(corners_A_right_px, corners_C_right_px)):
    print(f"  Corner {i}: A=({a[0]:.2f}, {a[1]:.2f})  C=({c[0]:.2f}, {c[1]:.2f})")
print()

print("Frame A2 corners (BL, BR, TR, TL):")
for i, (a, c) in enumerate(zip(corners_A_left_px, corners_C_right_px)):
    print(f"  Corner {i}: A2=({a[0]:.2f}, {a[1]:.2f})  C=({c[0]:.2f}, {c[1]:.2f})")
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


H_right_step = compute_homography(corners_A_right_px, corners_C_right_px)
H_left_step = compute_homography(corners_A_left_px, corners_C_left_px)

# Pure translation taking the left-camera projection of center_A to the
# right-camera projection of center_A.
_tx = px_A_right[0] - px_A_left[0]
_ty = px_A_right[1] - px_A_left[1]
H_left_pixel_to_right = np.array([
    [1.0, 0.0, _tx],
    [0.0, 1.0, _ty],
    [0.0, 0.0, 1.0],
])
H_right_pixel_to_left = np.array([
    [1.0, 0.0, -_tx],
    [0.0, 1.0, -_ty],
    [0.0, 0.0, 1.0],
])

print("=" * 60)
print("HOMOGRAPHY H (maps Frame A pixels -> Frame C pixels)")
print("=" * 60)
print("H_right_step =")
print(H_right_step)
print()

print("H_left_pixel_to_right =")
print(H_left_pixel_to_right)
print()

print("H_right_pixel_to_left =")
print(H_right_pixel_to_left)
print()

print("H_left_step =")
print(H_left_step)
print()

# ---- Verification ----
print("=" * 60)
print("VERIFICATION")
print("=" * 60)
print("Mapping A corners through H and comparing to C corners:")
for i in range(4):
    src_h = np.array([corners_A_right_px[i][0], corners_A_right_px[i][1], 1.0])
    dst_h = H_right_step @ src_h
    dst = dst_h[:2] / dst_h[2]
    expected = corners_C_right_px[i]
    err = np.linalg.norm(dst - expected)
    print(f"  Corner {i}: H(A)=({dst[0]:.2f}, {dst[1]:.2f})  "
          f"C=({expected[0]:.2f}, {expected[1]:.2f})  error={err:.4f} px")

print("Mapping A_left corners through H_right_base_to_left_base and comparing to A_right corners:")
for i in range(4):
    src_h = np.array([corners_A_left_px[i][0], corners_A_left_px[i][1], 1.0])
    dst_h = H_left_pixel_to_right @ src_h
    dst = dst_h[:2] / dst_h[2]
    expected = corners_A_right_px[i]
    err = np.linalg.norm(dst - expected)
    print(f"  Corner {i}: H_left_pixel_to_right(A_left)=({dst[0]:.2f}, {dst[1]:.2f})  "
          f"A_right=({expected[0]:.2f}, {expected[1]:.2f})  error={err:.4f} px")
print()

# Verify center point mapping
center_A_h = np.array([px_A_right[0], px_A_right[1], 1.0])
center_C_mapped = H_right_step @ center_A_h
center_C_mapped = center_C_mapped[:2] / center_C_mapped[2]
print(f"Center A pixel: ({px_A_right[0]:.2f}, {px_A_right[1]:.2f})")
print(f"H(center A):    ({center_C_mapped[0]:.2f}, {center_C_mapped[1]:.2f})")
print(f"Center C pixel: ({px_C_right[0]:.2f}, {px_C_right[1]:.2f})")
print(f"Error: {np.linalg.norm(center_C_mapped - np.array(px_C_right)):.4f} px")
print()

center_A_left_h = np.array([px_A_left[0], px_A_left[1], 1.0])
center_A_mapped = H_left_pixel_to_right @ center_A_left_h
center_A_mapped = center_A_mapped[:2] / center_A_mapped[2]
print(f"Center A_left pixel:                ({px_A_left[0]:.2f}, {px_A_left[1]:.2f})")
print(f"H_left_pixel_to_right(center A_left): ({center_A_mapped[0]:.2f}, {center_A_mapped[1]:.2f})")
print(f"Center A_right pixel:                 ({px_A_right[0]:.2f}, {px_A_right[1]:.2f})")
print(f"Error: {np.linalg.norm(center_A_mapped - np.array(px_A_right)):.4f} px")
print()

# ---- Check if H is affine ----
print("=" * 60)
print("AFFINITY CHECK")
print("=" * 60)
print(f"Last row of H_right_step: [{H_right_step[2,0]:.8f}, {H_right_step[2,1]:.8f}, {H_right_step[2,2]:.8f}]")
print(f"If H is affine, last row should be [0, 0, 1] (or proportional).")
print(f"H_right_step[2,0] / H_right_step[2,2] = {H_right_step[2,0]/H_right_step[2,2]:.8f}")
print(f"H_right_step[2,1] / H_right_step[2,2] = {H_right_step[2,1]/H_right_step[2,2]:.8f}")
if abs(H_right_step[2,0]) < 1e-4 and abs(H_right_step[2,1]) < 1e-4:
    print("-> H is approximately AFFINE.")
else:
    print("-> H is a general PROJECTIVE homography (not purely affine).")
    print("   This is expected: a pure translation in 3D on a plane viewed")
    print("   in perspective generally yields a projective homography,")
    print("   not strictly affine, unless the plane is fronto-parallel.")