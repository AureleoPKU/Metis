import torch
import torch.nn as nn

def rotation_6d_to_rotation_vector(rot_6d):
    """6D
    : rot_6d (N, 6, T, 1, 2) - 6D [r11, r12, r21, r22, r31, r32]
    : rot_vec (N, 3, T, 1, 2) -  [rx, ry, rz]"""
    # 6D6
    r11 = rot_6d[:, 0:1, :, :, :]  # (N, 1, T, 1, 2)
    r12 = rot_6d[:, 1:2, :, :, :]  # (N, 1, T, 1, 2)
    r21 = rot_6d[:, 2:3, :, :, :]  # (N, 1, T, 1, 2)
    r22 = rot_6d[:, 3:4, :, :, :]  # (N, 1, T, 1, 2)
    r31 = rot_6d[:, 4:5, :, :, :]  # (N, 1, T, 1, 2)
    r32 = rot_6d[:, 5:6, :, :, :]  # (N, 1, T, 1, 2)


    col1 = torch.cat([r11, r21, r31], dim=1)  # (N, 3, T, 1, 2)
    col2 = torch.cat([r12, r22, r32], dim=1)  # (N, 3, T, 1, 2)

    # Gram-Schmidt

    col1_norm = torch.norm(col1, dim=1, keepdim=True)
    col1 = col1 / (col1_norm + 1e-8)


    dot_product = torch.sum(col2 * col1, dim=1, keepdim=True)
    col2_proj = dot_product * col1
    col2_ortho = col2 - col2_proj
    col2_norm = torch.norm(col2_ortho, dim=1, keepdim=True)
    col2 = col2_ortho / (col2_norm + 1e-8)


    col3 = torch.cross(col1, col2, dim=1)

    # R = [col1, col2, col3]



    trace = col1[:, 0:1, :, :, :] + col2[:, 1:2, :, :, :] + col3[:, 2:3, :, :, :]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1 + 1e-8, 1 - 1e-8)
    angle = torch.acos(cos_angle)


    epsilon = 1e-8
    sin_angle = torch.sin(angle)


    sin_angle = torch.where(sin_angle < epsilon, epsilon, sin_angle)

    # Rodrigues
    # R
    rx = (col3[:, 1:2, :, :, :] - col2[:, 2:3, :, :, :]) / (2 * sin_angle)
    ry = (col1[:, 2:3, :, :, :] - col3[:, 0:1, :, :, :]) / (2 * sin_angle)
    rz = (col2[:, 0:1, :, :, :] - col1[:, 1:2, :, :, :]) / (2 * sin_angle)


    small_angle = angle < epsilon
    rx = torch.where(small_angle, torch.zeros_like(rx), rx)
    ry = torch.where(small_angle, torch.zeros_like(ry), ry)
    rz = torch.where(small_angle, torch.zeros_like(rz), rz)

    # =  *
    rot_vec = torch.cat([rx * angle, ry * angle, rz * angle], dim=1)

    return rot_vec


def quaternion_to_6d_rotation(quat):
    """6D
    : quat (N, 4, T, 1, 2) - [w, x, y, z]
    : rot_6d (N, 6, T, 1, 2) - 6D"""

    quat_norm = torch.norm(quat, dim=1, keepdim=True)
    quat = quat / (quat_norm + 1e-8)

    # [w, x, y, z]
    w = quat[:, 0:1, :, :, :]  # (N, 1, T, 1, 2)
    x = quat[:, 1:2, :, :, :]  # (N, 1, T, 1, 2)
    y = quat[:, 2:3, :, :, :]  # (N, 1, T, 1, 2)
    z = quat[:, 3:4, :, :, :]  # (N, 1, T, 1, 2)



    r11 = 1 - 2 * (y**2 + z**2)
    r21 = 2 * (x*y + w*z)
    r31 = 2 * (x*z - w*y)


    r12 = 2 * (x*y - w*z)
    r22 = 1 - 2 * (x**2 + z**2)
    r32 = 2 * (y*z + w*x)

    # 6D6
    rot_6d = torch.cat([r11, r12, r21, r22, r31, r32], dim=1)

    return rot_6d


def quaternion_to_rotation_vector(quat):
    """hf_pose.py"""

    quat_norm = torch.norm(quat, dim=1, keepdim=True)
    quat = quat / (quat_norm + 1e-8)

    # [w, x, y, z]
    w = quat[:, 0:1, :, :, :]  # (N, 1, T, 1, 2)
    x = quat[:, 1:2, :, :, :]  # (N, 1, T, 1, 2)
    y = quat[:, 2:3, :, :, :]  # (N, 1, T, 1, 2)
    z = quat[:, 3:4, :, :, :]  # (N, 1, T, 1, 2)

    # ()
    angle = 2 * torch.acos(torch.clamp(w, -1 + 1e-8, 1 - 1e-8))


    sin_half_angle = torch.sqrt(torch.clamp(1 - w**2, 0, 1))


    epsilon = 1e-8
    sin_half_angle = torch.where(sin_half_angle < epsilon, epsilon, sin_half_angle)


    axis_x = x / sin_half_angle
    axis_y = y / sin_half_angle
    axis_z = z / sin_half_angle


    small_angle = angle < epsilon
    axis_x = torch.where(small_angle, torch.zeros_like(axis_x), axis_x)
    axis_y = torch.where(small_angle, torch.zeros_like(axis_y), axis_y)
    axis_z = torch.where(small_angle, torch.zeros_like(axis_z), axis_z)

    # =  *
    rot_vec = torch.cat([axis_x * angle, axis_y * angle, axis_z * angle], dim=1)

    return rot_vec


def test_rotation_conversion():

    print("=== Testing rotation conversion accuracy ===")


    N, T = 2, 10
    quat = torch.randn(N, 4, T, 1, 2)


    quat_norm = torch.norm(quat, dim=1, keepdim=True)
    quat = quat / quat_norm

    print(f"Original quaternion shape: {quat.shape}")

    # Test 1: quaternion -> 6D -> rotation vector
    print("\n--- Test 1: quaternion -> 6D -> rotation vector ---")
    rot_6d = quaternion_to_6d_rotation(quat)
    print(f"6D rotation shape: {rot_6d.shape}")

    rot_vec_from_6d = rotation_6d_to_rotation_vector(rot_6d)
    print(f"Rotation vector shape from 6D: {rot_vec_from_6d.shape}")

    # Test 2: quaternion -> rotation vector (direct)
    print("\n--- Test 2: quaternion -> rotation vector (direct) ---")
    rot_vec_direct = quaternion_to_rotation_vector(quat)
    print(f"Direct rotation vector shape: {rot_vec_direct.shape}")


    diff = torch.abs(rot_vec_from_6d - rot_vec_direct).mean()
    print(f"\nConversion difference (mean absolute error): {diff.item():.8f}")

    # Test 3: verify 6D rotation matrix reconstruction
    print("\n--- Test 3: verify 6D rotation matrix reconstruction ---")
    test_6d_reconstruction_accuracy(rot_6d)

    return rot_vec_from_6d, rot_vec_direct


def test_6d_reconstruction_accuracy(rot_6d):
    """6D"""
    # 6D
    r11 = rot_6d[:, 0:1, :, :, :]
    r12 = rot_6d[:, 1:2, :, :, :]
    r21 = rot_6d[:, 2:3, :, :, :]
    r22 = rot_6d[:, 3:4, :, :, :]
    r31 = rot_6d[:, 4:5, :, :, :]
    r32 = rot_6d[:, 5:6, :, :, :]


    col1 = torch.cat([r11, r21, r31], dim=1)
    col2 = torch.cat([r12, r22, r32], dim=1)


    col1_norm = torch.norm(col1, dim=1, keepdim=True)
    col1 = col1 / (col1_norm + 1e-8)

    dot_product = torch.sum(col2 * col1, dim=1, keepdim=True)
    col2_proj = dot_product * col1
    col2_ortho = col2 - col2_proj
    col2_norm = torch.norm(col2_ortho, dim=1, keepdim=True)
    col2 = col2_ortho / (col2_norm + 1e-8)

    col3 = torch.cross(col1, col2, dim=1)


    # R^T * R
    R = torch.cat([col1, col2, col3], dim=1)  # (N, 3, T, 1, 2)
    R_flat = R.view(R.shape[0], 3, 3, -1)  # (N, 3, 3, ...)

    # R^T * R
    R_T = R_flat.transpose(1, 2)
    identity_approx = torch.bmm(R_T.contiguous().view(-1, 3, 3),
                               R_flat.contiguous().view(-1, 3, 3))
    identity_approx = identity_approx.view(R.shape[0], 3, 3, -1)


    identity = torch.eye(3, device=R.device).unsqueeze(0).unsqueeze(-1)
    identity = identity.expand_as(identity_approx)


    orthogonality_error = torch.abs(identity_approx - identity).mean()
    print(f"Rotation matrix orthogonality error: {orthogonality_error.item():.8f}")

    # Determinant should be close to 1
    det = torch.det(R_flat.contiguous().view(-1, 3, 3))
    det = det.view(R.shape[0], -1)
    det_error = torch.abs(det - 1.0).mean()
    print(f"Rotation matrix determinant error: {det_error.item():.8f}")

    return orthogonality_error, det_error


if __name__ == "__main__":
    test_rotation_conversion()
