import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, get_euler_xyz
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import copy


def ang_vel_interpolation(rot_c, rot_n, delta_t):
    rotvec = (R.from_quat(rot_n) * R.from_quat(rot_c).inv()).as_rotvec()
    angle = np.sqrt(np.sum(rotvec**2))
    axis = rotvec / (angle + 1e-8)
    #
    ang_vel = [
        (axis[0] * angle) / delta_t,
        (axis[1] * angle) / delta_t,
        (axis[2] * angle) / delta_t,
    ]
    return ang_vel


@torch.jit.script
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower


def slerp(q0, q1, fraction, eps=np.finfo(float).eps * 4.0, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdims=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < eps).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = copy.copy(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.arccos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < eps).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out


@torch.jit.script
def interpolate(val0, val1, blend):
    return (1.0 - blend) * val0 + blend * val1
