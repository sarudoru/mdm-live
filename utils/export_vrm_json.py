import json
import os
from typing import Dict, Any

import numpy as np
import torch

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import (
    t2m_kinematic_chain,
    t2m_raw_offsets,
    kit_kinematic_chain,
    kit_raw_offsets,
)


def _get_face_joint_indices(dataset: str):
    """Indices used to estimate facing direction in IK.

    Returns [r_hip, l_hip, r_shoulder, l_shoulder] for the dataset.
    """
    if dataset == 'humanml':
        # HML: ['pelvis','left_hip','right_hip',...,'left_shoulder','right_shoulder',...]
        # face_joint_indx = [r_hip, l_hip, sdr_r, sdr_l]
        return [2, 1, 17, 16]
    elif dataset == 'kit':
        # See motion_process.py main KIT example ([11, 16, 5, 8])
        return [11, 16, 5, 8]
    else:
        # default to humanml layout
        return [2, 1, 17, 16]


def _select_skeleton(dataset: str):
    if dataset == 'kit':
        n_raw_offsets = torch.from_numpy(kit_raw_offsets)
        kinematic_chain = kit_kinematic_chain
    else:
        n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        kinematic_chain = t2m_kinematic_chain
    return n_raw_offsets, kinematic_chain


def compute_local_quaternions_from_positions(positions: np.ndarray, dataset: str) -> np.ndarray:
    """
    Compute per-joint local quaternions via IK from world positions.

    positions: (T, J, 3) in meters.
    returns: (T, J, 4) quaternions (w,x,y,z) per joint, local to parent.
    """
    assert positions.ndim == 3 and positions.shape[-1] == 3
    T, J, _ = positions.shape

    n_raw_offsets, kinematic_chain = _select_skeleton(dataset)
    face_joint_indx = _get_face_joint_indices(dataset)

    skel = Skeleton(n_raw_offsets, kinematic_chain, device='cpu')
    # Scale the skeleton offsets to match the first frame bone lengths
    _ = skel.get_offsets_joints(torch.from_numpy(positions[0]))

    # IK to quaternions
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True, fix_bug=False)
    # Ensure quaternion continuity across frames
    from data_loaders.humanml.common.quaternion import qfix
    quat_params = qfix(quat_params)
    return quat_params  # (T, J, 4) wxyz


def to_vrm_motion_json(positions: np.ndarray, fps: float, dataset: str) -> Dict[str, Any]:
    """Build a VRM-friendly motion JSON from positions.

    JSON structure:
    {
      "version": 1,
      "fps": number,
      "joints": ["pelvis", ...],  # HML joint order
      "frames": [
         {"root_position": [x,y,z],
          "rotations": [[w,x,y,z], ... J],
          "positions": [[x,y,z], ... J]
         }, ... T
      ]
    }
    """
    from data_loaders.humanml_utils import HML_JOINT_NAMES

    T, J, _ = positions.shape
    quats = compute_local_quaternions_from_positions(positions, dataset)  # (T,J,4) wxyz

    frames = []
    for t in range(T):
        root = positions[t, 0].tolist()
        frames.append({
            'root_position': root,
            'rotations': quats[t].tolist(),
            'positions': positions[t].tolist(),
        })

    return {
        'version': 1,
        'fps': fps,
        'joints': HML_JOINT_NAMES,
        'frames': frames,
    }


def write_vrm_motion_json(out_path: str, positions: np.ndarray, fps: float, dataset: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = to_vrm_motion_json(positions, fps, dataset)
    with open(out_path, 'w') as f:
        json.dump(payload, f)
    return out_path
