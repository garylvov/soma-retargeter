# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen deploy position contracts from ProtoMotions deployment/robot/{g1,h1_2}.py.

Coordinates are keyed by joint name rather than MJCF actuator index; an enabled
retarget checks every model joint against the contract before solving. Keep
these tables aligned with the deployment profiles when they change.
"""

_H12 = (
    (-0.43, 0.43), (-3.14, 2.5), (-0.43, 3.14), (-0.12, 2.19), (-0.897334, 0.523598), (-0.261799, 0.261799),
    (-0.43, 0.43), (-3.14, 2.5), (-3.14, 0.43), (-0.12, 2.19), (-0.897334, 0.523598), (-0.261799, 0.261799),
    (-2.35, 2.35),
    (-3.14, 1.57), (-0.38, 3.4), (-2.66, 3.01), (-0.95, 3.18), (-3.01, 2.75), (-0.4625, 0.4625), (-1.27, 1.27),
    (-3.14, 1.57), (-3.4, 0.38), (-3.01, 2.66), (-0.95, 3.18), (-2.75, 3.01), (-0.4625, 0.4625), (-1.27, 1.27),
)
_G1 = (
    (-2.5307, 2.8798), (-0.5236, 2.9671), (-2.7576, 2.7576), (-0.087267, 2.8798), (-0.87267, 0.5236), (-0.2618, 0.2618),
    (-2.5307, 2.8798), (-2.9671, 0.5236), (-2.7576, 2.7576), (-0.087267, 2.8798), (-0.87267, 0.5236), (-0.2618, 0.2618),
    (-2.618, 2.618), (-0.52, 0.52), (-0.52, 0.52),
    (-3.0892, 2.6704), (-1.5882, 2.2515), (-2.618, 2.618), (-1.0472, 2.0944), (-1.97222, 1.97222), (-1.61443, 1.61443), (-1.61443, 1.61443),
    (-3.0892, 2.6704), (-2.2515, 1.5882), (-2.618, 2.618), (-1.0472, 2.0944), (-1.97222, 1.97222), (-1.61443, 1.61443), (-1.61443, 1.61443),
)
_H12_NAMES = (
    "left_hip_yaw", "left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle_pitch", "right_ankle_roll", "torso",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
_G1_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
DEPLOY_LIMITS = {
    "h1_2": dict(zip((name + "_joint" for name in _H12_NAMES), _H12, strict=True)),
    "unitree_g1": dict(zip((name + "_joint" for name in _G1_NAMES), _G1, strict=True)),
}


def margin_for_joint(name: str, margins: dict[str, float]) -> float:
    if "ankle" in name:
        return margins["ankle"]
    if "knee" in name:
        return margins["knee"]
    return margins["default"]
