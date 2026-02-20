from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING


@dataclass
class RobotConfig:
    nx: int = MISSING  # 2*nq + additional states
    nq: int = MISSING
    nu: int = MISSING
    q_min: list[float] = MISSING
    q_max: list[float] = MISSING
    dq_min: list[float] = MISSING
    dq_max: list[float] = MISSING
    u_min: list[float] = MISSING
    u_max: list[float] = MISSING
    collision_model: str = MISSING
    dh_parameters: list[list[float]] = MISSING
    custom_tool_translation: list[float] = field(default_factory=lambda: [0, 0, 0])
    custom_tool_rotation_matrix: list[list[float]] = field(default_factory=lambda: np.eye(3).tolist())
    joint_names: list[str] = MISSING


@dataclass
class UR5eConfig(RobotConfig):
    nx: int = 12  # 2*nq + additional states
    nq: int = 6
    nu: int = 6
    q_min: list[float] = field(default_factory=lambda: [-2 * np.pi] * UR5eConfig.nq)
    q_max: list[float] = field(default_factory=lambda: [2 * np.pi] * UR5eConfig.nq)
    dq_min: list[float] = field(default_factory=lambda: [-0.8 * np.pi] * UR5eConfig.nq)
    dq_max: list[float] = field(default_factory=lambda: [0.8 * np.pi] * UR5eConfig.nq)
    u_min: list[float] = field(default_factory=lambda: [-4 * np.pi] * UR5eConfig.nu)
    u_max: list[float] = field(default_factory=lambda: [4 * np.pi] * UR5eConfig.nu)
    collision_model: str = "capsules"
    dh_parameters: list[list[float]] = field(
        default_factory=lambda: [
            [0.1625, 0, 0, np.pi / 2],
            [0, 0, -0.425, 0],
            [0, 0, -0.3922, 0],
            [0.1333, 0, 0, np.pi / 2],
            [0.0997, 0, 0, -np.pi / 2],
            [0.0996, 0, 0, 0],
        ]
    )
    custom_tool_translation: list[float] = field(default_factory=lambda: [0, 0, 0.175])
    custom_tool_rotation_matrix: list[list[float]] = field(
        default_factory=lambda: [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ]
    )
    joint_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
    )
