from dataclasses import dataclass, field


@dataclass
class CostConfig:
    position_tracking: list[float] = field(default_factory=lambda: [100] * 3)
    orientation_tracking: float = 50
    joint_angles: list[float] = field(default_factory=lambda: [0.0] * 6)
    joint_velocities: list[float] = field(default_factory=lambda: [0.1] * 6)
    joint_accelerations: list[float] = field(default_factory=lambda: [0.02] * 6)
    anti_slosh: float = 0
    manipulability: float = 0
    slack: float = 5e4
    collision_multipliers: float = 0


@dataclass
class ProblemConfig:
    prediction_horizon: int = 8
    sample_time: float = 0.2
    stage_cost: CostConfig = field(default_factory=CostConfig)
    end_cost: CostConfig = field(default_factory=CostConfig)
    ignore_collisions: bool = False
    obstacles: list[str] = field(default_factory=list)
