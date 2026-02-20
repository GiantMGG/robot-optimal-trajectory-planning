from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from ._problem import CostConfig, ProblemConfig
from ._robot import RobotConfig, UR5eConfig
from ._solver import (
    CasadiMode,
    FatropSolverConfig,
    IpoptSolverConfig,
    KnitroSolverConfig,
    MadNlpSolverConfig,
    SolverConfig,
)


@dataclass
class MpcConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_", {"robot": "ur5e"}, {"solver": "ipopt"}])
    robot: RobotConfig = MISSING  # filled by defaults list
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    solver: SolverConfig = MISSING  # filled by defaults list


_cs = ConfigStore.instance()
_cs.store(name="mpc", node=MpcConfig)
_cs.store(name="ur5e", node=UR5eConfig, group="robot")
_cs.store(name="ipopt", node=IpoptSolverConfig, group="solver")
_cs.store(name="knitro", node=KnitroSolverConfig, group="solver")
_cs.store(name="fatrop", node=FatropSolverConfig, group="solver")
_cs.store(name="madnlp", node=MadNlpSolverConfig, group="solver")


def compose_config(overrides: list[str] | None = None, config_name: str = "mpc") -> MpcConfig:
    """Return the composed config as an instance of the dataclass.

    Assumes that hydra is initialized.
    """
    dictconfig = hydra.compose(config_name=config_name, overrides=overrides)
    return OmegaConf.to_object(dictconfig)  # pyright: ignore[reportReturnType]


def load_config(
    config_dir: str | None = None, overrides: list[str] | None = None, config_name: str = "mpc"
) -> MpcConfig:
    if config_dir is not None:
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            return compose_config(overrides=overrides, config_name=config_name)

    # no config_dir passed
    config_dir = str(files("robot_optimal_trajectory_planning").joinpath("config/custom"))
    try:
        # if a custom_mpc.yaml is given, respect it
        with hydra.initialize_config_dir(config_dir=config_dir):
            return compose_config(overrides=overrides, config_name="custom_" + config_name)
    except hydra.MissingConfigException:
        # otherwise load default config
        with hydra.initialize(version_base=None):
            return compose_config(overrides=overrides, config_name=config_name)


# the list of names that can be imported from this module
__all__ = ["CasadiMode", "CostConfig", "MpcConfig", "ProblemConfig", "RobotConfig", "SolverConfig", "load_config"]
