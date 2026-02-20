from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING


def _common_plugin_options() -> dict[str, Any]:
    return {
        "expand": True,
        "print_time": False,
        "jit": True,
        "jit_cleanup": False,
        "jit_temp_suffix": False,
        "jit_options": {
            "verbose": True,
            "flags": ["-O3", "-march=native", "-DNDEBUG", "-ffp-contract=fast", "-v"],
            "compiler": "ccache gcc",
        },
    }


class CasadiMode(Enum):
    OPTI = "opti"
    FUNCTION = "function"


@dataclass
class SolverConfig:
    name: str = MISSING
    casadi_mode: CasadiMode = CasadiMode.FUNCTION
    solver_options: dict = field(default_factory=dict)
    plugin_options: dict = field(default_factory=_common_plugin_options)


@dataclass
class IpoptSolverConfig(SolverConfig):
    name: str = "ipopt"
    solver_options: dict = field(
        default_factory=lambda: {
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 0.0001,
            "acceptable_tol": 0.001,
            "acceptable_constr_viol_tol": 0.01,
            "acceptable_obj_change_tol": 0.01,
            "acceptable_iter": 2,
            "tol": 0.001,
            "compl_inf_tol": 0.001,
            "constr_viol_tol": 0.001,
            "mu_strategy": "monotone",
            "accept_every_trial_step": "yes",
            "linear_solver": "mumps",
            "hsllib": "/usr/local/lib/libcoinhsl.so",
            "ma57_pre_alloc": 2,
            "ma57_block_size": 32,
            "fixed_variable_treatment": "make_constraint",
            "print_timing_statistics": "yes",
            "print_level": 0,
        }
    )


@dataclass
class KnitroSolverConfig(SolverConfig):
    name: str = "knitro"
    solver_options: dict = field(
        default_factory=lambda: {
            "tuner": 0,
            "algorithm": 1,
            "bar_conic_enable": 1,
            "bar_murule": 1,
            "datacheck": 0,
            "hessopt": 1,
            "linsolver": 4,
        }
    )


@dataclass
class FatropSolverConfig(SolverConfig):
    name: str = "fatrop"
    solver_options: dict = field(
        default_factory=lambda: {
            "print_level": 0,
            "tol": 1e-4,
        }
    )
    plugin_options: dict = field(
        default_factory=lambda: _common_plugin_options()
        | {
            "structure_detection": "auto",
            "debug": False,
        }
    )


@dataclass
class MadNlpSolverConfig(SolverConfig):
    name: str = "madnlp"
    solver_options: dict = field(
        default_factory=lambda: {
            "linear_solver": "mumps",
            "print_level": 2,
            "tol": 1e-4,
        }
    )
