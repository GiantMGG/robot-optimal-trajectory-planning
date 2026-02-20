from abc import ABC, abstractmethod

import numpy as np


class ModelBasedPlannerInterface(ABC):
    """Interface for model based planner"""

    @abstractmethod
    def __init__(self, config: dict) -> None:
        pass

    @abstractmethod
    def solve(self) -> None:
        pass

    @abstractmethod
    def set_x0(self, x0_value: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_reference(self, ref_pos: np.ndarray, ref_quat: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_x_sol(self) -> np.ndarray:
        """If a previous solution is available returns X0-Xn."""

    @abstractmethod
    def get_u_sol(self) -> np.ndarray:
        """If a previous solution is available returns U0-U(N-1)."""

    @abstractmethod
    def get_r_sol(self) -> np.ndarray | None:
        """If a previous solution is available returns Forward Kinematics 'r' for t:0->N."""
