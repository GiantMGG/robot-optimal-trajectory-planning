# noqa: INP001 (examples is not intended to be a python package)
"""Minimal usage example for ModelBasedPlannerCasadi."""
import numpy as np

from robot_optimal_trajectory_planning.ModelBasedPlannerCasadi import ModelBasedPlannerCasadi

planner = ModelBasedPlannerCasadi()

# set reference trajectories
position_reference = np.array([[0.35, 0.07, 0.10]] * planner.prediction_horizon).T
orientation_reference = np.array([[1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]] * planner.prediction_horizon).T
planner.set_reference(position_reference, orientation_reference)

# set initial state (6 joint angles and 6 joint velocities)
planner.set_x0(np.zeros(12))

planner.solve()

print(planner.get_x_sol()[0, :])  # print the trajectory of the first joint (shoulder pan joint)
