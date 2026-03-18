# Robot Optimal Trajectory Planning

[![website link](https://img.shields.io/badge/Website-github.io-blue)](https://tristanschneider2000.github.io/optimal-teleoperation)
<!-- ![doi](https://img.shields.io/badge/DOI--blue) -->

## Introduction

This package contains an optimal control based trajectory planner for a robot arm. The planner has been developed as a
part of a robotic teleoperation system to be used as an MPC. It can however also be used for other purposes.

### Features:
- Planning of joint trajectories for tracking reference end effector pose trajectory
- Multiple configurable minimization criteria:
    - Position tracking
    - Orientation tracking
    - Joint angles, velocities and accelerations
    - Local lateral acceleration of end effector (for liquid stabilization)
    - Manipulability
- Constraints:
    - Bounded joint angles, velocities and accelerations
    - Collision constraints
- Sphere- and capsule-based collision models for UR5e robot arm
- Compilation of solver function for real-time performance

## Installation

To install this package, clone the project and run
```sh
pip install .
```
in the root directory of this project. Dependencies are installed automatically.

In case you plan to modify code yourself, you can install this package in development mode:
```sh
pip install --editable .
```
When using the `-e/--editable` option, changes to the python code take effect immediately and there is no need to
reinstall the package after every change.

If you want to commit your changes, install the pre-commit script first:
```sh
pre-commit install
```
This will automatically format your code and check code style before committing. You can however also manually run it
using:
```sh
pre-commit run [--all-files]
```

## Getting started

The basic usage of the planner is demonstrated by the following [minimal example](examples/minimal_example.py):
```python
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
```
```
[0.         0.25132713 0.75398145 1.25663509 1.58884051 1.67753081
 1.682692   1.67239691 1.65616431]
```

More complicated usage including configuration and collision modelling can be found in the [examples folder](examples).
