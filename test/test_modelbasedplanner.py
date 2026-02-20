import time

import matplotlib.pyplot as plt
import numpy as np

from robot_optimal_trajectory_planning.config import load_config
from robot_optimal_trajectory_planning.ModelBasedPlannerCasadi import ModelBasedPlannerCasadi

NUM_RUNS = 10


def create_reference_trajectory(H: int, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # For simplicity: position moves linearly from x0-position to some target,
    # orientation remains constant (e.g., identity quaternion)
    # Assuming x0 is something like [q, dq] and q defines initial position in some manner.
    # Here we just create a simple linear trajectory in position and a constant orientation.
    ref_pos = np.zeros((3, H))
    ref_orient = np.zeros((4, H))
    ref_orient[0, :] = 1.0  # Identity quaternion (w=1, x=y=z=0)

    # Example: move in x-direction linearly
    final_pos = np.array([0.5, 0.0, 0.2])  # final desired position
    for k in range(H):
        alpha = k / (H - 1) if H > 1 else 1.0
        ref_pos[:, k] = (1 - alpha) * np.zeros(3) + alpha * final_pos

    return ref_pos, ref_orient


def load_real_states(number, H):
    match number:
        case 1:
            # H = 10 in scenario
            # Visualisierung z.B. mit ros2 launch ur_description view_ur.launch.py ur_type:=ur5e
            position = np.array(
                [
                    3.861261989087528,
                    -2.584271833974075,
                    -0.48166289951932784,
                    -0.17602520535300217,
                    5.883748045277765,
                    0.0782065248789722,
                ]
            )
            velocity = (
                np.array(
                    [
                        0.8672979692011697,
                        -4.165145712646668,
                        3.696906925249258,
                        0.9182061917324222,
                        2.0733097902186223,
                        0.8527466232409717,
                    ]
                )
                * 0.1
            )
            ref_pos = np.array(
                [
                    [
                        -0.85991182,
                        -0.89436557,
                        -0.92881932,
                        -0.96327307,
                        -0.99772682,
                        -1.03218057,
                        -1.06663432,
                        -1.10108807,
                        -1.135206,
                        -1.12912253,
                    ],
                    [
                        -0.23979266,
                        -0.28484136,
                        -0.32989005,
                        -0.37493875,
                        -0.41998745,
                        -0.46503614,
                        -0.51008484,
                        -0.55513353,
                        -0.60000474,
                        -0.6226902,
                    ],
                    [
                        0.45894502,
                        0.42698791,
                        0.39503081,
                        0.36307371,
                        0.3311166,
                        0.2991595,
                        0.26720239,
                        0.23524529,
                        0.20322807,
                        0.16534574,
                    ],
                ]
            )
            ref_orient = np.array(
                [
                    [
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                    ],
                    [
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                    ],
                    [
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                    ],
                    [
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                    ],
                ]
            )

        case 2:
            # Slight variation of case 1
            position = np.array(
                [
                    3.961261989087528,
                    -2.684271833974075,
                    -0.58166289951932784,
                    -0.27602520535300217,
                    5.983748045277765,
                    0.1782065248789722,
                ]
            )
            velocity = (
                np.array(
                    [
                        0.7672979692011697,
                        -4.065145712646668,
                        3.596906925249258,
                        0.8182061917324222,
                        1.9733097902186223,
                        0.7527466232409717,
                    ]
                )
                * 0.1
            )
            ref_pos = np.array(
                [
                    [
                        -0.89436557,
                        -0.92881932,
                        -0.96327307,
                        -0.99772682,
                        -1.03218057,
                        -1.06663432,
                        -1.10108807,
                        -1.135206,
                        -1.12912253,
                    ],
                    [
                        -0.28484136,
                        -0.32989005,
                        -0.37493875,
                        -0.41998745,
                        -0.46503614,
                        -0.51008484,
                        -0.55513353,
                        -0.60000474,
                        -0.6226902,
                    ],
                    [
                        0.42698791,
                        0.39503081,
                        0.36307371,
                        0.3311166,
                        0.2991595,
                        0.26720239,
                        0.23524529,
                        0.20322807,
                        0.16534574,
                    ],
                ]
            )
            ref_orient = np.array(
                [
                    [
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                        0.95656506,
                    ],
                    [
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                        -0.10129011,
                    ],
                    [
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                        -0.00866256,
                    ],
                    [
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                        0.2732189,
                    ],
                ]
            )

        case _:
            return None

    if H > ref_pos.shape[1]:
        pad_width = H - ref_pos.shape[1]
        ref_pos = np.hstack([ref_pos, np.repeat(ref_pos[:, -1:], pad_width, axis=1)])
        ref_orient = np.hstack([ref_orient, np.repeat(ref_orient[:, -1:], pad_width, axis=1)])

    return position, velocity, ref_pos[:, :H], ref_orient[:, :H]


def test_casadi_planner():
    # Testing the casadi implementation with the default config
    config = load_config()
    H = config.problem.prediction_horizon

    # Initial state + Reference from real scenario
    position, velocity, ref_pos, ref_orient = load_real_states(1, H)
    x0 = np.zeros((config.robot.nx, 1))
    x0[0:6] = position.reshape(6, 1)
    x0[6:] = velocity.reshape(6, 1)

    # Create planner
    planner = ModelBasedPlannerCasadi(config)
    print(f"FK Sym: {planner._system_model.casadi_fk_pos_function(x0[0])}")
    # print(f"FK Python: {planner.SystemModel.casadi_fk_pos_function2(x0[0])}")
    # print(f"FK Pin: {planner.SystemModel.casadi_fk_pos_function3(x0[0])}")
    planner.set_x0(x0)
    planner.set_reference(ref_pos, ref_orient)

    # Solve optimization
    planner.solve()

    # Test solving multiple times, warm start is implemented in planner object

    time_intervals = np.empty(NUM_RUNS)
    for i in range(NUM_RUNS):
        position, velocity, ref_pos, ref_orient = load_real_states(i % 2 + 1, H)
        x0 = np.zeros((config.robot.nx, 1))
        x0[0:6] = position.reshape(6, 1)
        x0[6:] = velocity.reshape(6, 1)

        planner.set_x0(x0)
        planner.set_reference(ref_pos, ref_orient)

        current_start = time.perf_counter()
        planner.solve()
        current_time = time.perf_counter() - current_start

        time_intervals[i] = current_time

    print(f"Average solving time is: {np.mean(time_intervals)} seconds.")
    print(f"Standard deviation: {np.std(time_intervals)}")
    print(f"Maximum time: {np.max(time_intervals)}")
    print(f"First time intervals: {time_intervals[np.array([0,1,2])]}, and last: {time_intervals[np.array([-2,-1])]}")

    # Get solutions
    x_sol = planner.get_x_sol()
    u_sol = planner.get_u_sol()
    r_sol = planner.get_r_sol()

    # Print results
    if False:
        print("X solution:\n", x_sol)
        print("U solution:\n", u_sol)
        print("R solution (Forward Kinematics):\n", r_sol)

    # Plot results
    if True:
        time_array = np.arange(H + 1) * config.problem.sample_time

        # Plot position reference vs planned end-effector position
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)
        for i, ax in enumerate(axs):
            ax.plot(time_array, r_sol[i, :], label="Planned")
            ax.plot(time_array[1:], ref_pos[i, :], "r--", label="Reference")
            ax.set_ylabel(f"pos_{i}")
            ax.legend()
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")  # X-axis at y=0
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", linewidth=0.5)
        axs[-1].set_xlabel("Time [s]")
        plt.suptitle("End-Effector Position")

        # Plot controls
        fig2, axs2 = plt.subplots(config.robot.nu, 1, figsize=(8, 6), sharex=True, sharey=True)
        for i, ax in enumerate(axs2):
            ax.step(time_array[:H], u_sol[i, :], where="post")
            ax.set_ylabel(f"u_{i}")
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", linewidth=0.5)
        axs2[-1].set_xlabel("Time [s]")
        plt.suptitle("Control Inputs")

        # Plot Joint Positions
        fig3, axs3 = plt.subplots(int(config.robot.nx / 2), 1, figsize=(8, 6), sharex=True, sharey=True)
        for i, ax in enumerate(axs3):
            ax.step(time_array[:], x_sol[i, :], where="post")
            ax.set_ylabel(f"x_{i}")
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", linewidth=0.5)
        axs3[-1].set_xlabel("Time [s]")
        plt.suptitle("Joint Positions")

        # Plot Joint Velocities
        fig4, axs4 = plt.subplots(int(config.robot.nx / 2), 1, figsize=(8, 6), sharex=True, sharey=True)
        for i, ax in enumerate(axs4):
            state = i + int(config.robot.nx / 2)
            ax.step(time_array[:], x_sol[state, :], where="post")
            ax.set_ylabel(f"x_{i}")
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", linewidth=0.5)
        axs4[-1].set_xlabel("Time [s]")
        plt.suptitle("Joint Velocities")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_casadi_planner()
