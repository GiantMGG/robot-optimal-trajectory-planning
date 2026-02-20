"""Model-Based Trajectory Planner using CasADi for Optimal Control.

This module implements a nonlinear model predictive control (NMPC) solver for robot
trajectory planning using the CasADi optimization framework. It formulates and solves
an optimal control problem that minimizes tracking error while respecting physical
constraints and avoiding collisions.
"""

import time
from collections import OrderedDict

import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation

from robot_optimal_trajectory_planning.config import CasadiMode, MpcConfig, load_config
from robot_optimal_trajectory_planning.ModelBasedPlannerInterface import ModelBasedPlannerInterface
from robot_optimal_trajectory_planning.SystemModel import SystemModel


class ModelBasedPlannerCasadi(ModelBasedPlannerInterface):
    """CasADi-based Model Predictive Control solver for robot trajectory planning.

    This class formulates and solves an optimal control problem for robot motion planning
    that considers:
    - Position and orientation tracking of the end-effector
    - Joint position, velocity, and acceleration limits
    - Collision avoidance constraints
    - Multiple cost objectives (tracking, smoothness, manipulability, anti-slosh)

    The solver can operate in two modes, selected by config:
    - "opti": Direct optimization using CasADi's Opti interface (slower, more flexible)
    - "function": Compilable CasADi function (faster, fixed structure)
    """

    def __init__(self, config: MpcConfig | None = None) -> None:
        """Initialize the CasADi-based trajectory planner.

        Args:
            config: Configuration object, see `config.MpcConfig`

        """
        if config is None:
            # use default configuration if no other configuration is passed
            config = load_config()
        self._config: MpcConfig = config
        # Robot kinematics and dynamics model
        self._system_model = SystemModel(config)

        # Possibly compiled solver function (unused in "opti" mode)
        self._loaded_solver: ca.Function | None = None

        # CasADi Opti object for formulating the optimization problem
        self._opti = ca.Opti()
        # Symbolic casadi variables for debugging purposes
        self.opti_variables_debug: OrderedDict[str, ca.MX] = OrderedDict()
        # Mapping from constraint index to human-readable label
        self._constraint_labels: dict[int, str] = {}

        # Formulate the optimization problem
        (
            self._X,
            self._U,
            self._collision_multipliers,
            self._slack,
        ) = self._create_decision_variables()
        (
            self._x0_param,
            self._ref_pos_param,
            self._ref_orient_param,
            self._stage_cost_weights_param,
            self._end_cost_weights_param,
        ) = self._create_casadi_parameters()
        cost = self._cost_function(
            self._X,
            self._U,
            self._collision_multipliers,
            self._slack,
            self._stage_cost_weights_param,
            self._end_cost_weights_param,
        )
        self._add_constraints(self._X, self._U, self._collision_multipliers, self._slack)
        self._opti.minimize(cost)
        self._setup_solver()

        # Initialize solution storage
        self._x_sol = np.zeros((config.robot.nx, config.problem.prediction_horizon + 1))
        self._u_sol = np.zeros((config.robot.nu, config.problem.prediction_horizon))
        self._s_sol = np.zeros((1, config.problem.prediction_horizon))  # Slack variables
        self._col_multipliers_sol = np.zeros(
            (self._system_model.collision_model.get_collision_multiplier_count(), config.problem.prediction_horizon)
        )
        self._lam_g_sol = np.zeros(self._opti.lam_g.shape)
        # Values of the debug variables
        self._debug_values: OrderedDict[str, float] = OrderedDict()
        self.current_sol: ca.OptiSol | None = None

        # Storage for current problem parameters
        self._x0_value: np.ndarray | None = None
        self._ref_pos: np.ndarray | None = None
        self._ref_orient: list[np.ndarray] | None = None

    def _cost_function(
        self,
        X: ca.MX,
        U: ca.MX,
        collision_multipliers: ca.MX,
        slack: ca.MX,
        stage_cost_weights: dict[str, ca.MX],
        end_cost_weights: dict[str, ca.MX],
    ) -> ca.MX:
        """Construct the multi-objective cost function for trajectory optimization.

        The cost function balances multiple objectives and supports different
        weights for stage costs (all steps except final) and end costs (final step):
        1. Position tracking: Minimize deviation from desired end-effector position
        2. Orientation tracking: Minimize rotation error using Frobenius norm
        3. Control effort: Penalize joint accelerations
        4. Velocity smoothness: Penalize joint velocities
        5. Anti-slosh: Minimize lateral accelerations in TCP frame (for liquid transport)
        6. Joint centering: Keep joints near zero position for better manipulability
        7. Manipulability: Maximize kinematic manipulability measure
        8. Soft constraints: Penalize slack variable usage
        9. Regularization: Penalize collision multipliers

        Args:
            X: State trajectory of shape (nx, (H+1))
            U: Control input trajectory of shape (nu, H)
            collision_multipliers: Collision constraint multipliers
            slack: Slack variables for soft collision constraints
            stage_cost_weights: Weights used for all stages except terminal
            end_cost_weights: Weights used for the terminal stage

        Returns:
            Total scalar cost (CasADi MX expression)

        """
        H = self._config.problem.prediction_horizon
        nq = self._config.robot.nq

        # Initialize cost terms as scalar CasADi expressions
        cost_position_tracking = ca.MX(1, 1)
        cost_orientation_tracking = ca.MX(1, 1)
        cost_joint_accelerations = ca.MX(1, 1)
        cost_joint_velocities = ca.MX(1, 1)
        cost_tcp_acceleration = ca.MX(1, 1)
        cost_joint_angles = ca.MX(1, 1)
        cost_manipulability = ca.MX(1, 1)
        cost_slack = ca.MX(1, 1)
        cost_collision_multipliers_regularization = ca.MX(1, 1)

        # Loop over prediction horizon: k = 0 to H-1
        for k in range(H):
            x_next = X[:, k + 1]
            uk = U[:, k]

            # choose weights depending on whether this is the terminal stage
            w = end_cost_weights if k == H - 1 else stage_cost_weights

            # 1. Position tracking error (quadratic)
            error_position = self._system_model.casadi_fk_pos_function(x_next[:nq]) - self._ref_pos_param[:, k]
            error_term = error_position.T @ (ca.diag(w["position_tracking"]) @ error_position)
            cost_position_tracking += error_term

            # 2. Orientation tracking error using Frobenius norm
            rotmat_next: ca.MX = self._system_model.casadi_fk_rotation_matrix_function(
                x_next[:nq]
            )  # pyright: ignore[reportAssignmentType]
            frobenius_squared = self._error_function_orientation_frobenius(rotmat_next, self._ref_orient_param[k])
            cost_orientation_tracking += w["orientation_tracking"] * frobenius_squared

            # 3. Joint acceleration penalty (control effort)
            cost_joint_accelerations += uk.T @ ca.diag(w["joint_accelerations"]) @ uk

            # 4. Joint velocity penalty (smoothness)
            dot_q_next = x_next[nq : 2 * nq]
            cost_joint_velocities += dot_q_next.T @ ca.diag(w["joint_velocities"]) @ dot_q_next

            # 5. Anti-slosh cost: minimize lateral accelerations in TCP frame
            # This is important for transporting liquids without spilling
            local_acc_gravity = self._system_model.casadi_local_acc_gravity_function(x_next, uk)
            gravity = ca.vcat([0, 0, -9.81])
            vertical_component_factor = 0.1  # Lateral components are more critical than vertical
            cost_tcp_acceleration += w["anti_slosh"] * ca.sumsqr(
                ca.vcat([1, 1, vertical_component_factor]) * (local_acc_gravity + gravity)
            )

            # 6. Joint centering: keep joints close to zero position
            q_next = x_next[:nq]
            cost_joint_angles += q_next.T @ ca.diag(w["joint_angles"]) @ q_next

            # 7. Manipulability optimization: maximize determinant of Jacobian
            cost_manipulability += w["manipulability"] / (
                self._system_model.casadi_jacobian_absolute_determinant_ur(x_next[:nq]) + 1e-2
            )  # pyright: ignore[reportOperatorIssue]

            # 8. Slack variable penalty (soft constraints)
            cost_slack += w["slack"] * slack[k]

            # 9. Collision multipliers regularization
            if collision_multipliers.numel() > 0:
                cost_collision_multipliers_regularization += (
                    w["collision_multipliers"] * ca.sumsqr(collision_multipliers) / collision_multipliers.numel()
                )

        # Combine all cost terms
        cost = (
            cost_position_tracking
            + cost_orientation_tracking
            + cost_joint_accelerations
            + cost_joint_velocities
            + cost_tcp_acceleration
            + cost_joint_angles
            + cost_manipulability
            + cost_slack
            + cost_collision_multipliers_regularization
        )

        self.opti_variables_debug.update(
            [
                ("cost", cost),
                ("cost_position_tracking", cost_position_tracking),
                ("cost_orientation_tracking", cost_orientation_tracking),
                ("cost_joint_velocities", cost_joint_velocities),
                ("cost_joint_accelerations", cost_joint_accelerations),
                ("cost_tcp_acceleration", cost_tcp_acceleration),
                ("cost_joint_angles", cost_joint_angles),
                ("cost_manipulability", cost_manipulability),
                ("cost_slack", cost_slack),
                ("cost_collision_multipliers_regularization", cost_collision_multipliers_regularization),
            ]
        )
        return cost

    def _add_constraints(self, X: ca.MX, U: ca.MX, collision_multipliers: ca.MX, slack: ca.MX) -> None:
        """Add all constraints to the optimization problem.

        Constraints include:
        1. System dynamics: x_{k+1} = f(x_k, u_k) (equality)
        2. Initial condition: x_0 = x0_param (equality)
        3. Joint angle limits: q_min ≤ q_k ≤ q_max (inequality)
        4. Joint velocity limits: dq_min ≤ dq_k ≤ dq_max (inequality)
        5. Joint acceleration limits: ddq_min ≤ u_k ≤ ddq_max (inequality)
        6. Collision avoidance: g(q_k) + slack_k ≥ 0 (soft inequality)
        7. Non-negativity of slack: slack_k ≥ 0 (inequality)

        Args:
            X: State trajectory decision variables
            U: Control input decision variables
            collision_multipliers: Collision constraint multiplier variables
            slack: Slack variables for soft collision constraints

        """
        H = self._config.problem.prediction_horizon
        nq = self._config.robot.nq

        # Get collision constraint function from the system model
        collision_constraints_fun = self._system_model.collision_model.get_constraint_function()
        collision_constraints_dimension, _ = collision_constraints_fun.size_out(0)

        constraint_index = 0

        # Loop over prediction horizon
        for k in range(H):
            xk = X[:, k]
            x_next = X[:, k + 1]
            uk = U[:, k]

            # ==================== System Dynamics Constraints ====================
            # Enforce that the predicted next state matches the dynamics model
            num_existing_constraints = self._opti.ng
            constraint_index = num_existing_constraints

            x_next_predicted: ca.MX = self._system_model.casadi_linear_robot_model_function(
                ca.vcat([xk, uk])
            )  # pyright: ignore[reportAssignmentType]

            # Apply constraints component-wise for better constraint labeling
            for i in range(x_next.shape[0]):
                constraint_name = f"state_dynamics_state_{i}_step_{k}"
                self._opti.subject_to(x_next[i] == x_next_predicted[i])
                self._constraint_labels[constraint_index] = constraint_name
                constraint_index += 1

            # ==================== Initial Conditions ====================
            if k == 0:
                self._opti.subject_to(xk == self._x0_param)

            # ==================== State Constraints (for k >= 1) ====================
            if k >= 1:
                # Joint angle bounds for each joint
                for i in range(nq):
                    constraint_name = f"joint_limit_{i}_step_{k}"
                    self._opti.subject_to(
                        self._opti.bounded(
                            self._config.robot.q_min[i],  # pyright: ignore[reportArgumentType]
                            xk[i],
                            self._config.robot.q_max[i],  # pyright: ignore[reportArgumentType]
                        )
                    )
                    self._constraint_labels[constraint_index] = constraint_name
                    constraint_index += 1

                # Joint velocity bounds for each joint
                for i in range(nq):
                    constraint_name = f"joint_velocity_limit_{i}_step_{k}"
                    dot_qk = xk[nq : 2 * nq]
                    self._opti.subject_to(
                        self._opti.bounded(
                            self._config.robot.dq_min[i],  # pyright: ignore[reportArgumentType]
                            dot_qk[i],
                            self._config.robot.dq_max[i],  # pyright: ignore[reportArgumentType]
                        )
                    )
                    self._constraint_labels[constraint_index] = constraint_name
                    constraint_index += 1

            # ==================== Control Input Constraints ====================
            # Joint acceleration bounds for each joint (applied at all time steps)
            for i in range(uk.shape[0]):
                constraint_name = f"joint_acceleration_limit_{i}_step_{k}"
                self._opti.subject_to(
                    self._opti.bounded(
                        self._config.robot.u_min[i],  # pyright: ignore[reportArgumentType]
                        uk[i],
                        self._config.robot.u_max[i],  # pyright: ignore[reportArgumentType]
                    )
                )  # pyright: ignore[reportArgumentType]
                self._constraint_labels[constraint_index] = constraint_name
                constraint_index += 1

            # ==================== Collision Constraints (for k >= 1) ====================
            if k >= 1:
                # Only add collision constraints if the collision model has constraints
                if collision_constraints_dimension > 0:
                    # Soft constraint: collision_constraint + slack >= 0
                    self._opti.subject_to(
                        collision_constraints_fun(xk[:nq], collision_multipliers[:, k - 1]) + slack[k - 1] >= 0
                    )
                # Slack variables must be non-negative
                self._opti.subject_to(slack[k - 1] >= 0)

        # ==================== Constraints for last time step (at k = H) ====================
        # Joint angle limits at final time step
        for i in range(X[:nq, H].shape[0]):
            constraint_name = f"joint_limit_{i}_step_{H}"
            self._opti.subject_to(
                self._opti.bounded(
                    self._config.robot.q_min[i],  # pyright: ignore[reportArgumentType]
                    X[i, H],
                    self._config.robot.q_max[i],  # pyright: ignore[reportArgumentType]
                )
            )
            self._constraint_labels[constraint_index] = constraint_name
            constraint_index += 1

        # Joint velocity limits at final time step
        for i in range(X[nq : 2 * nq, H].shape[0]):
            constraint_name = f"joint_velocity_limit_{i}_step_{H}"
            self._opti.subject_to(
                self._opti.bounded(
                    self._config.robot.dq_min[i],  # pyright: ignore[reportArgumentType]
                    X[nq + i, H],
                    self._config.robot.dq_max[i],  # pyright: ignore[reportArgumentType]
                )
            )
            self._constraint_labels[constraint_index] = constraint_name
            constraint_index += 1

        # Collision constraints at final time step
        if collision_constraints_dimension > 0:
            self._opti.subject_to(
                collision_constraints_fun(X[:nq, H], collision_multipliers[:, H - 1]) + slack[H - 1] >= 0
            )
        self._opti.subject_to(slack[H - 1] >= 0)

    def _create_casadi_parameters(
        self,
    ) -> tuple[ca.MX, ca.MX, list[ca.MX], OrderedDict[str, ca.MX], OrderedDict[str, ca.MX]]:
        """Create CasADi parameters for the optimization problem.

        Parameters are values that change between solves but are fixed during a solve.
        This includes the initial state, reference trajectory, and cost weights.

        Returns:
            Tuple containing:
                - x0: Initial state parameter of shape (nx, 1)
                - r_ref_pos: Reference position trajectory of shape (3, H)
                - R_ref_orient: List of reference rotation matrices (H elements of shape (3,3))
                - stage cost weight parameters (OrderedDict)
                - end cost weight parameters (OrderedDict)

        """
        H = self._config.problem.prediction_horizon
        nq = self._config.robot.nq
        nx = self._config.robot.nx

        # Initial state parameter
        x0 = self._opti.parameter(nx)

        # Cost function weight parameters (allows runtime tuning of weights)
        # Using OrderedDict because the order must not change between creating and calling solver function
        stage_w_params: OrderedDict[str, ca.MX] = OrderedDict(
            [
                ("position_tracking", self._opti.parameter(3)),  # Position tracking weights (x, y, z)
                ("orientation_tracking", self._opti.parameter(1)),  # Orientation tracking weight
                ("joint_accelerations", self._opti.parameter(nq)),  # Joint acceleration weights
                ("joint_velocities", self._opti.parameter(nq)),  # Joint velocity weights
                ("anti_slosh", self._opti.parameter(1)),  # Anti-slosh weight
                ("joint_angles", self._opti.parameter(nq)),  # Joint centering weights
                ("manipulability", self._opti.parameter(1)),  # Manipulability weight
                ("slack", self._opti.parameter(1)),  # Slack variable penalty
                ("collision_multipliers", self._opti.parameter(1)),  # Collision multiplier regularization
            ]
        )

        end_w_params: OrderedDict[str, ca.MX] = OrderedDict(
            [
                ("position_tracking", self._opti.parameter(3)),
                ("orientation_tracking", self._opti.parameter(1)),
                ("joint_accelerations", self._opti.parameter(nq)),
                ("joint_velocities", self._opti.parameter(nq)),
                ("anti_slosh", self._opti.parameter(1)),
                ("joint_angles", self._opti.parameter(nq)),
                ("manipulability", self._opti.parameter(1)),
                ("slack", self._opti.parameter(1)),
                ("collision_multipliers", self._opti.parameter(1)),
            ]
        )

        # Set default values from config (only needed for "opti" mode)
        if self._config.solver.casadi_mode == CasadiMode.OPTI:
            # populate stage weights
            for cost_name, cost_weight_param in stage_w_params.items():
                self._opti.set_value(cost_weight_param, getattr(self._config.problem.stage_cost, cost_name))
            # populate end weights
            for cost_name, cost_weight_param in end_w_params.items():
                self._opti.set_value(cost_weight_param, getattr(self._config.problem.end_cost, cost_name))

        # Reference position trajectory (end-effector position in 3D)
        r_ref_pos = self._opti.parameter(3, H)

        # Reference orientation trajectory (rotation matrices)
        # Each time step has a 3x3 rotation matrix
        R_ref_orient = [self._opti.parameter(3, 3) for _ in range(H)]

        return x0, r_ref_pos, R_ref_orient, stage_w_params, end_w_params

    def _create_decision_variables(self) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
        """Create decision variables for the optimization problem.

        Decision variables are the unknowns that the solver optimizes.

        Returns:
            Tuple containing:
                - X: State trajectory of shape (nx, H+1) - includes X_0 to X_H
                - U: Control input trajectory of shape (nu, H) - includes U_0 to U_{H-1}
                - collision_multipliers: Collision multipliers of shape (n_col, H)
                - slack: Slack variables for soft constraints, shape(1, H)

        """
        nx = self._config.robot.nx
        nu = self._config.robot.nu
        H = self._config.problem.prediction_horizon

        # Initialize empty matrices
        X: ca.MX = ca.MX(nx, 0)
        U: ca.MX = ca.MX(nu, 0)
        number_collision_multipliers = self._system_model.collision_model.get_collision_multiplier_count()
        collision_multipliers: ca.MX = ca.MX(number_collision_multipliers, 0)
        slack: ca.MX = ca.MX(1, 0)

        # Create variables for each time step k = 0 to H-1
        for k in range(H):
            # State variables at time k
            xk = ca.MX(0, 0)
            for _ in range(nx):
                xk = ca.vertcat(xk, self._opti.variable(1, 1))
            X = ca.hcat((X, xk))  # pyright: ignore[reportAssignmentType]

            # Collision multipliers and slack (only for k ≥ 1)
            if k >= 1:
                collision_multipliers_k = ca.MX(0, 1)
                for _ in range(number_collision_multipliers):
                    collision_multipliers_k = ca.vertcat(collision_multipliers_k, self._opti.variable(1, 1))
                collision_multipliers = ca.hcat(
                    (collision_multipliers, collision_multipliers_k)
                )  # pyright: ignore[reportAssignmentType]
                slack = ca.hcat((slack, self._opti.variable(1, 1)))  # pyright: ignore[reportAssignmentType]

            # Control input at time k
            uk = ca.MX(0, 0)
            for _ in range(nu):
                uk = ca.vertcat(uk, self._opti.variable(1, 1))
            U = ca.hcat((U, uk))  # pyright: ignore[reportAssignmentType]

        # Terminal state at time H
        xk = ca.MX(0, 0)
        for _ in range(nx):
            xk = ca.vertcat(xk, self._opti.variable(1, 1))
        X = ca.hcat((X, xk))  # pyright: ignore[reportAssignmentType]

        # Terminal collision multipliers and slack
        collision_multipliers_k = ca.MX(0, 1)
        for _ in range(number_collision_multipliers):
            collision_multipliers_k = ca.vertcat(collision_multipliers_k, self._opti.variable(1, 1))
        collision_multipliers = ca.hcat(
            (collision_multipliers, collision_multipliers_k)
        )  # pyright: ignore[reportAssignmentType]
        slack = ca.hcat((slack, self._opti.variable(1, 1)))  # pyright: ignore[reportAssignmentType]

        return X, U, collision_multipliers, slack

    def _setup_solver(self) -> None:
        """Configure and initialize the nonlinear programming (NLP) solver.

        This method:
        1. Retrieves solver options from the configuration
        2. Configures the CasADi Opti object with the selected solver
        3. Optionally compiles the problem into a CasADi function for faster repeated solves

        For "function" mode, the entire optimization problem is compiled into a single
        CasADi function that can be called repeatedly with different parameters.
        """
        # Configure the solver (e.g., ipopt, fatrop, etc.)
        self._opti.solver(
            self._config.solver.name,
            self._config.solver.plugin_options,
            self._config.solver.solver_options,
        )

        # Compile to a CasADi function for faster repeated solves
        if self._config.solver.casadi_mode == CasadiMode.FUNCTION:
            # create concatenated weight vectors: stage then end
            stage_weights_vec = ca.vcat(list(self._stage_cost_weights_param.values()))
            end_weights_vec = ca.vcat(list(self._end_cost_weights_param.values()))

            self._loaded_solver = self._opti.to_function(
                "solver_func",
                # Input arguments
                [
                    self._ref_pos_param,
                    ca.hcat(self._ref_orient_param),
                    self._x0_param,
                    self._X,
                    self._U,
                    self._opti.lam_g,
                    stage_weights_vec,
                    end_weights_vec,
                    self._slack,
                    self._collision_multipliers,
                ],
                # Output arguments
                [
                    self._X,
                    self._U,
                    self._opti.lam_g,
                    self._slack,
                    self._collision_multipliers,
                    ca.vcat(list(self.opti_variables_debug.values())),
                ],
                # Input names
                [
                    "r_ref_pos",
                    "R_ref_orient",
                    "x0",
                    "X_init",
                    "U_init",
                    "Dual Var Guess",
                    "Stage Cost Weights",
                    "End Cost Weights",
                    "S_init",
                    "Collision Multipliers_init",
                ],
                # Output names
                [
                    "X_opt",
                    "U_opt",
                    "Dual Var",
                    "S_opt",
                    "Collision Multipliers_sol",
                    "Debug Values " + ", ".join(self.opti_variables_debug.keys()),
                ],
            )

    def _error_function_orientation_frobenius(self, R_current: ca.MX, R_ref: ca.MX) -> ca.MX:
        """Compute orientation error using squared Frobenius norm of rotation matrix difference.

        The Frobenius norm measures the "distance" between two rotation matrices.
        For rotation matrices R_current and R_ref, the error is:
            ||I - R_current * R_ref^T||_F^2 = trace((I - R_current * R_ref^T)^T * (I - R_current * R_ref^T))

        This is a smooth, differentiable measure that equals 0 when R_current = R_ref
        and increases as the rotations diverge.

        Args:
            R_current: Current rotation matrix (3x3)
            R_ref: Reference rotation matrix (3x3)

        Returns:
            Squared Frobenius norm of the rotation error (scalar)

        """
        # Compute difference matrix (identity when rotations match)
        R_diff: ca.MX = ca.MX.eye(3) - R_current @ R_ref.T  # pyright: ignore[reportArgumentType]

        # Compute squared Frobenius norm: ||R_diff||_F^2 = trace(R_diff^T * R_diff)
        return ca.trace(R_diff.T @ R_diff)

    def solve(self) -> None:
        """Solve the optimal control problem.

        This method solves the trajectory optimization problem using the current
        parameters (initial state, reference trajectory, etc.). It uses warm-starting
        from the previous solution to improve convergence speed.

        The solution is stored internally and can be retrieved using getter methods.

        Raises:
            ValueError: If an unknown compilation mode is specified
            RuntimeError: If solver is not loaded in "function" mode

        """
        if self._ref_pos is None:
            msg = "Position reference not set, set_reference must be called before solving"
            raise RuntimeError(msg)
        if self._ref_orient is None:
            msg = "Orientation reference not set, set_reference must be called before solving"
            raise RuntimeError(msg)
        if self._x0_value is None:
            msg = "Initial state not set, set_x0 must be called before solving"
            raise RuntimeError(msg)

        match self._config.solver.casadi_mode:
            case CasadiMode.OPTI:
                # Standard Opti interface: set initial guess and solve
                self._opti.set_initial(self._X, self._x_sol)
                self._opti.set_initial(self._U, self._u_sol)
                self._opti.set_initial(self._opti.lam_g, self._lam_g_sol)
                self._opti.set_initial(self._slack, self._s_sol)
                self._opti.set_initial(self._collision_multipliers, self._col_multipliers_sol)

                start = time.perf_counter()
                sol = self._opti.solve()
                print(f"solving took {(time.perf_counter() - start) * 1000} ms")

                # Extract solution
                self.current_sol = sol
                self._x_sol = sol.value(self._X)
                self._u_sol = sol.value(self._U)
                self._lam_g_sol = sol.value(self._opti.lam_g)
                self._s_sol = sol.value(self._slack)
                self._col_multipliers_sol = sol.value(self._collision_multipliers)
                self._debug_values.update(
                    [(key, sol.value(variable)) for key, variable in self.opti_variables_debug.items()]
                )

            case CasadiMode.FUNCTION:
                # Compiled function mode: call the precompiled solver

                # get value of weights from config and assemble into vector (stage then end)
                stage_weight_params = ca.vcat(
                    [
                        getattr(self._config.problem.stage_cost, cost_name)
                        for cost_name in self._stage_cost_weights_param
                    ]
                )
                end_weight_params = ca.vcat(
                    [getattr(self._config.problem.end_cost, cost_name) for cost_name in self._end_cost_weights_param]
                )

                start = time.perf_counter()

                if self._loaded_solver is None:
                    # this should never happen as a solver is loaded during initialization for all compilation levels
                    # except "opti"
                    msg = "No solver is loaded"
                    raise RuntimeError(msg)

                # Call compiled solver function
                (
                    self._x_sol,
                    self._u_sol,
                    self._lam_g_sol,
                    self._s_sol,
                    self._col_multipliers_sol,
                    debug_values,
                ) = self._loaded_solver(
                    self._ref_pos,
                    ca.hcat(self._ref_orient),
                    self._x0_value,
                    self._x_sol,
                    self._u_sol,
                    self._lam_g_sol,
                    stage_weight_params,
                    end_weight_params,
                    self._s_sol,
                    self._col_multipliers_sol,
                )  # pyright: ignore[reportGeneralTypeIssues]

                needed_time = time.perf_counter() - start
                print(f"solving took {needed_time * 1000} ms")

                self._debug_values.update(
                    zip(self.opti_variables_debug.keys(), np.array(debug_values).flatten().tolist(), strict=True)
                )

            case unknown_compilation:
                msg = f"Wrong casadi compilation level {unknown_compilation}. Have a look at config file."
                raise ValueError(msg)

    def get_constraint_violations(self) -> list[tuple[int, float, float, float, str]]:
        """Identify constraints that are violated in the current solution.

        This is useful for debugging infeasible or poorly performing solutions.

        Returns:
            List of tuples, each containing:
            - Constraint index
            - Constraint value at solution
            - Lower bound
            - Upper bound
            - Constraint label (human-readable name)

        """
        # Extract constraint values at the solution
        constraint_values = np.array(self._opti.debug.value(self._opti.g))

        # Extract constraint bounds
        lbg = np.array(self._opti.debug.value(self._opti.lbg))
        ubg = np.array(self._opti.debug.value(self._opti.ubg))

        # Find violated constraints (values outside bounds)
        return [
            (
                i,
                constraint_values[i],
                lbg[i],
                ubg[i],
                self._constraint_labels.get(i, f"Unknown Constraint {i}"),
            )
            for i in range(len(constraint_values))
            if not (lbg[i] <= constraint_values[i] <= ubg[i])
        ]

    def print_constraint_violations(self) -> None:
        """Print a formatted report of constraint violations to the console.

        This is a convenience method for debugging constraint violations.
        Prints the constraint name, actual value, and bounds for each violated constraint.
        """
        violations = self.get_constraint_violations()
        if violations:
            print("\n--- Constraint Violations ---")
            for _, val, lb, ub, label in violations:
                print(f"{label}: Value={val}, Bounds=({lb}, {ub})")
        else:
            print("\nNo constraint violations.")

    def set_x0(self, x0_value: np.ndarray) -> None:
        """Set the initial state for the trajectory optimization.

        This is typically the current robot state (joint positions and velocities).

        Args:
            x0_value: Initial state vector containing [q, dq] where:
                      q = joint positions, dq = joint velocities

        """
        self._x0_value = x0_value
        if self._config.solver.casadi_mode == CasadiMode.OPTI:
            self._opti.set_value(self._x0_param, x0_value)

    def set_reference(self, ref_pos: np.ndarray, ref_quat: np.ndarray) -> None:
        """Set the reference trajectory for the end-effector.

        The planner will try to minimize tracking error with respect to this reference.

        Args:
            ref_pos: Reference positions (3, H) containing [x, y, z] coordinates
                     for each time step in the prediction horizon
            ref_quat: Reference quaternions (4, H) in scalar-first format [w, x, y, z]
                      representing desired end-effector orientations

        """
        self._ref_pos = ref_pos
        self._ref_orient = []

        # Convert quaternions to rotation matrices
        for k in range(ref_quat.shape[1]):
            rotation = Rotation.from_quat(ref_quat[:, k], scalar_first=True)
            rotmat = rotation.as_matrix()
            self._ref_orient.append(rotmat)

            # Set parameter values (only needed for "opti" mode)
            if self._config.solver.casadi_mode == CasadiMode.OPTI:
                for i in range(3):
                    for j in range(3):
                        self._opti.set_value(self._ref_orient_param[k][i, j], rotmat[i, j])

        if self._config.solver.casadi_mode == CasadiMode.OPTI:
            self._opti.set_value(self._ref_pos_param, ref_pos)

    def get_x_sol(self) -> np.ndarray:
        """Get the optimal state trajectory from the last solve.

        Returns:
            State trajectory of shape (nx, H+1) containing joint positions and velocities
            for all time steps from 0 to H

        """
        return np.array(self._x_sol)

    def get_u_sol(self) -> np.ndarray:
        """Get the optimal control input trajectory from the last solve.

        Returns:
            Control input trajectory of shape (nu, H) containing joint accelerations
            for time steps 0 to H-1

        """
        return np.array(self._u_sol)

    def get_x_sol_debug(self) -> np.ndarray | None:
        """Get the current value of state variables (for debugging).

        This retrieves the current value from the Opti object, which may be useful
        when debugging infeasible problems or examining intermediate solutions.

        Returns:
            Current state trajectory or None if not available

        """
        return self._opti.debug.value(self._X)

    def get_u_sol_debug(self) -> np.ndarray | None:
        """Get the current value of control input variables (for debugging).

        Returns:
            Current control input trajectory or None if not available

        """
        return self._opti.debug.value(self._U)

    def get_r_sol(self) -> np.ndarray | None:
        """Get the end-effector position trajectory from the last solve.

        This computes the forward kinematics for each state in the optimal trajectory
        to obtain the Cartesian positions of the tool center point (TCP).

        Returns:
            End-effector position trajectory of shape (3, H+1) or None if no solution exists

        """
        if np.all(self._x_sol == 0):
            return None

        # Extract joint positions from state trajectory
        q_last = self._x_sol[0 : self._config.robot.nq, :]

        # Compute forward kinematics for all time steps
        r_sol: ca.MX = self._system_model.casadi_fk_pos_function(q_last)  # pyright: ignore[reportAssignmentType]
        return np.array(r_sol.full())

    def get_local_acc_sol(self) -> np.ndarray | None:
        """Get the local acceleration including gravity (in TCP frame) from the last solve.

        This is the acceleration experienced in the tool frame, which is important
        for anti-slosh applications (e.g., transporting liquids without spilling).

        Returns:
            Local acceleration trajectory of shape (3, H) or None if no solution exists.
            Each column contains [ax, ay, az] in the TCP frame.

        """
        if np.all(self._x_sol == 0):
            return None

        # Extract states and controls
        xk = self._x_sol[0 : 2 * self._config.robot.nq, :][:, :-1]  # All states except terminal
        uk = self._u_sol

        # Compute local acceleration for each time step
        acc_list = []
        for i in range(uk.shape[1]):
            dotdot_pos_lat_k: ca.DM = self._system_model.casadi_local_acc_gravity_function(
                xk[:, i], uk[:, i]
            )  # pyright: ignore[reportAssignmentType]
            acc_list.append([float(dotdot_pos_lat_k[dim]) for dim in range(3)])

        return np.array(acc_list).T  # Transpose to get shape (3, H)

    def get_rotation_matrix_sol(self) -> np.ndarray | None:
        """Get the end-effector orientation trajectory from the last solve as rotation matrices.

        This computes the forward kinematics rotation matrices for each state
        in the optimal trajectory.

        Returns:
            Rotation matrix trajectory of shape (H+1, 3, 3) or None if no solution exists.
            Each element R[k, :, :] is a 3x3 rotation matrix for time step k.

        """
        if np.all(self._x_sol == 0):
            return None

        # Extract joint positions from state trajectory
        q_sol = self._x_sol[0 : self._config.robot.nq, :]

        # Compute rotation matrices for all time steps
        R_list = [
            np.array(self._system_model.casadi_fk_rotation_matrix_function(q_sol[:, i])) for i in range(q_sol.shape[1])
        ]
        return np.stack(R_list, axis=0)

    def get_quat_sol(self) -> np.ndarray | None:
        """Get the end-effector orientation trajectory from the last solve as quaternions.

        This computes the forward kinematics quaternions for each state
        in the optimal trajectory.

        Returns:
            Quaternion trajectory of shape (4, H+1) or None if no solution exists.
            Each element quat[:, k] is a quaternion for time step k in scalar-first format [w, x, y, z].

        """
        if np.all(self._x_sol == 0):
            return None

        # Extract joint positions from state trajectory
        q_sol = self._x_sol[0 : self._config.robot.nq, :]

        # Compute rotation matrices for all time steps
        quat_list = []
        for i in range(q_sol.shape[1]):
            rotmat = np.array(self._system_model.casadi_fk_rotation_matrix_function(q_sol[:, i]))
            quat_list.append(Rotation.from_matrix(rotmat).as_quat(scalar_first=True))
        return np.stack(quat_list, axis=1)

    def get_debug_values(self) -> OrderedDict[str, float]:
        """Get the values of the debug variables (e.g. cost terms) for the last solution."""
        return self._debug_values.copy()

    @property
    def prediction_horizon(self) -> int:
        """The prediction horizon (future time steps considered in planning).

        Cannot be modified after object creation, is determined by the config.
        """
        return self._config.problem.prediction_horizon
