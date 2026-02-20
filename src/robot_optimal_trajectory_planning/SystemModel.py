"""System Model for Robot Kinematics and Dynamics.

This module provides CasADi-based symbolic functions for robot modeling, including:
- Forward kinematics (position and orientation)
- Denavit-Hartenberg (DH) parameter transformations
- Linear discrete-time dynamics model
- Jacobian and manipulability computations
- Local acceleration transformations for anti-slosh control
"""

from functools import cached_property

import casadi as ca

from robot_optimal_trajectory_planning.collision import CollisionModel, model_registry
from robot_optimal_trajectory_planning.config import MpcConfig, load_config


class SystemModel:
    """Robot system model providing symbolic kinematics and dynamics functions.

    This class creates CasADi symbolic functions for robot modeling and control.
    It uses the Denavit-Hartenberg (DH) convention to represent the robot kinematics
    and provides cached properties for efficient repeated access to symbolic functions.

    The model supports:
    - Forward kinematics computation for all links and end-effector
    - Custom tool transformations (translation and rotation)
    - Linear discrete-time dynamics for MPC
    - Manipulability measures via Jacobian determinant
    - Local frame acceleration for anti-slosh objectives
    - Collision model integration

    Attributes:
    - collision_model: Collision constraint model instance

    """

    def __init__(self, config: MpcConfig | None = None) -> None:
        """Initialize the system model with configuration parameters.

        Args:
            config: Configuration object containing robot parameters and DH table.
                    If None, loads default configuration.

        """
        if config is None:
            config = load_config()  # Load default configuration
        self._config: MpcConfig = config

        # Initialize collision model based on configuration
        collision_model_type = model_registry.get_model(
            "dummy"  # No collision checking
            if self._config.problem.ignore_collisions
            else self._config.robot.collision_model
        )
        self.collision_model: CollisionModel = collision_model_type(self.casadi_fk_links_functions, self._config)

        # Note: CasADi functions are not initialized here - they are cached properties
        # that are computed once on first access, then reused for efficiency

    @cached_property
    def casadi_fk_links_functions(self) -> list[ca.Function]:
        """Generate forward kinematics functions for all robot links.

        This method creates a CasADi function for each link's transformation matrix
        using the Denavit-Hartenberg (DH) convention. The transformation from link i-1
        to link i is computed using the DH parameters (d, theta, r, alpha).

        The DH transformation matrix is:
            T_i = [cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)   r*cos(θ)]
                  [sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)   r*sin(θ)]
                  [0        sin(α)          cos(α)         d       ]
                  [0        0               0              1       ]

        The final transformation includes a custom tool frame transformation.

        Returns:
            List of CasADi functions, one for each link (including base and end-effector).
            Each function takes joint angles q as input and returns a 4×4 homogeneous
            transformation matrix.

        """
        nq = self._config.robot.nq
        q_sym: ca.SX = ca.SX.sym("q", nq, 1)  # pyright: ignore[reportArgumentType]

        # Start with identity transformation (base frame)
        T: ca.SX = ca.SX.eye(4)  # pyright: ignore[reportArgumentType]

        T_links: list[ca.Function] = []
        T_links.append(ca.Function("T_base", [q_sym], [T]).expand())

        # Accumulate DH transformations for each joint
        for joint, (d, theta_0, r, alpha) in enumerate(self._config.robot.dh_parameters):
            # Add joint angle to DH theta parameter
            theta = theta_0 + q_sym[joint]

            # Precompute trigonometric functions for efficiency
            ct = ca.cos(theta)
            st = ca.sin(theta)
            caa = ca.cos(alpha)
            sa = ca.sin(alpha)

            # Build DH transformation matrix for this joint
            T_i = ca.SX(4, 4)
            # Rotation and translation components
            T_i[0, 0] = ct
            T_i[0, 1] = -st * caa
            T_i[0, 2] = st * sa
            T_i[0, 3] = r * ct

            T_i[1, 0] = st
            T_i[1, 1] = ct * caa
            T_i[1, 2] = -ct * sa
            T_i[1, 3] = r * st

            T_i[2, 0] = 0
            T_i[2, 1] = sa
            T_i[2, 2] = caa
            T_i[2, 3] = d

            T_i[3, 0] = 0
            T_i[3, 1] = 0
            T_i[3, 2] = 0
            T_i[3, 3] = 1

            # Accumulate transformation: T = T_0 * T_1 * ... * T_i
            T = T @ T_i
            T_links.append(ca.Function(f"T_{joint + 1}", [q_sym], [T]).expand())

        # Apply custom tool transformation from configuration
        # This allows for custom end-effector frames (e.g., gripper, sensor)

        # Build 4x4 homogeneous tool transformation matrix
        tool = ca.SX.eye(4)  # pyright: ignore[reportArgumentType]
        tool[:3, :3] = ca.SX(self._config.robot.custom_tool_rotation_matrix)
        tool[:3, 3] = ca.SX(self._config.robot.custom_tool_translation)

        # Multiply with tool transform to get final end-effector frame
        T = T @ tool
        T_links.append(ca.Function("T_end_effector", [q_sym], [T]).expand())

        return T_links

    @cached_property
    def casadi_fk_pos_function(self) -> ca.Function:
        """Generate forward kinematics function for end-effector position.

        This extracts the translation component (first 3 elements of the 4th column)
        from the end-effector transformation matrix.

        Returns:
            CasADi function that takes joint angles q (nq × 1) and returns
            end-effector position [x, y, z] (3 × 1) in the base frame.

        """
        q_sym: ca.SX = ca.SX.sym("q", self._config.robot.nq, 1)  # pyright: ignore[reportArgumentType]
        T_end_effector: ca.SX = self.casadi_fk_links_functions[-1](q_sym)  # pyright: ignore[reportAssignmentType]

        # Extract position from transformation matrix (translation component)
        position = T_end_effector[:3, 3]

        return ca.Function("forward_kin_pos", [q_sym], [position]).expand()

    @cached_property
    def casadi_fk_rotation_matrix_function(self) -> ca.Function:
        """Generate forward kinematics function for end-effector orientation.

        This extracts the rotation matrix component (upper-left 3×3 submatrix)
        from the end-effector transformation matrix.

        Returns:
            CasADi function that takes joint angles q (nq × 1) and returns
            end-effector orientation as a 3×3 rotation matrix in the base frame.

        """
        q_sym: ca.SX = ca.SX.sym("q", self._config.robot.nq, 1)  # pyright: ignore[reportArgumentType]
        T_end_effector: ca.SX = self.casadi_fk_links_functions[-1](q_sym)  # pyright: ignore[reportAssignmentType]

        # Extract rotation matrix from transformation matrix (upper-left 3x3)
        rotmat = T_end_effector[:3, :3]

        return ca.Function("forward_kin_orient", [q_sym], [rotmat]).expand()

    @cached_property
    def casadi_linear_robot_model_function(self) -> ca.Function:
        """Generate discrete-time linear dynamics model for the robot.

        This implements a simple double integrator model for each joint:
            q_{k+1} = q_k + Δt * dq_k + (Δt²/2) * u_k
            dq_{k+1} = dq_k + Δt * u_k

        where:
            q = joint positions
            dq = joint velocities
            u = joint accelerations (control input)
            Δt = sample_time

        This is a discrete-time approximation of the
        continuous dynamics d²q/dt² = u

        Returns:
            CasADi function that takes state-control vector [q, dq, u] and returns
            next state [q_next, dq_next].

        """
        nq = self._config.robot.nq
        sample_time = self._config.problem.sample_time

        # Create symbolic variables for current state and control
        q = ca.DM(0, 0)
        dq = ca.DM(0, 0)
        u = ca.DM(0, 0)
        for i in range(nq):
            q = ca.vertcat(q, ca.SX.sym(f"q{i}", 1, 1))  # pyright: ignore[reportArgumentType]
            dq = ca.vertcat(dq, ca.SX.sym(f"dq{i}", 1, 1))  # pyright: ignore[reportArgumentType]
            u = ca.vertcat(u, ca.SX.sym(f"u{i}", 1, 1))  # pyright: ignore[reportArgumentType]

        # Discrete-time update equations (semi-implicit Euler integration)
        q_next = (
            q
            + ((ca.SX.eye(nq) * sample_time) @ dq)  # pyright: ignore[reportArgumentType]
            + ((ca.SX.eye(nq) * sample_time**2 / 2) @ u)  # pyright: ignore[reportArgumentType]
        )
        dq_next = dq + ca.SX.eye(nq) * sample_time @ u  # pyright: ignore[reportArgumentType]

        # Combined state vector
        x_next = ca.vertcat(q_next, dq_next)

        # Stack all inputs into single argument vector
        args_ca = ca.vcat([q, dq, u])
        return ca.Function("DiscreteLinearRobotKinematics", [args_ca], [x_next])

    @cached_property
    def casadi_continuous_linear_robot_model_function(self) -> ca.Function:
        """Generate continuous-time linear dynamics model for the robot.

        This implements the continuous-time double integrator:
            d²q/dt² = u  (acceleration is control input)

        Returns:
            CasADi function that takes state-control vector [q, dq, u] and returns
            state derivative [dq, ddq].

        """
        nq = self._config.robot.nq

        # Create symbolic variables
        q = ca.DM(0, 0)
        dq = ca.DM(0, 0)
        u = ca.DM(0, 0)
        for i in range(nq):
            q = ca.vertcat(q, ca.SX.sym(f"q{i}", 1, 1))  # pyright: ignore[reportArgumentType]
            dq = ca.vertcat(dq, ca.SX.sym(f"dq{i}", 1, 1))  # pyright: ignore[reportArgumentType]
            u = ca.vertcat(u, ca.SX.sym(f"u{i}", 1, 1))  # pyright: ignore[reportArgumentType]

        # Continuous-time dynamics: dx/dt = f(x, u)
        q_dot = dq  # dq/dt = velocity
        dq_dot = u  # d²q/dt² = acceleration (control input)
        x_dot = ca.vertcat(q_dot, dq_dot)

        args_ca = ca.vcat([q, dq, u])
        return ca.Function("ContinuousLinearRobotKinematics", [args_ca], [x_dot])

    @cached_property
    def casadi_jacobian_absolute_determinant_ur(self) -> ca.Function:
        """Generate manipulability measure function for Universal Robots (UR) arms.

        This computes the absolute value of the Jacobian determinant, which is a
        scalar measure of manipulability. Higher values indicate better dexterity
        and being further from singularities.

        The analytical formula used here is specific to UR robots:
            |det(J)| = |s₃ * s₅ * a₂ * a₃ * (c₂*a₂ + c₂₃*a₃ + s₂₃₄*d₅)|

        where:
            sᵢ = sin(qᵢ), cᵢ = cos(qᵢ)
            aᵢ = link length (DH parameter r)
            dᵢ = link offset (DH parameter d)
            c₂₃ = cos(q₂ + q₃), etc.

        **Note**: This function only works for Universal Robots (UR3, UR5, UR10, etc.)
        due to the specific analytical formula derived for their kinematic structure.

        Returns:
            CasADi function that takes joint configuration q (6 × 1) and returns
            the absolute value of the Jacobian determinant (scalar).

        """
        DH = ca.SX(self._config.robot.dh_parameters)
        r = DH[:, 2]  # Link lengths (sometimes called 'a' in literature)
        d = DH[:, 0]  # Link offsets

        configuration: ca.SX = ca.SX.sym("configuration", 6, 1)  # pyright: ignore[reportArgumentType] # Joint angles

        # Analytical formula for UR robots (zero-based indexing in code)
        # Formula: det(J) = s₃ * s₅ * a₂ * a₃ * (c₂*a₂ + c₂₃*a₃ + s₂₃₄*d₅)
        jac_determinant = (
            ca.sin(configuration[2])  # sin(q₃)
            * ca.sin(configuration[4])  # sin(q₅)
            * r[1]  # a₂
            * r[2]  # a₃
            * (
                ca.cos(configuration[1]) * r[1]  # cos(q₂) * a₂
                + ca.cos(configuration[1] + configuration[2]) * r[2]  # cos(q₂+q₃) * a₃
                + ca.sin(configuration[1] + configuration[2] + configuration[3]) * d[4]  # sin(q₂+q₃+q₄) * d₅
            )
        )

        # Return absolute value (manipulability is always non-negative)
        return ca.Function("Absolute_Jacobian_Determinant", [configuration], [ca.fabs(jac_determinant)])

    @cached_property
    def casadi_local_acc_gravity_function(self) -> ca.Function:
        """Generate function for computing acceleration in the end-effector frame.

        This function computes the acceleration experienced in the tool/end-effector
        coordinate frame, accounting for both motion-induced acceleration and gravity.
        This is critical for anti-slosh objectives (e.g., transporting liquids).

        The computation:
        1. Computes end-effector position p_end(q) using forward kinematics
        2. Computes velocity: dp_end = ∂p_end/∂q * dq (using Jacobian)
        3. Computes acceleration: ddp_end = d²p_end/dt² (using automatic differentiation)
        4. Transforms to local frame: a_local = R^T * (ddp_end - g)

        where:
            R = end-effector rotation matrix
            g = gravity vector [0, 0, -9.81] m/s²

        The result gives the acceleration felt in the tool frame, which for liquid
        transport should ideally be close to [0, 0, 9.81] (stationary in world frame).
        Large lateral components (x, y) cause sloshing.

        Returns:
            CasADi function with inputs:
                - x: State vector [q, dq] (2*nq × 1)
                - u: Control input (joint accelerations) ddq (nq × 1)
            and output:
                - local_acceleration_gravity: 3D acceleration in tool frame [ax, ay, az]

        """
        nq: int = self._config.robot.nq

        # Create symbolic variables for state and control
        q = ca.SX.sym("q", nq, 1)  # pyright: ignore[reportArgumentType] # Joint positions
        dq = ca.SX.sym("dq", nq, 1)  # pyright: ignore[reportArgumentType] # Joint velocities
        ddq = ca.SX.sym("ddq", nq, 1)  # pyright: ignore[reportArgumentType] # Joint accelerations (control input)

        # Get end-effector transformation and extract rotation and position
        T_end: ca.SX = self.casadi_fk_links_functions[-1](q)  # pyright: ignore[reportAssignmentType]
        R = T_end[:3, :3]  # Rotation matrix (world to end-effector)
        p_end = T_end[:3, 3]  # Position vector

        # Gravity vector in world frame
        gravity = ca.vcat([0, 0, -9.81])

        # Compute end-effector velocity using automatic differentiation (Jacobian)
        # dp_end/dt = J(q) * dq where J = ∂p_end/∂q
        dp_end = ca.jtimes(p_end, q, dq)

        # Compute end-effector acceleration using automatic differentiation
        # ddp_end/dt² = ∂(dp_end)/∂[q,dq] * [dq, ddq]
        ddp_end = ca.jtimes(dp_end, ca.vcat([q, dq]), ca.vcat([dq, ddq]))

        # Transform acceleration to local end-effector frame
        # Subtract gravity to get the "felt" acceleration
        local_acceleration_gravity: ca.SX = R.T @ (ddp_end - gravity)

        # Package state vector
        x = ca.vcat([q, dq])

        return ca.Function(
            "local_acc_gravity",
            [x, ddq],
            [local_acceleration_gravity],
            ["x", "u"],  # Input names
            ["local_acceleration_gravity"],  # Output name
        ).expand()
