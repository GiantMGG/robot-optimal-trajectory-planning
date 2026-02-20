from abc import ABC, abstractmethod
from enum import Enum

import casadi as ca
import numpy as np

from robot_optimal_trajectory_planning.config import MpcConfig, load_config


class CollisionObject(ABC):
    @abstractmethod
    def transformed(self, transformation_matrix) -> "CollisionObject":
        pass


class Sphere(CollisionObject):
    def __init__(self, radius=1.0, center=None):
        self.radius = radius
        self.center = ca.vcat([0, 0, 0]) if center is None else center

    def transformed(self, transformation_matrix) -> "Sphere":
        new_sphere = Sphere()
        new_sphere.center = (transformation_matrix @ ca.vcat([self.center, 1]))[:3]
        new_sphere.radius = self.radius
        return new_sphere


class Capsule(CollisionObject):
    def __init__(self, radius=1.0, start_point=None, end_point=None):
        self.start_point = ca.vcat([0, 0, 0]) if start_point is None else start_point
        self.end_point = ca.vcat([0, 0, 0]) if end_point is None else end_point
        self.radius = radius

    def transformed(self, transformation_matrix) -> "Capsule":
        new_capsule = Capsule()
        new_capsule.start_point = (transformation_matrix @ ca.vcat([self.start_point, 1]))[:3]
        new_capsule.end_point = (transformation_matrix @ ca.vcat([self.end_point, 1]))[:3]
        new_capsule.radius = self.radius
        return new_capsule


class ConvexPolytope(CollisionObject):
    def __init__(self, vertices: list | None = None, safety_distance=0.0):
        self.vertices = [] if vertices is None else vertices
        self.safety_distance = safety_distance

    def transformed(self, transformation_matrix) -> "ConvexPolytope":
        new_polytope = ConvexPolytope(safety_distance=self.safety_distance)
        for vertex in self.vertices:
            new_polytope.vertices.append((transformation_matrix @ ca.vcat([vertex, 1]))[:3])
        return new_polytope


class HalfSpace(CollisionObject):
    def __init__(self, normal_vector=None, offset=0.0):
        self.normal_vector = ca.vcat([1.0, 0.0, 0.0]) if normal_vector is None else normal_vector
        self.offset = offset

    def transformed(self, transformation_matrix) -> "HalfSpace":
        rotmat = transformation_matrix[:3, :3]
        new_halfspace = HalfSpace(normal_vector=rotmat @ self.normal_vector, offset=self.offset)
        return new_halfspace


class CollisionGroup(CollisionObject):
    def __init__(self, collision_objects: list | None = None):
        self.collision_objects = [] if collision_objects is None else collision_objects

    def transformed(self, transformation_matrix) -> "CollisionGroup":
        return CollisionGroup([obj.transformed(transformation_matrix) for obj in self.collision_objects])


class CollisionPairHandler(ABC):
    def get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        """Returns a list of casadi expressions which should all be >= 0 to satisfy the constraints"""
        # sort if necessary
        obj1, obj2 = sorted((obj1, obj2), key=lambda obj: type(obj).__name__)
        return self._get_constraint_terms(obj1, obj2, collision_multipliers)

    def get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        # sort if necessary
        obj1, obj2 = sorted((obj1, obj2), key=lambda obj: type(obj).__name__)
        return self._get_collision_multiplier_count(obj1, obj2)

    @abstractmethod
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        pass

    @abstractmethod
    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        pass


class CollisionGroupHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        multiplier_count = 0

        if isinstance(obj1, CollisionGroup) and isinstance(obj2, CollisionGroup):
            # both are groups
            for subobj1 in obj1.collision_objects:
                for subobj2 in obj2.collision_objects:
                    handler = handler_registry.get_handler(type(subobj1), type(subobj2))()
                    multiplier_count += handler.get_collision_multiplier_count(subobj1, subobj2)
        elif isinstance(obj1, CollisionGroup):
            # only obj1 is group
            for subobj in obj1.collision_objects:
                handler = handler_registry.get_handler(type(subobj), type(obj2))()
                multiplier_count += handler.get_collision_multiplier_count(subobj, obj2)
        elif isinstance(obj2, CollisionGroup):
            # only obj2 is group
            for subobj in obj2.collision_objects:
                handler = handler_registry.get_handler(type(obj1), type(subobj))()
                multiplier_count += handler.get_collision_multiplier_count(obj1, subobj)
        else:
            raise TypeError("None of the collision objects is a CollisionGroup")

        return multiplier_count

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        constraint_terms = []
        multiplier_vector_start_index = 0

        if isinstance(obj1, CollisionGroup) and isinstance(obj2, CollisionGroup):
            # both are groups
            for subobj1 in obj1.collision_objects:
                for subobj2 in obj2.collision_objects:
                    handler = handler_registry.get_handler(type(subobj1), type(subobj2))()
                    num_multipliers_pair = handler.get_collision_multiplier_count(subobj1, subobj2)
                    multipliers = collision_multipliers[
                        multiplier_vector_start_index : multiplier_vector_start_index + num_multipliers_pair
                    ]
                    multiplier_vector_start_index += num_multipliers_pair
                    constraint_terms += handler.get_constraint_terms(subobj1, subobj2, multipliers)
        elif isinstance(obj1, CollisionGroup):
            # only obj1 is group
            for subobj in obj1.collision_objects:
                handler = handler_registry.get_handler(type(subobj), type(obj2))()
                num_multipliers_pair = handler.get_collision_multiplier_count(subobj, obj2)
                multipliers = collision_multipliers[
                    multiplier_vector_start_index : multiplier_vector_start_index + num_multipliers_pair
                ]
                multiplier_vector_start_index += num_multipliers_pair
                constraint_terms += handler.get_constraint_terms(subobj, obj2, multipliers)
        elif isinstance(obj2, CollisionGroup):
            # only obj2 is group
            for subobj in obj2.collision_objects:
                handler = handler_registry.get_handler(type(obj1), type(subobj))()
                num_multipliers_pair = handler.get_collision_multiplier_count(obj1, subobj)
                multipliers = collision_multipliers[
                    multiplier_vector_start_index : multiplier_vector_start_index + num_multipliers_pair
                ]
                multiplier_vector_start_index += num_multipliers_pair
                constraint_terms += handler.get_constraint_terms(obj1, subobj, multipliers)
        else:
            raise TypeError("None of the collision objects is a CollisionGroup")

        return constraint_terms


class CollisionHandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[tuple[str, ...], type[CollisionPairHandler]] = {}

    def register(self, type1: type[CollisionObject], type2: type[CollisionObject]):
        def decorator(handler: type[CollisionPairHandler]) -> type[CollisionPairHandler]:
            """The function that is applied to the registered class"""
            self._handlers[tuple(sorted([type1.__name__, type2.__name__]))] = handler
            return handler

        return decorator

    def get_handler(self, type1: type[CollisionObject], type2: type[CollisionObject]) -> type[CollisionPairHandler]:
        if issubclass(type1, CollisionGroup) or issubclass(type2, CollisionGroup):
            # treat collision groups seperately, which is implemented in CollisionGroupHandler
            return CollisionGroupHandler
        return self._handlers[tuple(sorted([type1.__name__, type2.__name__]))]


handler_registry = CollisionHandlerRegistry()


@handler_registry.register(Sphere, Sphere)
class SphereSphereHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 0

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, Sphere)
        assert isinstance(obj2, Sphere)
        vector_between_centers = obj1.center - obj2.center
        center_distance_squared = vector_between_centers.T @ vector_between_centers
        constraint_terms = [center_distance_squared - (obj1.radius + obj2.radius) ** 2]
        return constraint_terms


@handler_registry.register(HalfSpace, Sphere)
class HalfSpaceSphereHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 0

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, HalfSpace)
        assert isinstance(obj2, Sphere)
        constraint_terms = [obj1.normal_vector.T @ obj2.center - obj1.offset - obj2.radius]
        return constraint_terms


@handler_registry.register(Capsule, Capsule)
class CapsuleCapsuleHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 5

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, Capsule)
        assert isinstance(obj2, Capsule)
        distance_vector_dual = collision_multipliers[:3]
        ineq_constraint_dual = collision_multipliers[3:5]
        constraint_terms = [
            -0.25 * distance_vector_dual.T @ distance_vector_dual
            - ineq_constraint_dual[0]
            - ineq_constraint_dual[1]
            - (obj1.radius + obj2.radius) ** 2,
            ca.hcat([obj1.start_point, obj1.end_point]).T @ distance_vector_dual
            + ineq_constraint_dual[0] * ca.vcat([1, 1]),
            -ca.hcat([obj2.start_point, obj2.end_point]).T @ distance_vector_dual
            + ineq_constraint_dual[1] * ca.vcat([1, 1]),
        ]
        return constraint_terms


@handler_registry.register(Capsule, Sphere)
class CapsuleSphereHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 4

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, Capsule)
        assert isinstance(obj2, Sphere)
        distance_vector_dual = collision_multipliers[:3]
        ineq_constraint_dual = collision_multipliers[3]
        return [
            -0.25 * distance_vector_dual.T @ distance_vector_dual
            - distance_vector_dual.T @ obj2.center
            - ineq_constraint_dual
            - (obj1.radius + obj2.radius) ** 2,
            ca.hcat([obj1.start_point, obj1.end_point]).T @ distance_vector_dual
            + ineq_constraint_dual * ca.vcat([1, 1]),
        ]


@handler_registry.register(Capsule, HalfSpace)
class CapsuleHalfSpaceHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 0

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, Capsule)
        assert isinstance(obj2, HalfSpace)
        constraint_terms = [
            obj2.normal_vector.T @ obj1.start_point - obj2.offset - obj1.radius,
            obj2.normal_vector.T @ obj1.end_point - obj2.offset - obj1.radius,
        ]
        return constraint_terms


@handler_registry.register(ConvexPolytope, Sphere)
class ConvexPolytopeSphereHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 4

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, ConvexPolytope)
        assert isinstance(obj2, Sphere)
        distance_vector_dual = collision_multipliers[:3]
        ineq_constraint_dual = collision_multipliers[3]
        return [
            -0.25 * distance_vector_dual.T @ distance_vector_dual
            - distance_vector_dual.T @ obj2.center
            - ineq_constraint_dual
            - (obj1.safety_distance + obj2.radius) ** 2,
            ca.hcat(obj1.vertices).T @ distance_vector_dual + ineq_constraint_dual * ca.vcat([1] * len(obj1.vertices)),
        ]


@handler_registry.register(ConvexPolytope, ConvexPolytope)
class ConvexPolytopeConvexPolytopeHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 5

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, ConvexPolytope)
        assert isinstance(obj2, ConvexPolytope)
        distance_vector_dual = collision_multipliers[:3]
        ineq_constraint_dual = collision_multipliers[3:5]
        return [
            -0.25 * distance_vector_dual.T @ distance_vector_dual
            - ineq_constraint_dual[0]
            - ineq_constraint_dual[1]
            - (obj1.safety_distance + obj2.safety_distance) ** 2,
            ca.hcat(obj1.vertices).T @ distance_vector_dual
            + ineq_constraint_dual[1] * ca.vcat([1] * len(obj1.vertices)),
            -ca.hcat(obj2.vertices).T @ distance_vector_dual
            + ineq_constraint_dual[0] * ca.vcat([1] * len(obj2.vertices)),
        ]


@handler_registry.register(Capsule, ConvexPolytope)
class CapsuleConvexPolytopeHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 5

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, Capsule)
        assert isinstance(obj2, ConvexPolytope)
        distance_vector_dual = collision_multipliers[:3]
        ineq_constraint_dual = collision_multipliers[3:5]
        return [
            -0.25 * distance_vector_dual.T @ distance_vector_dual
            - ineq_constraint_dual[0]
            - ineq_constraint_dual[1]
            - (obj1.radius + obj2.safety_distance) ** 2,
            -ca.hcat([obj1.start_point, obj1.end_point]).T @ distance_vector_dual
            + ineq_constraint_dual[1] * ca.vcat([1, 1]),
            ca.hcat(obj2.vertices).T @ distance_vector_dual
            + ineq_constraint_dual[0] * ca.vcat([1] * len(obj2.vertices)),
        ]


@handler_registry.register(ConvexPolytope, HalfSpace)
class ConvexPolytopeHalfSpaceHandler(CollisionPairHandler):
    def _get_collision_multiplier_count(self, obj1: CollisionObject, obj2: CollisionObject) -> int:
        return 0

    def _get_constraint_terms(
        self, obj1: CollisionObject, obj2: CollisionObject, collision_multipliers: ca.SX
    ) -> list[ca.SX]:
        assert isinstance(obj1, ConvexPolytope)
        assert isinstance(obj2, HalfSpace)
        constraint_terms = []
        for vertex in obj1.vertices:
            constraint_terms.append(obj2.normal_vector.T @ vertex - obj2.offset - obj1.safety_distance)
        return constraint_terms


example_transforms = {}

example_transforms["bottle"] = np.array([[1, 0, 0, 0.5], [0, 1, 0, -0.373], [0, 0, 1, 0], [0, 0, 0, 1]])

example_objects = {}


def generate_bottle_vertices(n=8):
    angles = np.linspace(0, 2 * np.pi, n)
    radius = 5e-2
    radius_lid = 1.5e-2
    h1 = 17e-2
    h2 = h1 + 6.5e-2
    vertices = [np.array([radius * np.cos(angle), radius * np.sin(angle), 0]) for angle in angles]
    vertices += [np.array([radius * np.cos(angle), radius * np.sin(angle), h1]) for angle in angles]
    vertices += [np.array([radius_lid * np.cos(angle), radius_lid * np.sin(angle), h2]) for angle in angles]
    vertices = [vertex.T for vertex in vertices]
    return vertices


JENGA_BLOCK_HEIGHT = 1.5e-2


def generate_single_jenga_tower():
    x, y, z = 617e-3, -124.5e-3, 0
    length, width, height = 90e-3, 90e-3, 280e-3
    vertices = [
        np.array([x, y, z]),
        np.array([x, y - width, z]),
        np.array([x + length, y - width, z]),
        np.array([x + length, y, z]),
        np.array([x, y, z + height]),
        np.array([x, y - width, z + height]),
        np.array([x + length, y, z + height]),
        np.array([x + length, y - width, z + height]),
    ]
    # Convert to column vectors
    vertices = [vertex.T for vertex in vertices]
    return vertices


def generate_jenga_vertices(position=None, orientation=None):
    """Generate vertices for a standard Jenga block as a ConvexPolytope.

    Standard Jenga block dimensions:
    - Length: 7.5 cm
    - Width: 2.5 cm
    - Height: 1.5 cm

    Returns:
        List of vertices representing a Jenga block centered at origin
    """
    length, width, height = 7.5e-2, 2.5e-2, JENGA_BLOCK_HEIGHT

    # Generate 8 vertices of the rectangular block
    vertices = [
        np.array([-length / 2, -width / 2, -height / 2]),  # bottom layer
        np.array([length / 2, -width / 2, -height / 2]),
        np.array([length / 2, width / 2, -height / 2]),
        np.array([-length / 2, width / 2, -height / 2]),
        np.array([-length / 2, -width / 2, height / 2]),  # top layer
        np.array([length / 2, -width / 2, height / 2]),
        np.array([length / 2, width / 2, height / 2]),
        np.array([-length / 2, width / 2, height / 2]),
    ]

    # Convert to column vectors
    vertices = [vertex.T for vertex in vertices]
    return vertices


def generate_jenga_spheres(center=None):
    """Generate a group of spheres approximating a Jenga block.

    Args:
        center: Optional center position for the Jenga block

    Returns:
        CollisionGroup of spheres representing a Jenga block
    """
    if center is None:
        center = ca.vcat([0, 0, 0])

    length, width, height = 7.5e-2, 2.5e-2, JENGA_BLOCK_HEIGHT

    # Use multiple spheres to approximate the block
    # We'll use 3 spheres along the length
    radius = min(width, height) / 2

    spheres = []
    # Place spheres along the length of the block
    for x_offset in [-length / 3, 0, length / 3]:
        spheres.append(Sphere(radius=radius, center=center + ca.vcat([x_offset, 0, 0])))

    return CollisionGroup(spheres)


def create_jenga_tower(num_layers=3, safety_distance=0.000):
    """Create a complete Jenga tower with alternating layer orientations.

    Args:
        num_layers: Number of layers in the tower
        safety_distance: Safety distance for the ConvexPolytope objects

    Returns:
        Dictionary of collision objects representing the Jenga tower
    """
    tower_objects = {}

    # Height of each layer
    layer_height = JENGA_BLOCK_HEIGHT  # 1.5 cm

    # Base position of the tower - aligned with generate_single_jenga_tower()
    # The single tower starts at (617e-3, -124.5e-3, 0) and has dimensions 85e-3 x 85e-3
    # We need to position our tower center to match this area
    tower_start_x, tower_start_y = 617e-3, -124.5e-3
    tower_width = 75e-3
    # Calculate the center of the tower's footprint
    base_position = np.array([tower_start_x + tower_width / 2, tower_start_y - tower_width / 2, layer_height / 2])

    for layer in range(num_layers):
        # Alternate layer orientation (x-aligned vs y-aligned)
        is_x_aligned = layer % 2 == 0

        # For each layer, create 3 blocks
        for block in range(3):
            # Position offset for this block within the layer
            if is_x_aligned:
                # Blocks aligned along x-axis
                offset = np.array([0, (block - 1) * 2.5e-2, 0])
                rotation = np.eye(3)  # No rotation needed
            else:
                # Blocks aligned along y-axis
                offset = np.array([(block - 1) * 2.5e-2, 0, 0])
                # Rotation matrix for 90 degrees around z-axis
                rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

            # Calculate position for this block
            position = base_position + offset + np.array([0, 0, layer * layer_height])

            # Create transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = position

            # Create objects
            block_name = f"jenga_block_l{layer}_b{block}"

            # Add vertex-based representation
            tower_objects[block_name] = ConvexPolytope(
                vertices=generate_jenga_vertices(), safety_distance=safety_distance
            ).transformed(transform)

            # Add sphere-based representation
            tower_objects[f"{block_name}_spheres"] = generate_jenga_spheres(center=ca.vcat([0, 0, 0])).transformed(
                transform
            )

    return tower_objects


# Add Jenga tower to example objects
jenga_tower = create_jenga_tower(num_layers=10)
example_objects.update(jenga_tower)
example_objects["single_jenga_tower"] = ConvexPolytope(generate_single_jenga_tower(), safety_distance=0.0)
example_objects["single_jenga_tower_spheres"] = CollisionGroup(
    [Sphere(radius=65e-3, center=ca.vcat([662e-3, -170e-3, z])) for z in np.linspace(0, 280e-3, 5)]
)


example_objects["bottle"] = ConvexPolytope(vertices=generate_bottle_vertices(n=8), safety_distance=0.0).transformed(
    example_transforms["bottle"]
)

example_objects["bottle_spheres"] = CollisionGroup(
    [
        Sphere(radius=8e-2, center=ca.vcat([0, 0, 5.75e-2])),
        Sphere(radius=8e-2, center=ca.vcat([0, 0, 17.25e-2])),
    ]
).transformed(example_transforms["bottle"])

polytope_info = []
if False:
    print("ConvexPolytope objects and their vertices:\n")
    for name, obj in example_objects.items():
        if isinstance(obj, ConvexPolytope):
            print(f"{name} vertices:\n{obj.vertices}\n")
        elif isinstance(obj, CollisionGroup):
            print(f"{name} contains the following collision objects:")
            for subobj in obj.collision_objects:
                if isinstance(subobj, Sphere):
                    print(f"Sphere center: {subobj.center}, radius: {subobj.radius}")
            print("\n")


class Link(Enum):
    BASE = 0
    SHOULDER = 1
    UPPER_ARM = 2
    FOREARM = 3
    WRIST_1 = 4
    WRIST_2 = 5
    WRIST_3 = 6
    CUSTOM_TOOL = 7


class CollisionModel(ABC):
    def __init__(self, casadi_fk_links_functions: list[ca.Function], config: MpcConfig | None = None) -> None:
        self._casadi_fk_links_functions: list[ca.Function] = casadi_fk_links_functions
        if config is None:
            config = load_config()  # Load default configuration
        self._config: MpcConfig = config

        self._collision_objects: dict[str, tuple[CollisionObject, Link]] = {}
        self._collision_pairs: list[tuple[str, str]] = []
        self._constraint_function: ca.Function

        obstacle_names = self._config.problem.obstacles
        self._obstacles: list[CollisionObject] = [example_objects[object_name] for object_name in obstacle_names]

        self._build_robot_collision_data()
        self._create_constraint_function()

    def get_constraint_function(self) -> ca.Function:
        """Get the collision constraint function.

        Returns a casadi function c such that
        c(configuration q, collision multipliers z) >= 0 is the collision constraint

        """
        return self._constraint_function

    def get_collision_multiplier_count(self) -> int:
        """Get the number of collision multipliers.

        Vector-Valued collision multipliers such as the "distance vector dual" ξ are counted as multiple collision
        multipliers, this method therefore returns the dimension of the collision multiplier vector (for a single time
        step).
        """
        count = 0
        for pair in self._collision_pairs:
            obj1, _ = self._collision_objects[pair[0]]
            obj2, _ = self._collision_objects[pair[1]]
            handler = handler_registry.get_handler(type(obj1), type(obj2))()
            count += handler.get_collision_multiplier_count(obj1, obj2)
        for robot_object, _ in self._get_all_moving_objects().values():
            for obstacle in self._obstacles:
                handler = handler_registry.get_handler(type(robot_object), type(obstacle))()
                count += handler.get_collision_multiplier_count(robot_object, obstacle)
        return count

    def set_obstacles(self, obstacles: list[CollisionObject]) -> None:
        self._obstacles = obstacles
        self._create_constraint_function()

    def extend_collision_objects(self, obstacles: list[CollisionObject]) -> None:
        self.set_obstacles(self._obstacles + obstacles)

    @abstractmethod
    def _build_robot_collision_data(self) -> None:
        pass

    def _create_constraint_function(self) -> None:
        nq = self._config.robot.nq
        q: ca.SX = ca.SX.sym("q", nq, 1)  # pyright: ignore[reportArgumentType]
        num_multipliers = self.get_collision_multiplier_count()
        z: ca.SX = ca.SX.sym("z", num_multipliers, 1)  # pyright: ignore[reportArgumentType]
        constraint_terms: list[ca.SX] = []

        multiplier_vector_start_index = 0
        for pair in self._collision_pairs:
            obj1, link1 = self._collision_objects[pair[0]]
            obj2, link2 = self._collision_objects[pair[1]]
            # we assume that these collision objects contain no symbolic variables
            # then we introduce the configuration q as a variable by applying the kinematic transformations
            obj1 = obj1.transformed(self._casadi_fk_links_functions[link1.value](q))
            obj2 = obj2.transformed(self._casadi_fk_links_functions[link2.value](q))
            handler = handler_registry.get_handler(type(obj1), type(obj2))()
            num_multipliers_pair = handler.get_collision_multiplier_count(obj1, obj2)
            multipliers = z[multiplier_vector_start_index : multiplier_vector_start_index + num_multipliers_pair]
            multiplier_vector_start_index += num_multipliers_pair
            constraint_terms += handler.get_constraint_terms(obj1, obj2, multipliers)
        for robot_object_local, link in self._get_all_moving_objects().values():
            for obstacle in self._obstacles:
                robot_object = robot_object_local.transformed(self._casadi_fk_links_functions[link.value](q))
                handler = handler_registry.get_handler(type(robot_object), type(obstacle))()
                num_multipliers_pair = handler.get_collision_multiplier_count(robot_object, obstacle)
                multipliers = z[multiplier_vector_start_index : multiplier_vector_start_index + num_multipliers_pair]
                multiplier_vector_start_index += num_multipliers_pair
                constraint_terms += handler.get_constraint_terms(robot_object, obstacle, multipliers)

        self._constraint_function = ca.Function("collision_constraints", [q, z], [ca.vcat(constraint_terms)])

    def _get_all_moving_objects(self) -> dict[str, tuple[CollisionObject, Link]]:
        return {name: (obj, link) for name, (obj, link) in self._collision_objects.items() if link != Link.BASE}


class CollisionModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, type[CollisionModel]] = {}

    def register(self, model_name: str):
        def decorator(model: type[CollisionModel]) -> type[CollisionModel]:
            """The function that is applied to the registered class."""
            self._models[model_name] = model
            return model

        return decorator

    def get_model(self, model_name: str) -> type[CollisionModel]:
        return self._models[model_name]


model_registry = CollisionModelRegistry()


@model_registry.register("dummy")
class DummyCollisionModel(CollisionModel):
    def _build_robot_collision_data(self) -> None:
        pass


@model_registry.register("capsules")
class CapsuleCollisionModel(CollisionModel):
    def _build_robot_collision_data(self) -> None:
        self._collision_objects = {
            "shoulder": (
                Capsule(radius=0.1, start_point=ca.vcat([0, 0, 0]), end_point=ca.vcat([0, 0, 0.14])),
                Link.SHOULDER,
            ),
            "upper_arm": (
                Capsule(radius=0.05, start_point=ca.vcat([0, 0, 0.14]), end_point=ca.vcat([0.452, 0, 0.14])),
                Link.UPPER_ARM,
            ),
            "elbow": (
                Capsule(radius=0.075, start_point=ca.vcat([0, 0, 0]), end_point=ca.vcat([0, 0, 0.14 + 0.025])),
                Link.UPPER_ARM,
            ),
            "lower_arm": (
                Capsule(radius=0.05, start_point=ca.vcat([0, 0, 0.01]), end_point=ca.vcat([0.39, 0, 0.01])),
                Link.FOREARM,
            ),
            "wrist": (
                Capsule(radius=0.065, start_point=ca.vcat([0, 0, -0.01]), end_point=ca.vcat([0, 0, 0.1333])),
                Link.FOREARM,
            ),
            "hand": (
                Capsule(radius=0.055, start_point=ca.vcat([0, 0, -0.025]), end_point=ca.vcat([0, 0, 0.0997 + 0.188])),
                Link.WRIST_2,
            ),
            "cable": (
                Capsule(radius=0.025, start_point=ca.vcat([0, 0.05, 0.018]), end_point=ca.vcat([-0.1, 0.05, 0.018])),
                Link.WRIST_3,
            ),
            "table": (
                HalfSpace(normal_vector=ca.vcat([0.0, 0.0, 1.0]), offset=0.0),
                Link.BASE,
            ),
        }
        self._collision_pairs = [
            # avoid robot link self collisions
            ("shoulder", "wrist"),
            ("shoulder", "hand"),
            ("upper_arm", "wrist"),
            ("upper_arm", "hand"),
            ("lower_arm", "hand"),
            ("shoulder", "cable"),
            ("upper_arm", "cable"),
            ("lower_arm", "cable"),
            # avoid collisions with table surface
            ("table", "elbow"),
            ("table", "wrist"),
            ("table", "hand"),
            ("table", "cable"),
            # TODO: make the checks with table more efficient by using only one sphere per capsule
        ]


@model_registry.register("spheres")
class SphereCollisionModel(CollisionModel):
    def _build_robot_collision_data(self) -> None:
        # define the number of spheres each capsule is replaced by
        sphere_numbers = {
            "shoulder": 2,
            "upper_arm": 7,
            "elbow": 3,
            "lower_arm": 5,
            "wrist": 2,
            "hand": 7,
            "cable": 3,
        }

        # the spheres need to have a higher radius to have a conservative geometry approximation
        radius_factor = 1.2

        # start from capsule definitions, which are identical to those of the capsule model
        capsule_definitions = {
            "shoulder": (
                Capsule(radius=0.1, start_point=ca.vcat([0, 0, 0]), end_point=ca.vcat([0, 0, 0.14])),
                Link.SHOULDER,
            ),
            "upper_arm": (
                Capsule(radius=0.05, start_point=ca.vcat([0, 0, 0.14]), end_point=ca.vcat([0.452, 0, 0.14])),
                Link.UPPER_ARM,
            ),
            "elbow": (
                Capsule(radius=0.075, start_point=ca.vcat([0, 0, 0]), end_point=ca.vcat([0, 0, 0.14 + 0.025])),
                Link.UPPER_ARM,
            ),
            "lower_arm": (
                Capsule(radius=0.05, start_point=ca.vcat([0, 0, 0.01]), end_point=ca.vcat([0.39, 0, 0.01])),
                Link.FOREARM,
            ),
            "wrist": (
                Capsule(radius=0.065, start_point=ca.vcat([0, 0, -0.01]), end_point=ca.vcat([0, 0, 0.1333])),
                Link.FOREARM,
            ),
            "hand": (
                Capsule(radius=0.055, start_point=ca.vcat([0, 0, -0.025]), end_point=ca.vcat([0, 0, 0.0997 + 0.188])),
                Link.WRIST_2,
            ),
            "cable": (
                Capsule(radius=0.025, start_point=ca.vcat([0, 0.05, 0.018]), end_point=ca.vcat([-0.1, 0.05, 0.018])),
                Link.WRIST_3,
            ),
        }

        # convert capsules to groups of spheres
        self._collision_objects = {}

        for name, (capsule, link) in capsule_definitions.items():
            # create spheres along the capsule
            sphere_centers = ca.vertsplit(ca.linspace(capsule.start_point, capsule.end_point, sphere_numbers[name]), 3)

            spheres = [Sphere(radius=capsule.radius * radius_factor, center=center) for center in sphere_centers]

            # special case hand: don't approximate geometry as conservatively, to be able to reach objects
            if name == "hand":
                spheres[-2].radius /= radius_factor
                spheres[-1].radius /= radius_factor

            # create collision group of all spheres from the capsule
            self._collision_objects[name] = (CollisionGroup(spheres), link)

        # add table plane
        self._collision_objects["table"] = (
            HalfSpace(normal_vector=ca.vcat([0.0, 0.0, 1.0]), offset=0.0),
            Link.BASE,
        )

        self._collision_pairs = [
            # avoid robot link self collisions
            ("shoulder", "wrist"),
            ("shoulder", "hand"),
            ("upper_arm", "wrist"),
            ("upper_arm", "hand"),
            ("lower_arm", "hand"),
            ("shoulder", "cable"),
            ("upper_arm", "cable"),
            ("lower_arm", "cable"),
            # avoid collisions with table surface
            ("table", "elbow"),
            ("table", "wrist"),
            ("table", "hand"),
            ("table", "cable"),
            # TODO: make the checks with table more efficient by using only one sphere per capsule
        ]
