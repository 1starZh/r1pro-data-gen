"""Scene model: the runtime, planner-visible view of an environment.

A :class:`SceneModel` holds *environment facts* (robot, objects, cameras,
contact sensors, world physics) loaded from a scene YAML file. It is pure
data (no isaaclab / omni / pxr imports) and is the shared input of:

- the planner (Claude today, an LLM later) -- to *understand* the environment
  and derive task-semantic parameters for a plan;
- the skills -- to run collision checks, compute reachability, and read
  obstacle geometry;
- the Isaac Sim adapter -- to build the scene data-driven from this model.

Task semantics (work pose, grasp offsets, place targets, approach directions)
are NOT part of the scene: the planner derives them from the task description
plus this model and writes them into the plan parameters.

Object shapes use metric units (m), world coordinates for positions, and
quaternions in (w, x, y, z) order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Mapping

try:  # pragma: no cover - exercised in tests
    from r1pro_data_gen.robot.robot_config import R1PRO_JOINT_LIMITS
except Exception:  # pragma: no cover - allow standalone import in tests
    R1PRO_JOINT_LIMITS = {}


class ObjectType(StrEnum):
    """Supported primitive object shapes."""

    CUBOID = "cuboid"
    CYLINDER = "cylinder"


class ObjectCapability(StrEnum):
    """Task-independent interaction capabilities authored in a scene."""

    MOVABLE = "movable"
    GRASPABLE = "graspable"
    PUSHABLE = "pushable"
    SUPPORTS_OBJECTS = "supports_objects"
    CONTAINS_OBJECTS = "contains_objects"
    NAVIGABLE = "navigable"


@dataclass(frozen=True, slots=True)
class RegionModel:
    """A named region expressed in its owning object's local frame."""

    name: str
    shape: ObjectType
    center: tuple[float, float, float]
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("region name must not be empty")
        _require_finite(self.center, f"region {self.name!r} center")
        if self.shape is ObjectType.CUBOID:
            if self.size is None or self.radius is not None or self.height is not None:
                raise ValueError(f"region {self.name!r}: cuboid requires only size")
            _require_positive(self.size, f"region {self.name!r} size")
        elif self.shape is ObjectType.CYLINDER:
            if self.size is not None or self.radius is None or self.height is None:
                raise ValueError(
                    f"region {self.name!r}: cylinder requires only radius and height"
                )
            _require_positive(
                (self.radius, self.height),
                f"region {self.name!r} radius and height",
            )


@dataclass(frozen=True, slots=True)
class SurfaceModel:
    """A named planar support surface in its owning object's local frame."""

    name: str
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    size: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("surface name must not be empty")
        _require_finite(self.center, f"surface {self.name!r} center")
        _require_finite(self.normal, f"surface {self.name!r} normal")
        if sum(value * value for value in self.normal) <= 0.0:
            raise ValueError(f"surface {self.name!r} normal must not be zero")
        _require_positive(self.size, f"surface {self.name!r} size")


@dataclass(frozen=True, slots=True)
class PhysicsProps:
    """Rigid-body / collision properties for an object."""

    kinematic: bool = False
    rigid_object: bool = False
    collision_enabled: bool = True
    planning_margin: float | None = None
    mass: float | None = None
    static_friction: float | None = None
    dynamic_friction: float | None = None
    friction_combine: str | None = None
    contact_offset: float | None = None

    def __post_init__(self) -> None:
        if self.planning_margin is not None:
            _require_non_negative(
                (self.planning_margin,),
                "physics planning_margin",
            )
        if self.mass is not None:
            _require_positive((self.mass,), "physics mass")
        for name, value in (
            ("static_friction", self.static_friction),
            ("dynamic_friction", self.dynamic_friction),
        ):
            if value is not None:
                _require_non_negative((value,), f"physics {name}")
        if self.contact_offset is not None:
            _require_positive((self.contact_offset,), "physics contact_offset")
        if self.friction_combine is not None and not self.friction_combine.strip():
            raise ValueError("physics friction_combine must not be empty")


@dataclass(frozen=True, slots=True)
class VisualProps:
    """Visual appearance for an object."""

    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    roughness: float | None = None

    def __post_init__(self) -> None:
        _require_finite(self.color, "visual color")
        if self.roughness is not None:
            _require_non_negative((self.roughness,), "visual roughness")


@dataclass(frozen=True, slots=True)
class ObjectModel:
    """One spawnable object instance in a scene.

    Exactly one of the shape fields must be set, matching ``type``:
    ``size`` (x, y, z) for cuboids, ``radius``+``height`` for cylinders.
    """

    name: str
    type: ObjectType
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    semantic_class: str | None = None
    aliases: tuple[str, ...] = ()
    capabilities: tuple[ObjectCapability, ...] = ()
    regions: tuple[RegionModel, ...] = ()
    surfaces: tuple[SurfaceModel, ...] = ()
    physics: PhysicsProps = field(default_factory=PhysicsProps)
    visual: VisualProps = field(default_factory=VisualProps)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("object name must not be empty")
        if len(self.pos) != 3:
            raise ValueError(f"object {self.name!r}: pos must be (x, y, z)")
        if len(self.quat) != 4:
            raise ValueError(f"object {self.name!r}: quat must be (w, x, y, z)")
        _require_finite(self.pos, f"object {self.name!r} pos")
        _require_finite(self.quat, f"object {self.name!r} quat")
        if self.semantic_class is not None and not self.semantic_class.strip():
            raise ValueError(f"object {self.name!r}: semantic_class must not be empty")
        if any(not alias.strip() for alias in self.aliases):
            raise ValueError(f"object {self.name!r}: aliases must not be empty")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError(f"object {self.name!r}: aliases must be unique")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError(f"object {self.name!r}: capabilities must be unique")
        region_names = [region.name for region in self.regions]
        if len(region_names) != len(set(region_names)):
            raise ValueError(f"object {self.name!r}: region names must be unique")
        surface_names = [surface.name for surface in self.surfaces]
        if len(surface_names) != len(set(surface_names)):
            raise ValueError(f"object {self.name!r}: surface names must be unique")
        if self.type is ObjectType.CUBOID:
            if self.size is None or len(self.size) != 3:
                raise ValueError(f"object {self.name!r}: cuboid requires size (x, y, z)")
            if self.radius is not None or self.height is not None:
                raise ValueError(f"object {self.name!r}: cuboid must not set radius/height")
            _require_positive(self.size, f"object {self.name!r} size")
        elif self.type is ObjectType.CYLINDER:
            if self.radius is None or self.height is None:
                raise ValueError(f"object {self.name!r}: cylinder requires radius and height")
            if self.size is not None:
                raise ValueError(f"object {self.name!r}: cylinder must not set size")
            _require_positive(
                (self.radius, self.height),
                f"object {self.name!r} radius and height",
            )
        else:  # pragma: no cover - guarded by StrEnum membership
            raise ValueError(f"object {self.name!r}: unsupported type {self.type}")

    @property
    def top_z(self) -> float:
        """World z of the object's top surface (for table/cylinder planning)."""
        if self.type is ObjectType.CYLINDER:
            return self.pos[2] + self.height / 2.0
        return self.pos[2] + self.size[2] / 2.0

    @property
    def vertical_extent_m(self) -> float:
        """Axis-aligned height of the primitive, independent of object type."""
        return object_vertical_extent_m(self)

    @property
    def xy_radius_m(self) -> float:
        """In-plane radius used for placement insets and support tests."""
        return object_xy_radius_m(self)


@dataclass(frozen=True, slots=True)
class CameraModel:
    """One RGB camera attached to the scene."""

    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    width: int = 640
    height: int = 480
    focal_length: float = 24.0
    data_types: tuple[str, ...] = ("rgb",)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("camera name must not be empty")
        if len(self.eye) != 3 or len(self.target) != 3:
            raise ValueError(f"camera {self.name!r}: eye/target must be (x, y, z)")
        _require_finite(self.eye, f"camera {self.name!r} eye")
        _require_finite(self.target, f"camera {self.name!r} target")
        if isinstance(self.width, bool) or self.width <= 0:
            raise ValueError(f"camera {self.name!r}: width must be positive")
        if isinstance(self.height, bool) or self.height <= 0:
            raise ValueError(f"camera {self.name!r}: height must be positive")
        _require_positive((self.focal_length,), f"camera {self.name!r} focal_length")
        if not self.data_types or any(not item.strip() for item in self.data_types):
            raise ValueError(f"camera {self.name!r}: data_types must not be empty")


@dataclass(frozen=True, slots=True)
class ContactSensorModel:
    """A contact-force sensor on a robot body, optionally filtered to objects."""

    name: str
    body: str
    filter: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("sensor name must not be empty")
        if not self.body.strip():
            raise ValueError(f"sensor {self.name!r}: body must not be empty")
        if any(not item.strip() for item in self.filter):
            raise ValueError(f"sensor {self.name!r}: filter names must not be empty")
        if len(self.filter) != len(set(self.filter)):
            raise ValueError(f"sensor {self.name!r}: filter names must be unique")


@dataclass(frozen=True, slots=True)
class RobotModel:
    """Robot instance facts: asset, initial pose, home configuration.

    ``navigation_footprint_radius_m`` is an execution-calibrated planning
    parameter. It is kept with the scene facts so an external planner uses the
    same hard footprint inflation as the runtime navigation skill.
    """

    asset: str
    init_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    home_joint_pos: Mapping[str, float] = field(default_factory=dict)
    navigation_footprint_radius_m: float | None = None

    def __post_init__(self) -> None:
        if not self.asset.strip():
            raise ValueError("robot asset must not be empty")
        if len(self.init_pose) != 3:
            raise ValueError("robot init_pose must be (x, y, yaw)")
        _require_finite(self.init_pose, "robot init_pose")
        if self.navigation_footprint_radius_m is not None:
            _require_positive(
                (self.navigation_footprint_radius_m,),
                "robot navigation footprint radius",
            )
        for name, value in self.home_joint_pos.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("robot home joint names must not be empty")
            _require_finite((value,), f"robot home joint {name!r}")
            if R1PRO_JOINT_LIMITS and name not in R1PRO_JOINT_LIMITS:
                raise ValueError(f"robot home joint {name!r} is not a known R1Pro joint")


@dataclass(frozen=True, slots=True)
class WorldModel:
    """World physics and ground settings."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    dt: float = 1.0 / 60.0
    ground: bool = True
    ground_size: tuple[float, float] = (20.0, 20.0)
    ground_color: tuple[float, float, float] = (0.25, 0.26, 0.28)

    def __post_init__(self) -> None:
        _require_finite(self.gravity, "world gravity")
        _require_positive((self.dt,), "world dt")
        _require_positive(self.ground_size, "world ground_size")
        _require_finite(self.ground_color, "world ground_color")


@dataclass(frozen=True, slots=True)
class SceneModel:
    """A complete environment: world, robot, objects, cameras, sensors."""

    name: str
    world: WorldModel
    robot: RobotModel
    objects: tuple[ObjectModel, ...] = ()
    cameras: tuple[CameraModel, ...] = ()
    contact_sensors: tuple[ContactSensorModel, ...] = ()
    collision_sensors: tuple[ContactSensorModel, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scene name must not be empty")
        names = [o.name for o in self.objects]
        if len(names) != len(set(names)):
            raise ValueError("scene object names must be unique")
        cam_names = [c.name for c in self.cameras]
        if len(cam_names) != len(set(cam_names)):
            raise ValueError("scene camera names must be unique")
        sensor_names = [sensor.name for sensor in self.contact_sensors]
        if len(sensor_names) != len(set(sensor_names)):
            raise ValueError("scene contact sensor names must be unique")
        collision_sensor_names = [sensor.name for sensor in self.collision_sensors]
        if len(collision_sensor_names) != len(set(collision_sensor_names)):
            raise ValueError("scene collision sensor names must be unique")
        if set(sensor_names) & set(collision_sensor_names):
            raise ValueError("scene sensor names must be unique across contact and collision sensors")
        known_objects = set(names)
        for sensor in (*self.contact_sensors, *self.collision_sensors):
            if sensor in self.collision_sensors and not sensor.filter:
                raise ValueError(f"collision sensor {sensor.name!r} requires a non-empty filter")
            missing = set(sensor.filter) - known_objects
            if missing:
                raise ValueError(
                    f"sensor {sensor.name!r} filters unknown objects: {sorted(missing)}"
                )

    def object(self, name: str) -> ObjectModel:
        """Look up an object by name."""
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"scene {self.name!r} has no object {name!r}")

    def contact_sensor(self, name: str) -> ContactSensorModel:
        """Look up a contact sensor by name."""
        for sensor in self.contact_sensors:
            if sensor.name == name:
                return sensor
        raise KeyError(f"scene {self.name!r} has no sensor {name!r}")

    # ------------------------------------------------------------------
    # Deserialization helpers (pure dict <-> model, YAML handled by data.scenes).
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneModel":
        """Build a SceneModel from a dict (the parsed scene YAML)."""
        if not isinstance(data, Mapping):
            raise TypeError("scene YAML must be a mapping")
        _reject_unknown_fields(
            data,
            {
                "name",
                "world",
                "robot",
                "objects",
                "cameras",
                "contact_sensors",
                "collision_sensors",
            },
            "scene",
        )
        if "name" not in data:
            raise ValueError("scene YAML requires a name")
        world = data.get("world", {}) or {}
        robot = data.get("robot", {}) or {}
        if not isinstance(world, Mapping):
            raise TypeError("scene world must be a mapping")
        if not isinstance(robot, Mapping):
            raise TypeError("scene robot must be a mapping")
        _reject_unknown_fields(
            world,
            {"gravity", "dt", "ground", "ground_size", "ground_color"},
            "world",
        )
        _reject_unknown_fields(
            robot,
            {
                "asset",
                "init_pose",
                "home_joint_pos",
                "navigation_footprint_radius_m",
            },
            "robot",
        )
        if "asset" not in robot:
            raise ValueError("scene YAML requires robot.asset")

        raw_objects = data.get("objects", []) or []
        if not isinstance(raw_objects, list):
            raise TypeError("scene objects must be an array")
        objects = tuple(
            _object_from_dict(obj, index)
            for index, obj in enumerate(raw_objects)
        )
        raw_cameras = data.get("cameras", []) or []
        if not isinstance(raw_cameras, list):
            raise TypeError("scene cameras must be an array")
        cameras = tuple(
            _camera_from_dict(cam, index)
            for index, cam in enumerate(raw_cameras)
        )
        raw_sensors = data.get("contact_sensors", []) or []
        if not isinstance(raw_sensors, list):
            raise TypeError("scene contact_sensors must be an array")
        sensors = tuple(
            _sensor_from_dict(sensor, index)
            for index, sensor in enumerate(raw_sensors)
        )
        raw_collision_sensors = data.get("collision_sensors", []) or []
        if not isinstance(raw_collision_sensors, list):
            raise TypeError("scene collision_sensors must be an array")
        collision_sensors = tuple(
            _collision_sensor_from_dict(sensor, index)
            for index, sensor in enumerate(raw_collision_sensors)
        )
        raw_home_joint_pos = robot.get("home_joint_pos", {}) or {}
        if not isinstance(raw_home_joint_pos, Mapping):
            raise TypeError("robot home_joint_pos must be a mapping")
        return cls(
            name=_require_string(data["name"], "scene name"),
            world=WorldModel(
                gravity=_triple(world.get("gravity", (0.0, 0.0, -9.81)), "gravity"),
                dt=_number(world.get("dt", 1.0 / 60.0), "world dt"),
                ground=_boolean(world.get("ground", True), "world ground"),
                ground_size=_pair(world.get("ground_size", (20.0, 20.0)), "ground_size"),
                ground_color=_triple(world.get("ground_color", (0.25, 0.26, 0.28)), "ground_color"),
            ),
            robot=RobotModel(
                asset=_require_string(robot["asset"], "robot asset"),
                init_pose=_triple(robot.get("init_pose", (0.0, 0.0, 0.0)), "init_pose"),
                home_joint_pos={
                    _require_string(name, "robot home joint name"): _number(
                        value,
                        f"robot home joint {name!r}",
                    )
                    for name, value in raw_home_joint_pos.items()
                },
                navigation_footprint_radius_m=(
                    _number(
                        robot["navigation_footprint_radius_m"],
                        "robot navigation footprint radius",
                    )
                    if robot.get("navigation_footprint_radius_m") is not None
                    else None
                ),
            ),
            objects=objects,
            cameras=cameras,
            contact_sensors=sensors,
            collision_sensors=collision_sensors,
        )


def _reject_unknown_fields(
    data: Mapping[str, Any],
    allowed: set[str],
    what: str,
) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {what} fields: {sorted(unknown)}")


def _object_from_dict(raw: Any, index: int) -> ObjectModel:
    what = f"object[{index}]"
    if not isinstance(raw, Mapping):
        raise TypeError(f"scene {what} must be a mapping")
    _reject_unknown_fields(
        raw,
        {
            "name",
            "type",
            "pos",
            "quat",
            "size",
            "radius",
            "height",
            "semantic_class",
            "aliases",
            "capabilities",
            "regions",
            "surfaces",
            "kinematic",
            "rigid_object",
            "collision_enabled",
            "planning_margin",
            "mass",
            "static_friction",
            "dynamic_friction",
            "friction_combine",
            "contact_offset",
            "color",
            "roughness",
        },
        "object",
    )
    for required in ("name", "type"):
        if required not in raw:
            raise ValueError(f"scene {what} requires {required}")

    raw_aliases = _string_array(raw.get("aliases", []), f"{what} aliases")
    raw_capabilities = raw.get("capabilities", []) or []
    if not isinstance(raw_capabilities, list):
        raise TypeError(f"{what} capabilities must be an array")
    capabilities: list[ObjectCapability] = []
    for capability_index, value in enumerate(raw_capabilities):
        text = _require_string(
            value,
            f"{what} capabilities[{capability_index}]",
        )
        try:
            capabilities.append(ObjectCapability(text))
        except ValueError as exc:
            raise ValueError(f"unknown capability: {text!r}") from exc

    raw_regions = raw.get("regions", []) or []
    if not isinstance(raw_regions, list):
        raise TypeError(f"{what} regions must be an array")
    raw_surfaces = raw.get("surfaces", []) or []
    if not isinstance(raw_surfaces, list):
        raise TypeError(f"{what} surfaces must be an array")

    object_type_text = _require_string(raw["type"], f"{what} type")
    try:
        object_type = ObjectType(object_type_text)
    except ValueError as exc:
        raise ValueError(f"{what} has unsupported type {object_type_text!r}") from exc

    semantic_class = raw.get("semantic_class")
    if semantic_class is not None:
        semantic_class = _require_string(semantic_class, f"{what} semantic_class")
    return ObjectModel(
        name=_require_string(raw["name"], f"{what} name"),
        type=object_type,
        pos=_triple(raw.get("pos", (0.0, 0.0, 0.0)), f"{what} pos"),
        quat=_quad(raw.get("quat", (1.0, 0.0, 0.0, 0.0)), f"{what} quat"),
        size=_triple(raw["size"], f"{what} size") if raw.get("size") is not None else None,
        radius=_optional_positive_number(raw, "radius", what),
        height=_optional_positive_number(raw, "height", what),
        semantic_class=semantic_class,
        aliases=raw_aliases,
        capabilities=tuple(capabilities),
        regions=tuple(
            _region_from_dict(region, index, region_index)
            for region_index, region in enumerate(raw_regions)
        ),
        surfaces=tuple(
            _surface_from_dict(surface, index, surface_index)
            for surface_index, surface in enumerate(raw_surfaces)
        ),
        physics=PhysicsProps(
            kinematic=_boolean(raw.get("kinematic", False), f"{what} kinematic"),
            rigid_object=_boolean(
                raw.get("rigid_object", False),
                f"{what} rigid_object",
            ),
            collision_enabled=_boolean(
                raw.get("collision_enabled", True),
                f"{what} collision_enabled",
            ),
            planning_margin=_optional_number(raw, "planning_margin", what),
            mass=_optional_number(raw, "mass", what),
            static_friction=_optional_number(raw, "static_friction", what),
            dynamic_friction=_optional_number(raw, "dynamic_friction", what),
            friction_combine=(
                _require_string(raw["friction_combine"], f"{what} friction_combine")
                if raw.get("friction_combine") is not None
                else None
            ),
            contact_offset=_optional_number(raw, "contact_offset", what),
        ),
        visual=VisualProps(
            color=_triple(raw.get("color", (0.5, 0.5, 0.5)), f"{what} color"),
            roughness=_optional_number(raw, "roughness", what),
        ),
    )


def _region_from_dict(raw: Any, object_index: int, region_index: int) -> RegionModel:
    what = f"object[{object_index}] region[{region_index}]"
    if not isinstance(raw, Mapping):
        raise TypeError(f"{what} must be a mapping")
    _reject_unknown_fields(
        raw,
        {"name", "shape", "center", "size", "radius", "height"},
        "region",
    )
    for required in ("name", "shape", "center"):
        if required not in raw:
            raise ValueError(f"{what} requires {required}")
    shape_text = _require_string(raw["shape"], f"{what} shape")
    try:
        shape = ObjectType(shape_text)
    except ValueError as exc:
        raise ValueError(f"{what} has unsupported shape {shape_text!r}") from exc
    return RegionModel(
        name=_require_string(raw["name"], f"{what} name"),
        shape=shape,
        center=_triple(raw["center"], f"{what} center"),
        size=_triple(raw["size"], f"{what} size") if raw.get("size") is not None else None,
        radius=_optional_positive_number(raw, "radius", what),
        height=_optional_positive_number(raw, "height", what),
    )


def _surface_from_dict(raw: Any, object_index: int, surface_index: int) -> SurfaceModel:
    what = f"object[{object_index}] surface[{surface_index}]"
    if not isinstance(raw, Mapping):
        raise TypeError(f"{what} must be a mapping")
    _reject_unknown_fields(raw, {"name", "center", "normal", "size"}, "surface")
    for required in ("name", "center", "normal", "size"):
        if required not in raw:
            raise ValueError(f"{what} requires {required}")
    return SurfaceModel(
        name=_require_string(raw["name"], f"{what} name"),
        center=_triple(raw["center"], f"{what} center"),
        normal=_triple(raw["normal"], f"{what} normal"),
        size=_pair(raw["size"], f"{what} size"),
    )


def _camera_from_dict(raw: Any, index: int) -> CameraModel:
    what = f"camera[{index}]"
    if not isinstance(raw, Mapping):
        raise TypeError(f"scene {what} must be a mapping")
    _reject_unknown_fields(
        raw,
        {"name", "eye", "target", "width", "height", "focal_length", "data_types"},
        "camera",
    )
    if "name" not in raw:
        raise ValueError(f"scene {what} requires name")
    return CameraModel(
        name=_require_string(raw["name"], f"{what} name"),
        eye=_triple(raw.get("eye", (3.5, -3.5, 2.5)), f"{what} eye"),
        target=_triple(raw.get("target", (0.0, 0.0, 0.8)), f"{what} target"),
        width=_positive_integer(raw.get("width", 640), f"{what} width"),
        height=_positive_integer(raw.get("height", 480), f"{what} height"),
        focal_length=_number(raw.get("focal_length", 24.0), f"{what} focal_length"),
        data_types=_string_array(raw.get("data_types", ["rgb"]), f"{what} data_types"),
    )


def _sensor_from_dict(raw: Any, index: int) -> ContactSensorModel:
    what = f"contact_sensor[{index}]"
    if not isinstance(raw, Mapping):
        raise TypeError(f"scene {what} must be a mapping")
    _reject_unknown_fields(raw, {"name", "body", "filter"}, "contact sensor")
    for required in ("name", "body"):
        if required not in raw:
            raise ValueError(f"scene {what} requires {required}")
    return ContactSensorModel(
        name=_require_string(raw["name"], f"{what} name"),
        body=_require_string(raw["body"], f"{what} body"),
        filter=_string_array(raw.get("filter", []), f"{what} filter"),
    )


def _collision_sensor_from_dict(raw: Any, index: int) -> ContactSensorModel:
    what = f"collision_sensor[{index}]"
    if not isinstance(raw, Mapping):
        raise TypeError(f"scene {what} must be a mapping")
    _reject_unknown_fields(raw, {"name", "body", "filter"}, "collision sensor")
    for required in ("name", "body"):
        if required not in raw:
            raise ValueError(f"scene {what} requires {required}")
    return ContactSensorModel(
        name=_require_string(raw["name"], f"{what} name"),
        body=_require_string(raw["body"], f"{what} body"),
        filter=_string_array(raw.get("filter", []), f"{what} filter"),
    )


def _optional_number(data: Mapping[str, Any], key: str, what: str) -> float | None:
    value = data.get(key)
    return _number(value, f"{what} {key}") if value is not None else None


def _optional_positive_number(
    data: Mapping[str, Any],
    key: str,
    what: str,
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    _require_positive((value,), f"{what} {key}")
    return float(value)


def _require_string(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _string_array(value: Any, what: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{what} must be an array")
    return tuple(
        _require_string(item, f"{what}[{index}]")
        for index, item in enumerate(value)
    )


def _boolean(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{what} must be a boolean")
    return value


def _positive_integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{what} must be a positive integer")
    return value


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be finite")
    return result


def _require_finite(values: tuple[float, ...], what: str) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"{what} must contain only finite numbers")


def _require_positive(values: tuple[float, ...], what: str) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in values
    ):
        raise ValueError(f"{what} must be finite and positive")


def _require_non_negative(values: tuple[float, ...], what: str) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError(f"{what} must be finite and non-negative")


def _numeric_sequence(value: Any, length: int, what: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{what} must be a numeric array")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{what} must be a numeric array") from exc
    if len(values) != length:
        labels = {2: "(a, b)", 3: "(a, b, c)", 4: "(a, b, c, d)"}
        raise ValueError(f"{what} must be {labels[length]}")
    return tuple(_number(item, f"{what}[{index}]") for index, item in enumerate(values))


def _pair(value: Any, what: str) -> tuple[float, float]:
    first, second = _numeric_sequence(value, 2, what)
    return first, second


def _triple(value: Any, what: str) -> tuple[float, float, float]:
    first, second, third = _numeric_sequence(value, 3, what)
    return first, second, third


def _quad(value: Any, what: str) -> tuple[float, float, float, float]:
    first, second, third, fourth = _numeric_sequence(value, 4, what)
    return first, second, third, fourth


def object_vertical_extent_m(model: Any) -> float:
    """Return the primitive's vertical size for cylinder or cuboid models.

    Carry, support inference and placement use this instead of reading
    ``height`` (cylinders only) so a cuboid scene object does not crash.
    """
    height = getattr(model, "height", None)
    if height is not None:
        return float(height)
    size = getattr(model, "size", None)
    if size is not None and len(size) >= 3 and size[2] is not None:
        return float(size[2])
    raise ValueError("object has no vertical extent (height or size.z)")


def object_xy_radius_m(model: Any) -> float:
    """In-plane radius: cylinder radius, or half the smaller cuboid XY side."""
    radius = getattr(model, "radius", None)
    if radius is not None:
        return float(radius)
    size = getattr(model, "size", None)
    if size is not None and len(size) >= 2:
        return 0.5 * min(float(size[0]), float(size[1]))
    return 0.0


def object_xy_half_extents_m(model: Any) -> tuple[float, float]:
    """Return a conservative world-XY half extent for a scene primitive.

    Navigation and collision helpers must agree even when a scene author
    rotates a cuboid. The returned rectangle is the projection of the
    primitive's oriented bounding box onto world XY; it is conservative and
    therefore safe for footprint inflation. Cylinders use their radial extent
    because the current scene schema models them as vertical primitives.
    """
    radius = getattr(model, "radius", None)
    if radius is not None:
        value = abs(float(radius))
        return value, value
    size = getattr(model, "size", None)
    if size is None or len(size) < 3:
        raise ValueError("object has no supported XY geometry")
    half_x, half_y, half_z = (0.5 * float(value) for value in size[:3])
    quaternion = tuple(float(value) for value in getattr(model, "quat", (1.0, 0.0, 0.0, 0.0)))
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w, x, y, z = (value / norm for value in quaternion)
    # First two rows of the quaternion rotation matrix. Projection of an OBB
    # onto one world axis is the sum of absolute rotated half axes.
    row_x = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
    )
    row_y = (
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
    )
    half_axes = (half_x, half_y, half_z)
    return (
        sum(abs(component) * extent for component, extent in zip(row_x, half_axes)),
        sum(abs(component) * extent for component, extent in zip(row_y, half_axes)),
    )


def object_surface_distance_m(
    object_center: Any,
    query_point: Any,
    model: Any,
) -> float:
    """Unsigned distance from a query point to the object's bounding surface.

    Cylinders and cuboids share the same conservative capsule-of-AABB test so
    finger-window alignment does not require a cylinder ``radius`` field.
    """
    center = tuple(float(v) for v in object_center[:3])
    point = tuple(float(v) for v in query_point[:3])
    radial = math.hypot(point[0] - center[0], point[1] - center[1])
    vertical = abs(point[2] - center[2])
    radial_outside = max(0.0, radial - object_xy_radius_m(model))
    vertical_outside = max(0.0, vertical - object_vertical_extent_m(model) * 0.5)
    return math.hypot(radial_outside, vertical_outside)


__all__ = [
    "CameraModel",
    "ContactSensorModel",
    "ObjectCapability",
    "ObjectModel",
    "ObjectType",
    "PhysicsProps",
    "RegionModel",
    "RobotModel",
    "SceneModel",
    "SurfaceModel",
    "VisualProps",
    "WorldModel",
    "object_xy_half_extents_m",
    "object_vertical_extent_m",
    "object_xy_radius_m",
    "object_surface_distance_m",
]
