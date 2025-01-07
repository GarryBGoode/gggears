import dataclasses
import copy
import gggears.curve as crv
import numpy as np
from gggears.defs import *
from scipy.spatial.transform import Rotation as scp_Rotation

# If a dataclass tends to be user input, it should be named param.
# If a dataclass tends to be generated or manipulated by functions,
# it should be named data.


def make_callables(indict):
    for key, value in indict.items():
        if isinstance(value, dict):
            make_callables(value)
        elif not callable(value):
            indict[key] = lambda z, v=value: v


def eval_callables(indict, z):
    for key, value in indict.items():
        if callable(value):
            indict[key] = value(z)
    return indict


class ZFunctionMixin:
    """
    Mixin class to seemlessly handle callable parameters in dataclasses.
    3D gear features are sometimes defined by one or more of their parameters
    being a function z.
    """

    def __call__(self, z):
        # copy the dict to avoid changing the original
        dict_vals = copy.deepcopy(self.__dict__)
        # replace all callable values with their evaluated value
        dict_vals = eval_callables(dict_vals, z)
        # This is interesting in the inheritance context.
        # When a class inherits from a parameter dataclass and this mixin,
        # the returned class should be backward compatible with the original dataclass.
        return self.__class__(**dict_vals)


@dataclasses.dataclass
class TransformData:
    """Data class for general 3D transformation (move, rotate, scale).

    Attributes
    ----------
    center : np.ndarray
        Center displacement the transformation.
    orientation : np.ndarray
        Orientation matrix of the transformation.
    scale : float
        Scale factor of the transformation.
    """

    center: np.ndarray = dataclasses.field(default_factory=lambda: ORIGIN)
    orientation: np.ndarray = dataclasses.field(default_factory=lambda: UNIT3X3)
    scale: float = 1.0

    @property
    def x_axis(self):
        return self.orientation[:, 0]

    @property
    def y_axis(self):
        return self.orientation[:, 1]

    @property
    def z_axis(self):
        return self.orientation[:, 2]

    @property
    def affine_matrix(self):
        return np.block(
            [[self.orientation * self.scale, self.center[:, np.newaxis]], [0, 0, 0, 1]]
        )

    def __mul__(self, other):
        if isinstance(other, TransformData):
            return TransformData(
                center=self.center + self.orientation @ other.center * self.scale,
                orientation=self.orientation @ other.orientation,
                scale=self.scale * other.scale,
            )
        else:
            return NotImplemented


def apply_transform(points: np.ndarray, data: TransformData) -> np.ndarray:
    """
    Apply a general 3D transformation to a set of points.

    Parameters
    ----------
    points : np.ndarray
        An array of points to be transformed. Each point should be a 3D coordinate.
    data : TransformData
        An object containing the transformation data, including orientation, scale,
        and center shift.

    Returns
    -------
    np.ndarray
        The transformed points as an array of the same shape as the input.
    """
    return points @ data.orientation.transpose() * data.scale + data.center


class Transform(TransformData):
    """
    A callable class for applying a general 3D transformation to a set of points.
    """

    def __call__(self, points) -> np.ndarray:
        return apply_transform(points, self)


@dataclasses.dataclass
class GearTransformData(TransformData):
    """
    Data class for gear base transformation.
    Besides the general base transform, the gear's angle is included.
    This helps track the gear's rotation-advance, phase angle, etc.
    separately from its orientation.

    Attributes
    ----------
    center : np.ndarray
        Center displacement the transformation.
    orientation : np.ndarray
        Orientation matrix of the transformation.
    scale : float
        Scale factor of the transformation.
    angle : float
        The angle of the gear in radians.
    """

    angle: float = 0

    @property
    def affine_matrix(self):
        # override to include angle as well
        orient2 = (
            self.orientation @ scp_Rotation.from_euler("z", self.angle).as_matrix()
        )
        return np.block(
            [[orient2 * self.scale, self.center[:, np.newaxis]], [0, 0, 0, 1]]
        )

    def __mul__(self, other):
        if isinstance(other, GearTransformData):
            return GearTransformData(
                center=self.center + self.orientation @ other.center * self.scale,
                orientation=self.orientation @ other.orientation,
                scale=self.scale * other.scale,
                angle=self.angle + other.angle,
            )
        else:
            return NotImplemented


def apply_gear_transform(points: np.ndarray, data: GearTransformData) -> np.ndarray:
    """Apply GearTransform to a set of points."""
    rot_z = scp_Rotation.from_euler("z", data.angle).as_matrix()
    return (
        points @ rot_z.transpose() @ data.orientation.transpose() * data.scale
        + data.center
    )


class GearTransform(GearTransformData):
    """A callable class for applying a gear transformation to a set of points.
    Inherited from GearTransformData."""

    def __call__(self, points) -> np.ndarray:
        return apply_gear_transform(points, self)

    def __mul__(self, other):
        if isinstance(other, GearTransformData):
            return GearTransform(
                center=self.center + self.orientation @ other.center * self.scale,
                orientation=self.orientation @ other.orientation,
                scale=self.scale * other.scale,
                angle=self.angle + other.angle,
            )
        else:
            return NotImplemented


@dataclasses.dataclass
class GearToothParam:
    """
    Data class for gear teeth.
    By convention, negative teeth number results inverting the gear
    (i.e. inside teeth).
    Non-integer teeth number results in the actual number rounded down,
    but the size of the gear and teeth matching the rational input.

    Notes
    -----
    It makes no sense for a gear to have a non-integer number of teeth,
    but it can make sense to design a single tooth or a partial gear with
    a size corresponding to a non-integer number of teeth.

    There is no lower limit for the number of teeth,
    but a value around 0...3 might break things.

    Attributes
    ----------
    num_teeth : float
        Number of teeth. Negative will set inside teeth.
        Non-integer will be rounded down, but there will be a gap.
    num_cutout_teeth : int
        Number of teeth not realized in the gear.
    inside_teeth : bool
        Used for creating inside-ring gears.
    """

    num_teeth: float = 16
    num_cutout_teeth: int = 0
    inside_teeth: bool = False

    def __post_init__(self):
        if self.num_teeth < 0:
            self.num_teeth *= -1
            self.inside_teeth = not self.inside_teeth

    @property
    def num_teeth_act(self):
        """Actual (integer) number of teeth, considering rounding and cutout."""
        return int(np.floor(self.num_teeth - self.num_cutout_teeth))

    @property
    def pitch_angle(self):
        """Pitch angle in radians"""
        return 2 * PI / self.num_teeth


@dataclasses.dataclass
class ToothLimitParam:
    """Dataclass for radial limiting coefficients (addendum, dedendum, etc.).

    Attributes
    ----------
    h_a : float
        Addendum height coefficient.
    h_d : float
        Dedendum height coefficient.
    h_o : float
        Outside ring height coefficient.
    """

    h_a: float = 1
    h_d: float = 1.2
    h_o: float = 2


@dataclasses.dataclass
class GearRefCircles:
    """Data class for gear reference circles as Curve objects.

    Attributes
    ----------
    r_a_curve : crv.ArcCurve
        Addendum circle.
    r_p_curve : crv.ArcCurve
        Pitch circle.
    r_d_curve : crv.ArcCurve
        Dedendum circle.
    r_o_curve : crv.ArcCurve
        Outside (or inside) ring circle.
    """

    r_a_curve: crv.ArcCurve  # addendum circle
    r_p_curve: crv.ArcCurve  # pitch circle
    r_d_curve: crv.ArcCurve  # dedendum circle
    r_o_curve: crv.ArcCurve  # outside (or inside) ring circle

    @property
    def r_a(self):
        """Radius of the addendum circle."""
        return self.r_a_curve.r

    @property
    def r_p(self):
        """Radius of the pitch circle."""
        return self.r_p_curve.r

    @property
    def r_d(self):
        """Radius of the dedendum circle."""
        return self.r_d_curve.r

    @property
    def r_o(self):
        """Radius of the outside (or inside) ring circle."""
        return self.r_o_curve.r

    @property
    def center(self):
        """Center of the pitch circle."""
        return self.r_p_curve.center

    @center.setter
    def center(self, value):
        self.r_p_curve.center = value
        self.r_a_curve.center = value
        self.r_d_curve.center = value
        self.r_o_curve.center = value


@dataclasses.dataclass
class ConicData:
    """Dataclass for cone parameters."""

    cone_angle: float = 0
    base_radius: float = 1

    @property
    def gamma(self):
        return self.cone_angle / 2

    @property
    def height(self):
        return self.base_radius / np.tan(self.gamma)

    @property
    def center(self):
        """Spherical center (tip) of the cone."""
        return OUT * self.height

    @property
    def spherical_radius(self):
        return self.base_radius / np.sin(self.gamma)

    # shorthands
    @property
    def R(self):
        return self.spherical_radius

    @property
    def r(self):
        return self.base_radius


class GearToothGenerator(ZFunctionMixin):
    def __init__(
        self,
        pitch_intersect_angle: float = PI / 16,
        pitch_radius: float = 1.0,
        tooth_angle: float = 0,
    ):
        self.pitch_intersect_angle = pitch_intersect_angle
        self.pitch_radius = pitch_radius
        self.tooth_angle = tooth_angle

    def generate_tooth_curve(self) -> crv.Curve:
        p0 = scp_Rotation.from_euler("z", -self.pitch_intersect_angle).apply(
            (RIGHT * self.pitch_radius)
        )
        rot_ta = scp_Rotation.from_euler("z", self.tooth_angle)
        dp = rot_ta.apply(p0 * 0.2)
        return crv.LineCurve(p0=p0 - dp, p1=p0 + dp)


class GearToothConicGenerator(GearToothGenerator):
    def __init__(
        self,
        pitch_intersect_angle: float = PI / 16,
        pitch_radius: float = 1.0,
        cone_angle: float = PI / 4,
        tooth_angle: float = 0,
    ):
        self.pitch_intersect_angle = pitch_intersect_angle
        self.pitch_radius = pitch_radius
        self.cone_angle = cone_angle
        self.tooth_angle = tooth_angle

    @property
    def conic_data(self):
        return ConicData(cone_angle=self.cone_angle, base_radius=self.pitch_radius)

    def generate_tooth_curve(self) -> crv.Curve:

        if self.cone_angle == 0:
            return super().generate_tooth_curve()
        else:

            cone = ConicData(cone_angle=self.cone_angle, base_radius=self.pitch_radius)
            R = cone.R
            h = cone.height
            gamma = cone.gamma

            p0 = scp_Rotation.from_euler("z", -self.pitch_intersect_angle).apply(
                (RIGHT * self.pitch_radius)
            )
            axis = np.cross(p0 / np.linalg.norm(p0), OUT)
            rot_ta = scp_Rotation.from_rotvec(
                -p0 / np.linalg.norm(p0) * self.tooth_angle
            )
            axis = rot_ta.apply(axis)

            return crv.ArcCurve.from_point_center_angle(
                p0=p0, center=OUT * h, angle=0.1, axis=axis
            )
