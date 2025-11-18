import dataclasses
import copy
import gggears.curve as crv
import numpy as np
from gggears.defs import *
from scipy.spatial.transform import Rotation as scp_Rotation
from gggears.function_generators import angle_between_vector_and_plane

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
        elif isinstance(other, np.ndarray):
            return apply_transform(other, self)
        else:
            return NotImplemented

    def invert(self):
        """Invert the transformation."""
        inv_orient = self.orientation.transpose()
        return TransformData(
            center=-inv_orient @ self.center * self.scale,
            orientation=inv_orient,
            scale=1 / self.scale,
        )


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

    def __mul__(self, other):
        if isinstance(other, TransformData):
            return Transform(
                center=self.center + self.orientation @ other.center * self.scale,
                orientation=self.orientation @ other.orientation,
                scale=self.scale * other.scale,
            )
        elif isinstance(other, np.ndarray):
            return apply_transform(other, self)
        else:
            return NotImplemented

    def invert(self):
        """Invert the transformation."""
        inv_orient = self.orientation.transpose()
        return Transform(
            center=-inv_orient @ self.center * self.scale,
            orientation=inv_orient,
            scale=1 / self.scale,
        )


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
        elif isinstance(other, np.ndarray):
            return apply_gear_transform(other, self)
        else:
            return NotImplemented

    def invert(self):
        """Invert the transformation."""
        inv_orient = self.orientation.transpose()
        inv_angle = scp_Rotation.from_euler("z", -self.angle).as_matrix()
        return GearTransformData(
            center=-inv_orient @ inv_angle @ self.center * self.scale,
            orientation=inv_orient,
            scale=1 / self.scale,
            angle=-self.angle,
        )


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
                center=self.center
                + self.orientation
                @ scp_Rotation.from_euler("z", self.angle).as_matrix()
                @ other.center
                * self.scale,
                orientation=self.orientation @ other.orientation,
                scale=self.scale * other.scale,
                angle=self.angle + other.angle,
            )
        elif isinstance(other, np.ndarray):
            return apply_gear_transform(other, self)
        else:
            return NotImplemented

    def invert(self):
        """Invert the transformation."""
        inv_orient = self.orientation
        inv_angle = scp_Rotation.from_euler("z", self.angle).as_matrix()
        return GearTransform(
            center=-self.center @ inv_orient @ inv_angle / self.scale,
            orientation=inv_orient.transpose(),
            scale=1 / self.scale,
            angle=-self.angle,
        )


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
class RackToothParam:
    num_teeth: int = 16

    @property
    def pitch(self):
        # Pitch is the distance between tooth flanks.
        # By convention, module is considered 1 on this level.
        return PI

    @property
    def num_teeth_act(self):
        """Not implemented for racks, but added here for compatibility"""
        return self.num_teeth

    @property
    def inside_teeth(self):
        """Not implemented for racks, but added here for compatibility"""
        return False


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
class RackRefLines:
    """Data class for rack reference lines as Curve objects.

    Attributes
    ----------
    a_line : crv.LineCurve
        Addendum line.
    p_line : crv.LineCurve
        Pitch line.
    d_line : crv.LineCurve
        Dedendum line.
    o_line : crv.LineCurve
        Other side line for closing the rack.
    """

    a_line: crv.LineCurve
    p_line: crv.LineCurve
    d_line: crv.LineCurve
    o_line: crv.LineCurve

    @property
    def center(self):
        """Center of the pitch line."""
        return (self.p_line.p0 + self.p_line.p1) / 2

    @center.setter
    def center(self, value):
        diff = value - self.center
        self.a_line.p0 += diff
        self.a_line.p1 += diff
        self.p_line.p0 += diff
        self.p_line.p1 += diff
        self.d_line.p0 += diff
        self.d_line.p1 += diff
        self.o_line.p0 += diff
        self.o_line.p1 += diff


@dataclasses.dataclass
class ConicData:
    """Dataclass for cone parameters."""

    cone_angle: float = 0
    base_radius: float = 1
    transform: TransformData = dataclasses.field(
        default_factory=lambda: TransformData()
    )

    @property
    def gamma(self):
        return self.cone_angle / 2

    @property
    def height(self):
        return self.base_radius / np.tan(self.gamma) * self.transform.scale

    @property
    def center(self):
        """Spherical center (tip) of the cone."""
        return apply_transform(
            OUT * self.base_radius / np.tan(self.gamma), self.transform
        )

    @property
    def center_base(self):
        """Center of the base circle of the cone."""
        return apply_transform(ORIGIN, self.transform)

    @property
    def spherical_radius(self):
        """Radius of the sphere that is concentric with the cone and contains the base
        circle. Always positive."""
        return np.abs(self.base_radius / np.sin(self.gamma) * self.transform.scale)

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
        ref_limits: ToothLimitParam = None,
    ):
        self.pitch_intersect_angle = pitch_intersect_angle
        self.pitch_radius = pitch_radius
        self.tooth_angle = tooth_angle
        self.ref_limits = ref_limits if ref_limits is not None else ToothLimitParam()

    def generate_tooth_curve(self) -> crv.Curve:
        p0 = scp_Rotation.from_euler("z", -self.pitch_intersect_angle).apply(
            (RIGHT * self.pitch_radius)
        )
        rot_ta = scp_Rotation.from_euler("z", self.tooth_angle)
        dp = rot_ta.apply(p0 * 0.2)
        return crv.CurveChain(crv.LineCurve(p0=p0 - dp, p1=p0 + dp))

    def check_lower_curve_limit(self, tooth_curve: crv.Curve) -> crv.Curve:
        r_d = self.pitch_radius - self.ref_limits.h_d
        circle_d = crv.ArcCurve(
            center=ORIGIN, radius=r_d, angle=-self.pitch_intersect_angle * 2
        )
        cross_point = crv.find_curve_intersect(tooth_curve, circle_d)
        if not cross_point.success:
            cross_point = crv.find_curve_intersect(
                tooth_curve, circle_d, method=crv.IntersectMethod.MINDISTANCE
            )
            tooth_curve.set_start_on(cross_point.x[0])
            r_low = np.linalg.norm(tooth_curve(0)[:2])
            if r_low > r_d:
                connector_curve = crv.LineCurve(
                    p0=tooth_curve(0) * r_d / r_low, p1=tooth_curve(0)
                )
                return crv.CurveChain(connector_curve, tooth_curve)
            else:
                return tooth_curve

        else:
            tooth_curve.set_start_on(cross_point.x[0])
            return tooth_curve


class GearToothConicGenerator(GearToothGenerator):
    def __init__(
        self,
        pitch_intersect_angle: float = PI / 16,
        pitch_radius: float = 1.0,
        cone_angle: float = PI / 4,
        tooth_angle: float = 0,
        ref_limits: ToothLimitParam = None,
    ):
        self.pitch_intersect_angle = pitch_intersect_angle
        self.pitch_radius = pitch_radius
        self.cone_angle = cone_angle
        self.tooth_angle = tooth_angle
        self.ref_limits = ref_limits if ref_limits is not None else ToothLimitParam()

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
            t_a_axis = OUT * h - p0
            t_a_axis /= np.linalg.norm(t_a_axis)
            rot_ta = scp_Rotation.from_rotvec(
                t_a_axis * self.tooth_angle * np.sign(gamma)
            )
            axis = rot_ta.apply(axis)

            return crv.CurveChain(
                crv.ArcCurve.from_point_center_angle(
                    p0=p0, center=OUT * h, angle=0.1, axis=axis
                )
            )

    def check_lower_curve_limit(self, tooth_curve: crv.Curve) -> crv.Curve:
        if self.cone_angle == 0:
            return super().check_lower_curve_limit(tooth_curve)
        else:
            gamma = self.conic_data.gamma
            R = self.conic_data.R
            h_d_angle = PI / 2 - gamma + self.ref_limits.h_d / R
            z_center = R * np.cos(gamma)
            h_d_point = crv.find_curve_plane_intersect(
                tooth_curve,
                plane_normal=scp_Rotation.from_euler("y", h_d_angle).apply(OUT),
                offset=z_center * OUT,
                guess=0,
            )
            if not h_d_point.success:
                h_d_point = crv.find_curve_plane_intersect(
                    tooth_curve,
                    plane_normal=scp_Rotation.from_euler("y", h_d_angle).apply(OUT),
                    offset=z_center * OUT,
                    method=crv.IntersectMethod.MINDISTANCE,
                    guess=0,
                )
                tooth_curve.set_start_on(h_d_point.x[0])
                bottom_angle = np.abs(
                    angle_between_vector_and_plane(tooth_curve(0), OUT) - h_d_angle
                )
                connector_curve = crv.ArcCurve.from_point_center_angle(
                    p0=tooth_curve(0),
                    center=OUT * R * np.cos(gamma),
                    angle=bottom_angle,
                    axis=UP,
                )
                connector_curve.reverse()
                return crv.CurveChain(connector_curve, tooth_curve)
            else:
                tooth_curve.set_start_on(h_d_point.x[0])
                return tooth_curve


class RackToothGenerator(ZFunctionMixin):
    def __init__(
        self,
        tooth_width: float = PI / 2,
        pitch: float = PI,
        tooth_angle: float = 20 * PI / 180,
    ):
        self.pitch = pitch
        self.tooth_width = tooth_width
        self.tooth_angle = tooth_angle

    def generate_tooth_curve(self) -> crv.Curve:
        p0 = RIGHT
        rot_ta = scp_Rotation.from_euler("z", self.tooth_angle)
        dp = rot_ta.apply(p0)

        return crv.CurveChain(
            crv.LineCurve(p0=DOWN * PI / 4 - dp / 2, p1=DOWN * PI / 4 + dp / 2)
        )


@dataclasses.dataclass
class RecipeKeyParams:
    gamma: float
    h: float
    angle: float
    radius: float
    beta: float

    @property
    def center(self):
        return self.h * OUT

    @property
    def cone_angle(self):
        return self.gamma * 2
