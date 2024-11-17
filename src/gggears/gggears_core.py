"""
Copyright 2024 Gergely Bencsik
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from gggears.function_generators import *
from gggears.defs import *
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
import dataclasses
import gggears.curve as crv
import copy
from typing import Callable


# If a dataclass tends to be user input, it should be named param.
# If a dataclass tends to be generated or manipulated by functions,
# it should be named data.


@dataclasses.dataclass
class TransformData:
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


def apply_transform(points: np.ndarray, data: TransformData):
    return points @ data.orientation.transpose() * data.scale + data.center


class Transform(TransformData):
    def __call__(self, points):
        return apply_transform(points, self)


@dataclasses.dataclass
class GearTransformData(TransformData):
    """
    Data class for gear base transformation.
    Besides the general base transform, the gear's angle is included.
    This helps track the gear's rotation-advance, phase angle, etc.
    separately from its orientation.
    """

    angle: float = 0


def apply_gear_transform(points: np.ndarray, data: GearTransformData):
    rot_z = scp_Rotation.from_euler("z", data.angle).as_matrix()
    return (
        points @ rot_z.transpose() @ data.orientation.transpose() * data.scale
        + data.center
    )


class GearTransform(GearTransformData):
    def __call__(self, points):
        return apply_gear_transform(points, self)


@dataclasses.dataclass
class GearToothParam:
    """
    Data class for gear teeth.
    By convention, negative teeth number results inverting the gear
    (i.e. inside teeth).
    Non-integer teeth number results in the actual number rounded down,
    but the size of the gear and teeth matching the rational input.
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
        return int(np.floor(self.num_teeth - self.num_cutout_teeth))

    @property
    def pitch_angle(self):
        return 2 * PI / self.num_teeth


@dataclasses.dataclass
class ToothLimitParam:
    h_a: float = 1
    h_d: float = 1.2
    h_o: float = 2


@dataclasses.dataclass
class GearRefCircles:
    r_a_curve: crv.ArcCurve  # addendum circle
    r_p_curve: crv.ArcCurve  # pitch circle
    r_d_curve: crv.ArcCurve  # dedendum circle
    r_o_curve: crv.ArcCurve  # outside (or inside) ring circle

    @property
    def r_a(self):
        return self.r_a_curve.r

    @property
    def r_p(self):
        return self.r_p_curve.r

    @property
    def r_d(self):
        return self.r_d_curve.r

    @property
    def r_o(self):
        return self.r_o_curve.r


@dataclasses.dataclass
class ConicData:
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


class GearPolarTransform(ConicData):
    def __init__(self, cone_angle=0, base_radius=1):
        self.cone_angle = cone_angle
        self.base_radius = base_radius

    def __call__(self, point):
        return self.polar_transform(point)

    # shorthand
    def inv(self, point):
        return self.inverse_polar_transform(point)

    def polar_transform(self, point):
        """
        Return polar coordinates helpful for gear generation.
        Convention: [out - angle - height]
        "out" is the quasi radial coordinate, the gear teeth grow in this direction.
        "angle" is the angle around the gear.
        "height" is the 3rd direction, typically the extrusion direction.
        """
        if self.cone_angle == 0:
            return xyz_to_cylindrical(point)
        else:
            point = xyz_to_spherical(point, center=self.center)
            # R theta phi in spherical
            # quasi r = (PI/2-phi) * self.R
            # theta = theta
            # z = self.R-R
            if point.ndim == 1:
                return np.array(
                    [
                        (PI / 2 - point[2]) * self.R,
                        point[1],
                        (self.R - point[0]),
                    ]
                )
            else:
                return np.array(
                    [
                        (PI / 2 - point[:, 2]) * self.R,
                        point[:, 1],
                        (self.R - point[:, 0]),
                    ]
                ).transpose()

    def inverse_polar_transform(self, point):
        if self.cone_angle == 0:
            return cylindrical_to_xyz(point)
        else:
            if point.ndim == 1:
                point2 = np.array(
                    [self.R - point[2], point[1], PI / 2 - point[0] / self.R]
                )
                return spherical_to_xyz(point2, center=self.center)
            else:
                point2 = np.array(
                    [
                        self.R - point[:, 2],
                        point[:, 1],
                        PI / 2 - point[:, 0] / self.R,
                    ]
                ).transpose()
                return spherical_to_xyz(point2, center=self.center)


@dataclasses.dataclass
class InvoluteConstructData:
    pressure_angle: float = 20 * PI / 180
    angle_pitch_ref: float = PI / 32  # normally a quarter of the pitch angle
    pitch_radius: float = 8.0
    # cone_angle: float = 0  # cone angle should be kept in the cone data


@dataclasses.dataclass
class UndercutData:
    pitch_radius: float = 8.0
    cone_angle: float = 0
    undercut_ref_point: np.ndarray = dataclasses.field(default_factory=lambda: ORIGIN)


@dataclasses.dataclass
class FilletParam:
    tip_fillet: float = 0.0
    root_fillet: float = 0.0
    tip_reduction: float = 0.0


def generate_involute_curve(
    input: InvoluteConstructData, conic_transform: GearPolarTransform
):
    rp = input.pitch_radius
    alpha = input.pressure_angle
    gamma = conic_transform.cone_angle / 2

    if conic_transform.cone_angle == 0:

        involute_curve = crv.InvoluteCurve(
            r=input.pitch_radius * np.cos(alpha), angle=0
        )

        pitch_circle = crv.ArcCurve(radius=rp, angle=2 * PI)

        sol2 = crv.find_curve_intersect(involute_curve, pitch_circle, guess=[0.5, 0])
        involute_angle_0 = angle_between_vectors(RIGHT, involute_curve(sol2.x[0]))

        involute_curve.angle = -(input.angle_pitch_ref + involute_angle_0)
        sol1 = crv.find_curve_plane_intersect(involute_curve, plane_normal=UP, guess=1)
        involute_curve.set_end_on(sol1.x[0])
        connector_line = crv.LineCurve(p0=0.5 * involute_curve(0), p1=involute_curve(0))

        return crv.CurveChain(connector_line, involute_curve)

    else:
        R = rp / np.sin(gamma)
        C_sph = 1 / R  # spherical curvature

        def involute_angle_func(x):
            t = x[0]
            r = x[1]
            p0 = involute_sphere(t, r, angle=0, C=C_sph)
            p1 = involute_sphere(t + DELTA, r, angle=0, C=C_sph)
            p2 = involute_sphere(t - DELTA, r, angle=0, C=C_sph)
            tan = normalize_vector(p1 - p2)
            center = np.array([0, 0, np.sqrt(R**2 - r**2)])
            sph_tan = normalize_vector(
                np.cross(p0 - center, np.array([p0[0], p0[1], 0]))
            )
            angle = angle_between_vectors(tan, sph_tan)
            rad_diff = p0[0] ** 2 + p0[1] ** 2 - rp**2
            angle_diff = angle - PI / 2 - alpha
            return [rad_diff, angle_diff]

        base_res = root(
            involute_angle_func,
            [alpha / 2, rp * np.cos(alpha)],
            tol=1e-14,
        )
        involute_curve = crv.SphericalInvoluteCurve(r=base_res.x[1], c_sphere=C_sph)
        angle_0 = angle_between_vectors(
            involute_sphere(base_res.x[0], base_res.x[1], angle=0, C=C_sph)
            * np.array([1, 1, 0]),
            RIGHT,
        )
        angle_offset = -(input.angle_pitch_ref + angle_0)
        involute_curve.angle = angle_offset
        involute_curve.z_offs = -involute_sphere(base_res.x[0], base_res.x[1], C=C_sph)[
            2
        ]
        sol1 = crv.find_curve_plane_intersect(
            involute_curve, offset=ORIGIN, plane_normal=UP, guess=1
        )
        involute_curve.set_end_on(sol1.x[0])

        connector_curve = crv.ArcCurve.from_point_center_angle(
            p0=involute_curve(0),
            center=involute_curve.center_sphere,
            angle=0.1,
            axis=normalize_vector(np.cross(OUT, involute_curve(0))),
        )
        connector_curve.reverse()

        return crv.CurveChain(connector_curve, involute_curve)


def generate_involute_rack_curve(
    input: InvoluteConstructData,
    ref_limits: ToothLimitParam,
    conic_transform: GearPolarTransform,
):
    if conic_transform.cone_angle == 0:
        rp = input.pitch_radius
        pitch_len_ref = rp * input.angle_pitch_ref
        p0 = RIGHT * rp + DOWN * pitch_len_ref
        direction = rotate_vector(RIGHT, input.pressure_angle)
        p1 = p0 + direction * ref_limits.h_a / direction[0]
        p2 = p0 - direction * ref_limits.h_d / direction[0]
        return crv.LineCurve(p0=p2, p1=p1)
    else:
        alpha = input.pressure_angle

        def rack_flank_func(t, a):
            axis1 = scp_Rotation.from_euler("x", -alpha).apply(OUT)
            v0 = conic_transform.R * RIGHT
            an1 = t * np.cos(alpha)
            v1 = scp_Rotation.from_rotvec(-axis1 * an1).apply(v0)
            v2 = scp_Rotation.from_euler("z", t + a).apply(v1)
            return v2

        an_tooth_sph = conic_transform.r / conic_transform.R * input.angle_pitch_ref
        curve1 = crv.Curve(rack_flank_func, t0=-1, t1=1, params={"a": -an_tooth_sph})

        sol2 = root(
            lambda t: np.arcsin(curve1(t[0])[2] / conic_transform.R)
            + ref_limits.h_d / conic_transform.R,
            [0],
        )

        # sol1 = crv.find_curve_plane_intersect(curve1, plane_normal=UP, guess=1)
        sol1 = root(
            lambda t: np.arcsin(curve1(t[0])[2] / conic_transform.R)
            - ref_limits.h_a / conic_transform.R,
            [0],
        )
        curve1.set_start_and_end_on(sol2.x[0], sol1.x[0])
        return curve1


def generate_undercut_curve(input: UndercutData):
    if input.cone_angle == 0:
        undercut_curve = crv.InvoluteCurve(
            r=input.pitch_radius,
            angle=0,
            v_offs=input.undercut_ref_point - RIGHT * input.pitch_radius,
            t0=0,
            t1=-1,
        )

    else:
        gamma = input.cone_angle / 2
        R = input.pitch_radius / np.sin(gamma)
        C_sph = 1 / R  # spherical curvature
        v_offs = scp_Rotation.from_euler("y", PI / 2 * np.sign(C_sph)).apply(
            input.undercut_ref_point - R * RIGHT
        )
        undercut_curve = crv.SphericalInvoluteCurve(
            r=input.pitch_radius,
            c_sphere=C_sph,
            v_offs=v_offs,
            t0=0,
            t1=-1,
        )

    sol1 = root(
        lambda t: np.dot(undercut_curve(t[0]), undercut_curve(t[0]))
        - input.pitch_radius**2,
        [0.5],
    )
    undercut_curve.set_end_on(sol1.x[0])
    return undercut_curve


def trim_involute_undercut(tooth_curve, undercut_curve, guess=(0.5, 1)):
    sol = crv.find_curve_intersect(tooth_curve, undercut_curve, guess=guess)
    # solcheck = np.linalg.norm(tooth_curve(sol.x[0]) - undercut_curve(sol.x[1]))

    tooth_curve.set_start_on(sol.x[0])
    undercut_curve.set_end_on(sol.x[1])
    return crv.CurveChain(undercut_curve, tooth_curve)


def generate_reference_circles(
    rp, limitparam: ToothLimitParam, coneparam: GearPolarTransform
):
    p0 = RIGHT * rp
    pa = coneparam.inverse_polar_transform(
        coneparam.polar_transform(p0) + np.array([limitparam.h_a, 0, 0])
    )
    pd = coneparam.inverse_polar_transform(
        coneparam.polar_transform(p0) + np.array([-limitparam.h_d, 0, 0])
    )
    po = coneparam.inverse_polar_transform(
        coneparam.polar_transform(p0) + np.array([-limitparam.h_o, 0, 0])
    )

    rp_circle = crv.ArcCurve.from_point_center_angle(
        p0=p0, center=OUT * p0[2], angle=2 * PI
    )
    ra_circle = crv.ArcCurve.from_point_center_angle(
        p0=pa, center=OUT * pa[2], angle=2 * PI
    )
    rd_circle = crv.ArcCurve.from_point_center_angle(
        p0=pd, center=OUT * pd[2], angle=2 * PI
    )
    ro_circle = crv.ArcCurve.from_point_center_angle(
        p0=po, center=OUT * po[2], angle=2 * PI
    )
    return GearRefCircles(ra_circle, rp_circle, rd_circle, ro_circle)


def apply_tip_reduction(
    tooth_curve,
    addendum_height,
    dedendum_height,
    tip_reduction,
    polar_transformer: GearPolarTransform,
):
    soldata = []
    rah = addendum_height
    rdh = dedendum_height
    r_out = rah
    for guess in np.linspace(0.1, 0.9, 4):
        sol1 = crv.find_curve_plane_intersect(tooth_curve, plane_normal=UP, guess=guess)
        r_sol = polar_transformer.polar_transform(tooth_curve(sol1.x[0]))[0]
        if sol1.success and r_sol > rdh:
            soldata.append((sol1, r_sol))

    if len(soldata) > 0:
        sol, r_sol = soldata[np.argmin([soltuple[1] for soltuple in soldata])]

        if r_sol - tip_reduction < rah:
            if tip_reduction > 0:
                r_out = r_sol - tip_reduction
            else:
                r_out = r_sol

    return r_out


def apply_fillet(
    tooth_curve: crv.CurveChain,
    pitch_angle: float,
    target_circle: crv.ArcCurve,
    fillet_radius: float,
    direction=1,
):
    def angle_check(p):
        angle = -np.arctan2(p[1], p[0])
        return 0 < angle < pitch_angle / 2

    sol1 = crv.find_curve_intersect(
        tooth_curve,
        target_circle,
        guess=[0.5, -pitch_angle / 4 / (2 * PI) * 1.01],
    )
    if sol1.success and angle_check(target_circle(sol1.x[1])):
        sharp_root = False
        guesses = np.asarray([0.5, 1, 1.5]) * fillet_radius
        if direction == 1:
            for guess in guesses:
                start_locations = [
                    sol1.x[1] - guess / target_circle.length,
                    sol1.x[0] + guess / tooth_curve.length,
                ]
                arc, t1, t2, sol = crv.calc_tangent_arc(
                    target_circle,
                    tooth_curve,
                    fillet_radius,
                    start_locations=start_locations,
                )
                if sol.success:
                    break
        else:
            for guess in guesses:
                start_locations = [
                    sol1.x[0] - guess / tooth_curve.length,
                    sol1.x[1] + guess / target_circle.length,
                ]
                arc, t1, t2, sol = crv.calc_tangent_arc(
                    tooth_curve,
                    target_circle,
                    fillet_radius,
                    start_locations=start_locations,
                )
                if sol.success:
                    break
        if angle_check(arc(0)) and angle_check(arc(1)):
            if direction == 1:
                tooth_curve.set_start_on(t2)
                tooth_curve.insert(0, arc)
            else:
                tooth_curve.set_end_on(t1)
                tooth_curve.append(arc)
        else:
            sharp_root = True
    else:
        sharp_root = True

    if sharp_root:
        if direction == 1:
            plane_normal = rotate_vector(UP, -pitch_angle / 2)
        else:
            plane_normal = UP
        mirror_curve = crv.MirroredCurve(tooth_curve, plane_normal=plane_normal)
        mirror_curve.reverse()
        start_locations = [
            1 - fillet_radius / tooth_curve.length,
            0 + fillet_radius / tooth_curve.length,
        ]

        if direction == 1:
            arc, t1, t2, sol = crv.calc_tangent_arc(
                mirror_curve,
                tooth_curve,
                fillet_radius,
                start_locations=start_locations,
            )
            arc.set_start_on(0.5)
            tooth_curve.set_start_on(t2)
            tooth_curve.insert(0, arc)

        else:
            arc, t1, t2, sol = crv.calc_tangent_arc(
                tooth_curve,
                mirror_curve,
                fillet_radius,
                start_locations=start_locations,
            )
            arc.set_end_on(0.5)
            tooth_curve.set_end_on(t1)
            tooth_curve.append(arc)

    return tooth_curve


@dataclasses.dataclass
class GearRefProfile:
    ra_curve: crv.ArcCurve
    rd_curve: crv.ArcCurve
    ro_curve: crv.ArcCurve
    tooth_curve: crv.Curve
    tooth_curve_mirror: crv.MirroredCurve
    profile: crv.CurveChain
    pitch_angle: float
    transform: GearTransform = dataclasses.field(
        default_factory=lambda: GearTransform()
    )


def trim_reference_profile(
    tooth_curve: crv.Curve,
    ref_curves: GearRefCircles,
    # transform: GearPolarTransform,
    fillet: FilletParam,
    pitch_angle: float,
):

    # if tip fillet is used, tooth curve tip is already settled
    # in fact this solver tends to fail due to tangential nature of fillet
    if not fillet.tip_fillet > 0:
        ra_guess = -pitch_angle / 8 / ref_curves.r_a_curve.length
        sol_tip = crv.find_curve_intersect(
            tooth_curve,
            ref_curves.r_a_curve,
            guess=[0.9, ra_guess],
            method=crv.IntersectMethod.EQUALITY,
        )
        if not sol_tip.success:
            # try the other way
            sol_tip = crv.find_curve_intersect(
                tooth_curve,
                ref_curves.r_a_curve,
                guess=[0.9, ra_guess],
                method=crv.IntersectMethod.MINDISTANCE,
            )
        solcheck = np.linalg.norm(
            tooth_curve(sol_tip.x[0]) - ref_curves.r_a_curve(sol_tip.x[1])
        )
        if (sol_tip.success or solcheck < 1e-5) and tooth_curve(sol_tip.x[0])[1] < 0:
            tooth_curve.set_end_on(sol_tip.x[0])
        else:
            sol_mid = crv.find_curve_plane_intersect(
                tooth_curve, plane_normal=UP, guess=1
            )
            tooth_curve.set_end_on(sol_mid.x[0])

    if not fillet.root_fillet > 0:
        rd_guess = -pitch_angle / 2 / ref_curves.r_d_curve.length
        sol_root = crv.find_curve_intersect(
            tooth_curve,
            ref_curves.r_d_curve,
            guess=[0.3, rd_guess],
            method=crv.IntersectMethod.EQUALITY,
        )
        solcheck = np.linalg.norm(
            tooth_curve(sol_root.x[0]) - ref_curves.r_d_curve(sol_root.x[1])
        )
        if not sol_root.success:
            # try the other way
            sol_root_2 = crv.find_curve_intersect(
                tooth_curve,
                ref_curves.r_d_curve,
                guess=[0, rd_guess],
                method=crv.IntersectMethod.MINDISTANCE,
            )
            solcheck2 = np.linalg.norm(
                tooth_curve(sol_root.x[0]) - ref_curves.r_d_curve(sol_root.x[1])
            )
            if sol_root_2.success or solcheck2 < 1e-5:
                solcheck = solcheck2
                sol_root = sol_root_2
        angle_check = np.arctan2(
            tooth_curve(sol_root.x[0])[1], tooth_curve(sol_root.x[0])[0]
        )
        if (sol_root.success or solcheck < 1e-5) and angle_check > -pitch_angle / 2:
            tooth_curve.set_start_on(sol_root.x[0])
        else:
            plane_norm = rotate_vector(UP, -pitch_angle / 2)
            sol_mid2 = crv.find_curve_plane_intersect(
                tooth_curve, plane_normal=plane_norm, guess=0
            )
            tooth_curve.set_start_on(sol_mid2.x[0])

    tooth_mirror = crv.MirroredCurve(tooth_curve, plane_normal=UP)
    tooth_mirror.reverse()
    tooth_rotate = crv.RotatedCurve(tooth_mirror, angle=-pitch_angle, axis=OUT)

    pa1 = tooth_curve(1)
    pa2 = tooth_mirror(0)
    center_a = ((pa1 + pa2) / 2 * np.array([0, 0, 1])) * OUT
    ra_curve = crv.ArcCurve.from_2_point_center(p0=pa1, p1=pa2, center=center_a)
    if ra_curve.length < 1e-9:
        ra_curve.active = False

    pd1 = tooth_curve(0)
    pd2 = tooth_rotate(1)
    center_d = ((pd1 + pd2) / 2 * np.array([0, 0, 1])) * OUT
    rd_curve = crv.ArcCurve.from_2_point_center(p0=pd2, p1=pd1, center=center_d)
    if rd_curve.length < DELTA**2:
        rd_curve.active = False

    profile = crv.CurveChain(rd_curve, tooth_curve, ra_curve, tooth_mirror)
    angle_0 = np.arctan2(profile(0)[1], profile(0)[0])
    angle_1 = np.arctan2(profile(1)[1], profile(1)[0])

    ro_curve = crv.ArcCurve(
        ref_curves.r_o_curve.r,
        center=ref_curves.r_o_curve.center,
        angle=angle_1 - angle_0,
        yaw=angle_0,
    )
    return GearRefProfile(
        ra_curve, rd_curve, ro_curve, tooth_curve, tooth_mirror, profile, pitch_angle
    )


###############################################################################
###############################################################################


@dataclasses.dataclass
class InvoluteProfileDataCollector:
    """
    All data collected to be able to generate 1 reference profile
    for an involute gear.
    """

    involute: InvoluteConstructData
    cone: ConicData
    limits: ToothLimitParam
    pitch_angle: float
    transform: GearTransformData
    fillet: FilletParam


def generate_reference_profile(
    inputdata: InvoluteProfileDataCollector, enable_undercut=True
) -> GearRefProfile:
    conic_transform = GearPolarTransform(inputdata.cone.cone_angle, inputdata.cone.r)
    ref_curves = generate_reference_circles(
        inputdata.involute.pitch_radius, inputdata.limits, conic_transform
    )
    tooth_curve = generate_involute_curve(inputdata.involute, conic_transform)
    if enable_undercut:
        rack_curve = generate_involute_rack_curve(
            inputdata.involute, inputdata.limits, conic_transform
        )
        tooth_curve_ucut = generate_undercut_curve(
            UndercutData(
                inputdata.involute.pitch_radius,
                inputdata.cone.cone_angle,
                rack_curve(0),
            )
        )
        tooth_curve = trim_involute_undercut(tooth_curve, tooth_curve_ucut)

    if inputdata.fillet.tip_reduction > 0:
        r_ah = apply_tip_reduction(
            tooth_curve=tooth_curve,
            addendum_height=conic_transform.polar_transform(ref_curves.r_a_curve(0))[0],
            dedendum_height=conic_transform.polar_transform(ref_curves.r_d_curve(0))[0],
            tip_reduction=inputdata.fillet.tip_reduction,
            polar_transformer=conic_transform,
        )
        pa = conic_transform.inverse_polar_transform(np.array([r_ah, 0, 0]))
        ref_curves.r_a_curve = crv.ArcCurve.from_point_center_angle(
            p0=pa, center=OUT * pa[2], angle=2 * PI
        )
    if inputdata.fillet.tip_fillet > 0:
        tooth_curve = apply_fillet(
            tooth_curve,
            inputdata.pitch_angle,
            ref_curves.r_a_curve,
            inputdata.fillet.tip_fillet,
            direction=-1,
        )
    if inputdata.fillet.root_fillet > 0:
        tooth_curve = apply_fillet(
            tooth_curve,
            inputdata.pitch_angle,
            ref_curves.r_d_curve,
            inputdata.fillet.root_fillet,
            direction=1,
        )
    profile = trim_reference_profile(
        tooth_curve, ref_curves, inputdata.fillet, inputdata.pitch_angle
    )
    profile.transform = GearTransform(**(inputdata.transform.__dict__))
    return profile


def generate_profile_closed(profile: GearRefProfile, cone_data: ConicData):
    if cone_data.cone_angle == 0:
        ro_connector_0 = crv.LineCurve(p0=profile.ro_curve(0), p1=profile.profile(0))
        ro_connector_1 = crv.LineCurve(p1=profile.ro_curve(1), p0=profile.profile(1))

    else:
        ro_connector_0 = crv.ArcCurve.from_2_point_center(
            p0=profile.ro_curve(0),
            p1=profile.profile(0),
            center=cone_data.center,
        )
        ro_connector_1 = crv.ArcCurve.from_2_point_center(
            p1=profile.ro_curve(1),
            p0=profile.profile(1),
            center=cone_data.center,
        )

    return crv.CurveChain(
        profile.profile.copy(),
        ro_connector_1,
        profile.ro_curve.copy().reverse(),
        ro_connector_0,
    )


def generate_boundary_chain(profile: GearRefProfile, toothdata: GearToothParam):
    """
    Create gear boundary by repeating reference profile in a CurveChain.
    """
    crv_list = []
    for i in range(toothdata.num_teeth_act):
        crv_list.append(
            crv.RotatedCurve(
                curve=profile.profile.copy(), angle=i * toothdata.pitch_angle, axis=OUT
            )
        )
    return crv.CurveChain(*crv_list)


def generate_boundary(profile: GearRefProfile, toothdata: GearToothParam):
    """
    Create gear boundary by defining custom repeating function for the profile.
    """

    def loc_func(t, curve=profile.profile):
        i = t * toothdata.num_teeth_act // 1
        t2 = t * toothdata.num_teeth_act % 1
        return (
            curve(t2)
            @ scp_Rotation.from_euler("z", i * toothdata.pitch_angle).as_matrix().T
        )

    return crv.Curve(loc_func, t0=0, t1=1, params={"curve": profile.profile})


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


# "Recipe" names should refer to parameter sets that define certain kinds of gears in
# 3D, eg. bevel, helical, etc. using callable parameters that represent the
# parameter value as a function of the extrusion distance z.


class InvoluteToothRecipe(InvoluteProfileDataCollector, ZFunctionMixin):
    pass


class InvoluteProfileParamRecipe(InvoluteConstructData, ZFunctionMixin):
    pass


class ConicDataRecipe(ConicData, ZFunctionMixin):
    pass


class ToothLimitParamRecipe(ToothLimitParam, ZFunctionMixin):
    pass


class GearTransformRecipe(GearTransformData, ZFunctionMixin):
    pass


class FilletDataRecipe(FilletParam, ZFunctionMixin):
    pass


def default_gear_recipe(teeth_data: GearToothParam, module: float = 1, cone_angle=0):
    rp_ref = teeth_data.num_teeth / 2
    pitch_angle = 2 * PI / teeth_data.num_teeth
    gamma = cone_angle / 2
    return InvoluteToothRecipe(
        involute=InvoluteConstructData(
            pressure_angle=20 * PI / 180,
            angle_pitch_ref=pitch_angle / 4,
            pitch_radius=rp_ref,
        ),
        cone=ConicData(base_radius=rp_ref, cone_angle=cone_angle),
        limits=ToothLimitParam(),
        pitch_angle=teeth_data.pitch_angle,
        transform=GearTransformRecipe(
            scale=lambda z: 1 * (1 - z * 2 * np.sin(gamma) / teeth_data.num_teeth),
            center=lambda z: 1 * z * OUT * np.cos(gamma),
        ),
        fillet=FilletDataRecipe(),
    )


class InvoluteGear:
    def __init__(
        self,
        z_vals: np.ndarray = np.array([0, 1]),
        module: float = 1,
        tooth_param: GearToothParam = None,
        shape_recipe: InvoluteToothRecipe = None,
        transform: GearTransform = None,
        cone: ConicData = None,
        enable_undercut: bool = True,
    ):
        self.module = module
        self.z_vals = z_vals
        self.enable_undercut = enable_undercut
        if tooth_param is None:
            self.tooth_param = GearToothParam()
        else:
            self.tooth_param = tooth_param
        if cone is None:
            self.cone = ConicData()
        else:
            self.cone = cone
        self.cone.base_radius = tooth_param.num_teeth / 2
        if shape_recipe is None:
            self.shape_recipe = default_gear_recipe(
                teeth_data=tooth_param, module=module, cone_angle=self.cone.cone_angle
            )
        else:
            self.shape_recipe = shape_recipe
        if transform is None:
            self.transform = GearTransform(scale=self.module)
        else:
            self.transform = transform

    @property
    def rp(self):
        return self.shape_recipe(0).involute.pitch_radius * self.module

    @property
    def R(self):
        return self.cone.spherical_radius * self.module

    @property
    def pitch_angle(self):
        return self.tooth_param.pitch_angle

    @property
    def center(self):
        return self.transform.center

    @center.setter
    def center(self, value):
        self.transform.center = value

    @property
    def center_sphere(self):
        return (
            self.transform.center
            + self.R * np.cos(self.cone.gamma) * self.transform.z_axis
        )

    def curve_gen_at_z(self, z):
        return generate_reference_profile(self.shape_recipe(z), self.enable_undercut)

    def copy(self) -> "InvoluteGear":
        return copy.deepcopy(self)

    def mesh_to(self, other: "InvoluteGear", target_dir=RIGHT, distance_offset=0):
        """
        Move this gear into a meshing position with other gear,
        so that the point of contact of the pitch circles is in target_dir direction.
        """
        target_dir_norm = (
            target_dir
            - np.dot(target_dir, other.transform.z_axis) * other.transform.z_axis
        )
        if np.linalg.norm(target_dir_norm) < 1e-12:
            # target_dir is parallel to x axis
            target_dir_norm = other.transform.x_axis
        else:
            target_dir_norm = normalize_vector(target_dir_norm)

        target_plane_norm = np.cross(other.transform.z_axis, target_dir_norm)

        target_angle_other = angle_between_vectors(
            other.transform.x_axis, target_dir_norm
        )
        sign_corrector = np.dot(
            np.cross(other.transform.x_axis, target_dir_norm), other.transform.z_axis
        )
        if sign_corrector < 0:
            target_angle_other = -target_angle_other

        target_phase_other = (
            (target_angle_other - other.transform.angle) / other.tooth_param.pitch_angle
        ) % 1

        if self.cone.gamma == 0 and other.cone.gamma == 0:
            # both are cylindrical
            self.transform.orientation = other.transform.orientation

            if self.tooth_param.inside_teeth or other.tooth_param.inside_teeth:
                phase_offset = 0
                angle_turnaround = 0
                phase_sign = -1
            else:
                phase_offset = 0.5
                angle_turnaround = PI
                phase_sign = 1

            target_angle_self = target_angle_other + angle_turnaround
            angle_offs = (
                target_angle_self
                + (phase_sign * (target_phase_other) - phase_offset)
                * self.tooth_param.pitch_angle
            )
            r1 = self.rp  # + self.paramref.profile_shift * self.paramref.module
            r2 = other.rp
            if self.tooth_param.inside_teeth:
                distance_ref = r2 - r1 + distance_offset
            elif other.tooth_param.inside_teeth:
                distance_ref = r1 - r2 - distance_offset
            else:
                distance_ref = r1 + r2 + distance_offset

            center_offs = distance_ref * target_dir_norm
            self.transform.center = center_offs + other.transform.center
            self.transform.angle = angle_offs + self.transform.angle
            pass

        elif self.cone.gamma != 0 and other.cone.gamma != 0:
            # both are spherical
            # start off by identical orientation
            self.transform.orientation = other.transform.orientation
            # angle-phase math is the same as cylindrical
            if self.tooth_param.inside_teeth or other.tooth_param.inside_teeth:
                phase_offset = 0
                angle_turnaround = 0
                phase_sign = -1
            else:
                phase_offset = 0.5
                angle_turnaround = PI
                phase_sign = 1

            target_angle_self = target_angle_other + angle_turnaround
            angle_offs = (
                target_angle_self
                + (phase_sign * (target_phase_other) - phase_offset)
                * self.tooth_param.pitch_angle
            )
            r1 = self.rp  # + self.paramref.profile_shift * self.paramref.module
            r2 = other.rp

            # compatible bevel gears should have the same spherical radius
            # and the same center sphere when placed on xy plane at the orgin

            if self.tooth_param.inside_teeth:
                distance_ref = r2 - r1 + distance_offset
            elif other.tooth_param.inside_teeth:
                distance_ref = r1 - r2 - distance_offset
            else:
                distance_ref = r1 + r2 + distance_offset

            # angle_ref = distance_ref / self.paramref.R
            angle_ref = self.cone.gamma + other.cone.gamma
            center_sph = np.sqrt(self.R**2 - self.rp**2) * OUT
            center_sph_other = np.sqrt(other.R**2 - other.rp**2) * OUT
            rot1 = scp_Rotation.from_rotvec(-target_plane_norm * angle_ref)
            center_offs = rot1.apply(-center_sph) + center_sph_other
            self.transform.orientation = other.transform.orientation @ rot1.as_matrix()
            self.transform.center = self.transform.center + center_offs
            self.transform.angle = self.transform.angle + angle_offs
            pass  # to be able to stop here in debugging

        else:
            # one is cylindrical, the other is spherical
            Warning("Meshing cylindrical and spherical gears are not supported")
