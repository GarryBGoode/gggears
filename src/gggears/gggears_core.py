# Copyright 2024 Gergely Bencsik
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gggears.function_generators import *
from gggears.defs import *
from gggears.gggears_base_classes import *
import gggears.curve as crv

from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
import dataclasses
import copy
from typing import Callable


class GearPolarTransform(ConicData):
    """Callable class for polar transformation of points.

    Uses a cylindric transform for 0 cone angle, otherwise spherical polar transform."""

    def __init__(self, cone_angle=0, base_radius=1):
        self.cone_angle = cone_angle
        self.base_radius = base_radius

    def __call__(self, point):
        return self.polar_transform(point)

    # shorthand
    def inv(self, point) -> np.ndarray:
        return self.inverse_polar_transform(point)

    def polar_transform(self, point) -> np.ndarray:
        """
        Return polar coordinates helpful for gear generation.

        Parameters
        ----------
        point : np.ndarray
            A 3D point to be transformed.

        Returns
        -------
        np.ndarray
            The transformed point in polar coordinates.
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

    def inverse_polar_transform(self, point) -> np.ndarray:
        """Inverse of the `polar_transform`."""
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
class FilletParam:
    """Data class for tooth tip and root modification parameters.

    Attributes
    ----------
    tip_fillet : float
        Tip fillet radius coefficent.
    root_fillet : float
        Root fillet radius coefficent.
    tip_reduction : float
        Tip reduction (truncation) coefficient.
    """

    tip_fillet: float = 0.0
    root_fillet: float = 0.0
    tip_reduction: float = 0.0


def generate_reference_circles(
    pitch_radius, limitparam: ToothLimitParam, coneparam: GearPolarTransform
) -> GearRefCircles:
    """Generates reference circles as Curve objects for a gear tooth."""
    # TODO: this could be a classmethod of GearRefCircles?
    p0 = RIGHT * pitch_radius
    if limitparam.h_o > pitch_radius - DELTA:
        h_o = pitch_radius - DELTA
    else:
        h_o = limitparam.h_o

    pa = coneparam.inverse_polar_transform(
        coneparam.polar_transform(p0) + np.array([limitparam.h_a, 0, 0])
    )
    pd = coneparam.inverse_polar_transform(
        coneparam.polar_transform(p0) + np.array([-limitparam.h_d, 0, 0])
    )
    po = coneparam.inverse_polar_transform(
        coneparam.polar_transform(p0) + np.array([-h_o, 0, 0])
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
    tooth_curve: crv.CurveChain,
    addendum_height: float,
    dedendum_height: float,
    tip_reduction: float,
    polar_transformer: GearPolarTransform,
) -> float:
    """Apply tip reduction feature and return radius.

    Checks if tip reduction is necessary (due to sharp point tip) and returns the
    reduced addendum radius to be used for truncated tip.

    Returns
    -------
    float
        The reduced addendum radius."""
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
) -> crv.CurveChain:
    """Apply fillet to the tooth curve.

    Parameters
    ----------
    tooth_curve : crv.CurveChain
        The tooth curve to be filleted.
    pitch_angle : float
        The pitch angle of the gear in radians.
    target_circle : crv.ArcCurve
        The circle to be filleted to. Either addendum or dedendum circle.
    fillet_radius : float
        The radius of the fillet.
    direction : int
        Use -1 for tip fillet, 1 for root fillet."""

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
    """Data class collecting all curve segments for a single tooth's profile.

    This class should contain all data necessary to construct a single tooth's profile,
    which is 1 unit of the repeating pattern making up the gear."""

    ra_curve: crv.ArcCurve
    rd_curve: crv.ArcCurve
    ro_curve: crv.ArcCurve
    tooth_curve: crv.Curve
    tooth_curve_mirror: crv.MirroredCurve
    pitch_angle: float
    transform: GearTransform

    @property
    def profile(self):
        return crv.CurveChain(
            self.rd_curve, self.tooth_curve, self.ra_curve, self.tooth_curve_mirror
        )


@dataclasses.dataclass
class GearRefProfileExtended(GearRefProfile):
    """Data class with additional curve segments around the tooth profile, such as
    connectors to the outside ring."""

    ro_connector_0: crv.Curve
    ro_connector_1: crv.Curve
    ro_connector_2: crv.Curve
    rd_connector: crv.Curve
    ra_connector: crv.Curve
    ro_curve_tooth: crv.Curve
    ro_curve_dedendum: crv.Curve
    tooth_centerline: crv.Curve

    @property
    def profile_closed(self):
        """Closed curve chain that includes the tooth and the dedendum arc, returns via
        the outside (or inside) ring."""
        return crv.CurveChain(
            self.rd_curve,
            self.tooth_curve,
            self.ra_curve,
            self.tooth_curve_mirror,
            self.ro_connector_2.copy().reverse(),
            self.ro_curve.copy().reverse(),
            self.ro_connector_0,
        )

    @property
    def tooth_profile_closed(self):
        """Closed curve chain that includes just the tooth, closes at the root of the
        tooth with an arc."""
        return crv.CurveChain(
            self.tooth_curve,
            self.ra_curve,
            self.tooth_curve_mirror,
            self.rd_connector.copy().reverse(),
        )

    @property
    def tooth_profile_closed_outer(self):
        """Closed curve chain that includes just the tooth as it were part of an
        outside-ring gear, closing at the addendum arc."""

        return crv.CurveChain(
            crv.RotatedCurve(
                self.tooth_curve_mirror, angle=-self.pitch_angle, axis=OUT
            ),
            self.rd_curve,
            self.tooth_curve,
            self.ra_connector,
        )

    @property
    def tooth_profile_closed_ring(self):
        """Closed curve chain that includes just the tooth, closes at the outside (or
        inside) ring."""
        return crv.CurveChain(
            self.tooth_curve,
            self.ra_connector,
            self.tooth_curve_mirror,
            self.ro_connector_2,
            self.ro_curve_tooth.copy().reverse(),
            self.ro_connector_1.copy().reverse(),
        )

    @classmethod
    def from_refprofile(cls, profile: GearRefProfile, cone: ConicData):
        return generate_profile_extensions(profile, cone)


def trim_reference_profile(
    tooth_curve: crv.Curve,
    ref_curves: GearRefCircles,
    fillet: FilletParam,
    pitch_angle: float,
) -> GearRefProfile:
    """Find intersections of tooth curve and reference curves and trim them."""

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
        # angle check: check if the found solution is within the pitch angle range
        angle_check = np.arctan2(
            tooth_curve(sol_root.x[0])[1], tooth_curve(sol_root.x[0])[0]
        )
        if (sol_root.success or solcheck < 1e-5) and angle_check > -pitch_angle / 2:
            tooth_curve.set_start_on(sol_root.x[0])
        else:
            plane_norm = rotate_vector(UP, -pitch_angle / 2)
            if sol_root.success or solcheck < 1e-5:
                guess = sol_root.x[0]
            else:
                guess = 0.1
            sol_mid2 = crv.find_curve_plane_intersect(
                tooth_curve, plane_normal=plane_norm, guess=guess
            )

            # sol_bot = crv.find_curve_intersect(
            #     tooth_curve,
            #     ref_curves.r_d_curve,
            #     guess=[sol_mid2.x[0], rd_guess],
            #     method=crv.IntersectMethod.MINDISTANCE,
            # )
            sol_bot = minimize(
                lambda t: np.linalg.norm(tooth_curve(t)[:2]), sol_mid2.x[0]
            )
            if sol_bot.x[0] > sol_mid2.x[0]:
                sol_mid2 = sol_bot

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
    if rd_curve.length < 1e-9:
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
        ra_curve,
        rd_curve,
        ro_curve,
        tooth_curve,
        tooth_mirror,
        pitch_angle,
        GearTransform(),
    )


###############################################################################
###############################################################################


@dataclasses.dataclass
class GearProfileDataCollector:
    """
    All input data collected to be able to generate 1 reference profile
    for an involute gear.
    """

    tooth_generator: GearToothConicGenerator
    cone: ConicData
    limits: ToothLimitParam
    pitch_angle: float
    transform: GearTransformData
    fillet: FilletParam


def generate_reference_profile(inputdata: GearProfileDataCollector) -> GearRefProfile:
    """Perform all steps to generate a single tooth profile for an involute gear."""
    conic_transform = GearPolarTransform(
        cone_angle=inputdata.cone.cone_angle, base_radius=inputdata.cone.base_radius
    )
    ref_curves = generate_reference_circles(
        inputdata.tooth_generator.pitch_radius,
        inputdata.limits,
        conic_transform,
    )
    tooth_curve = inputdata.tooth_generator.generate_tooth_curve()

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


def generate_profile_extensions(
    profile: GearRefProfile, cone_data: ConicData
) -> GearRefProfileExtended:
    """Generate additional curve segments around the tooth profile, such as connectors"""
    tooth_start_point = profile.tooth_curve(0)
    tooth_start_plane_normal = np.cross(OUT, normalize_vector(tooth_start_point))
    sol_ro_midpoint = crv.find_curve_plane_intersect(
        profile.ro_curve, plane_normal=tooth_start_plane_normal, guess=0.5
    )
    ro_midpoint = profile.ro_curve(sol_ro_midpoint.x[0])
    ro_curve_tooth = profile.ro_curve.copy()
    ro_curve_tooth.set_start_on(sol_ro_midpoint.x[0])
    ro_curve_dedendum = profile.ro_curve.copy()
    ro_curve_dedendum.set_end_on(sol_ro_midpoint.x[0])

    rd_connector = crv.ArcCurve.from_2_point_center(
        p0=profile.tooth_curve(0),
        p1=profile.tooth_curve_mirror(1),
        center=profile.rd_curve.center,
    )

    ra_connector = crv.ArcCurve.from_2_point_center(
        p0=rotate_vector(profile.ra_curve(1), -profile.pitch_angle),
        p1=profile.ra_curve(0),
        center=profile.ra_curve.center,
    )

    if cone_data.cone_angle == 0:
        ro_connector_0 = crv.LineCurve(p0=profile.ro_curve(0), p1=profile.profile(0))
        ro_connector_1 = crv.LineCurve(p0=ro_midpoint, p1=profile.tooth_curve(0))
        ro_connector_2 = crv.LineCurve(
            p0=profile.ro_curve(1), p1=profile.tooth_curve_mirror(1)
        )
        tooth_centerline = crv.LineCurve(p0=rd_connector(0.5), p1=profile.ra_curve(0.5))
    else:
        ro_connector_0 = crv.ArcCurve.from_2_point_center(
            p0=profile.ro_curve(0),
            p1=profile.profile(0),
            center=cone_data.center,
        )
        ro_connector_1 = crv.ArcCurve.from_2_point_center(
            p0=ro_midpoint,
            p1=profile.tooth_curve(0),
            center=cone_data.center,
        )
        ro_connector_2 = crv.ArcCurve.from_2_point_center(
            p0=profile.ro_curve(1),
            p1=profile.tooth_curve_mirror(1),
            center=cone_data.center,
        )
        tooth_centerline = crv.ArcCurve.from_2_point_center(
            p0=rd_connector(0.5), p1=profile.ra_curve(0.5), center=cone_data.center
        )

    return GearRefProfileExtended(
        ra_curve=profile.ra_curve,
        rd_curve=profile.rd_curve,
        ro_curve=profile.ro_curve,
        tooth_curve=profile.tooth_curve,
        tooth_curve_mirror=profile.tooth_curve_mirror,
        pitch_angle=profile.pitch_angle,
        transform=profile.transform,
        ro_connector_0=ro_connector_0,
        ro_connector_1=ro_connector_1,
        ro_connector_2=ro_connector_2,
        rd_connector=rd_connector,
        ra_connector=ra_connector,
        ro_curve_tooth=ro_curve_tooth,
        ro_curve_dedendum=ro_curve_dedendum,
        tooth_centerline=tooth_centerline,
    )


def generate_profile_closed(profile: GearRefProfile, cone_data: ConicData):
    """Generate a closed profile for a gear tooth.

    The profile contains 2 tooth flanks, 1 addendum and 1 dedendum curve segments,
    the outside (or inside) ring curve segment, and 2 connector curves.
    Ordering: dedendum->tooth->addendum->tooth_mirror->connector_1->outside_ring->connector_0.
    """
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


def generate_boundary_chain(
    profile: GearRefProfile, toothdata: GearToothParam
) -> crv.CurveChain:
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


def generate_boundary(profile: GearRefProfile, toothdata: GearToothParam) -> crv.Curve:
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

    return crv.Curve(
        loc_func,
        t0=0,
        t1=1,
        params={"curve": profile.profile},
        lenght_approx_ndiv=toothdata.num_teeth * 20,
    )


# "Recipe" names should refer to parameter sets that define certain kinds of gears in
# 3D, eg. bevel, helical, etc. using callable parameters that represent the
# parameter value as a function of the extrusion distance z.


class GearProfileRecipe(GearProfileDataCollector, ZFunctionMixin):
    pass


class ConicDataRecipe(ConicData, ZFunctionMixin):
    pass


class ToothLimitParamRecipe(ToothLimitParam, ZFunctionMixin):
    pass


class GearTransformRecipe(GearTransformData, ZFunctionMixin):
    pass


class FilletDataRecipe(FilletParam, ZFunctionMixin):
    pass


def default_gear_recipe(
    teeth_data: GearToothParam,
    tooth_generator: GearToothConicGenerator,
    module: float = 1,
    cone_angle=0,
) -> GearProfileRecipe:
    """This creates the default recipe for a 3D gear tooth profile.

    Recipe refers to a collection of parameter-generator callable functions that
    describe how the tooth profile changes along the extrusion direction (z axis).
    This function creates a recipe considering cone angle and number of teeth."""
    rp_ref = teeth_data.num_teeth / 2
    pitch_angle = 2 * PI / teeth_data.num_teeth
    gamma = cone_angle / 2
    tooth_generator.pitch_radius = rp_ref
    tooth_generator.cone_angle = cone_angle
    tooth_generator.pitch_intersect_angle = pitch_angle / 4
    return GearProfileRecipe(
        tooth_generator=tooth_generator,
        cone=ConicData(base_radius=rp_ref, cone_angle=cone_angle),
        limits=ToothLimitParam(),
        pitch_angle=teeth_data.pitch_angle,
        transform=GearTransformRecipe(
            scale=lambda z: 1 * (1 - z * 2 * np.sin(gamma) / teeth_data.num_teeth),
            center=lambda z: 1 * z * OUT * np.cos(gamma),
        ),
        fillet=FilletDataRecipe(),
    )


class Gear:
    """Manager class that pulls everything together to generate a 3D gear."""

    def __init__(
        self,
        z_vals: np.ndarray = np.array([0, 1]),
        module: float = 1,
        tooth_param: GearToothParam = None,
        tooth_generator: GearToothGenerator = None,
        shape_recipe: GearProfileRecipe = None,
        transform: GearTransform = None,
        cone: ConicData = None,
    ):
        self.module = module
        self.z_vals = z_vals
        if tooth_generator is None:
            # it is updated in the default recipe
            self.tooth_generator = GearToothConicGenerator()
        else:
            self.tooth_generator = tooth_generator
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
                teeth_data=tooth_param,
                tooth_generator=self.tooth_generator,
                module=module,
                cone_angle=self.cone.cone_angle,
            )
        else:
            self.shape_recipe = shape_recipe
        if transform is None:
            self.transform = GearTransform(scale=self.module)
        else:
            self.transform = transform

    @property
    def rp(self):
        return self.tooth_param.num_teeth / 2 * self.module

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
        return generate_reference_profile(self.shape_recipe(z))

    def boundary_at_z(self, z):
        return crv.TransformedCurve(
            self.transform, generate_boundary(self.curve_gen_at_z(z), self.tooth_param)
        )

    def copy(self) -> "Gear":
        return copy.deepcopy(self)

    def swap_tooth_generator(self, tooth_generator: GearToothGenerator):
        tg = copy.deepcopy(tooth_generator)
        tg.pitch_radius = self.shape_recipe.tooth_generator.pitch_radius
        tg.cone_angle = self.shape_recipe.tooth_generator.cone_angle
        tg.pitch_intersect_angle = (
            self.shape_recipe.tooth_generator.pitch_intersect_angle
        )
        self.tooth_generator = tg
        self.shape_recipe.tooth_generator = tg

    def mesh_to(self, other: "Gear", target_dir=RIGHT, distance_offset=0):
        """
        Move this gear into a meshing position with other gear,
        so that the point of contact of the pitch circles is in target_dir direction.
        """
        self.transform.center = calc_mesh_placement_vector(
            self, other, target_dir, distance_offset
        )
        self.transform.orientation = calc_mesh_orientation(
            self, other, target_dir, distance_offset
        )
        self.transform.angle = calc_mesh_angle(self, other)


def calc_mesh_angle(gear1: "Gear", gear2: "Gear"):
    """Calculate the rotation angle for this gear to mesh with the other gear. A
    bias value can be used for positioning within backlash."""

    center_diff_dir = gear2.transform.center - gear1.transform.center
    if np.linalg.norm(center_diff_dir) < 1e-12:
        # if gears are co-axial, use x axis of other gear as reference
        center_diff_dir = gear2.transform.x_axis
    else:
        center_diff_dir = normalize_vector(center_diff_dir)

    if gear1.tooth_param.inside_teeth:
        contact_dir_gear1 = center_diff_dir
        contact_dir_other = center_diff_dir
        phase_offset = 0
        phase_sign = 1
    elif gear2.tooth_param.inside_teeth:
        contact_dir_gear1 = -center_diff_dir
        contact_dir_other = -center_diff_dir
        phase_offset = 0
        phase_sign = 1
    else:
        contact_dir_gear1 = center_diff_dir
        contact_dir_other = -center_diff_dir
        phase_offset = 0.5
        phase_sign = -1

    # mult from right: vector in other's coordinate system
    angle_of_other = angle_of_vector_in_xy(
        contact_dir_other @ gear2.transform.orientation
    )
    target_angle_gear1 = angle_of_vector_in_xy(
        contact_dir_gear1 @ gear1.transform.orientation
    )

    phase_of_other = (
        (gear2.transform.angle - angle_of_other) / gear2.tooth_param.pitch_angle
    ) % 1

    angle_out = (
        target_angle_gear1
        + ((phase_sign * phase_of_other + phase_offset) % 1)
        * gear1.tooth_param.pitch_angle
    )

    return angle_out


def calc_mesh_placement_vector(
    gear1: Gear, gear2: "Gear", target_dir: np.ndarray = RIGHT, offset: float = 0
):
    target_dir_norm = (
        target_dir - np.dot(target_dir, gear2.transform.z_axis) * gear2.transform.z_axis
    )
    if np.linalg.norm(target_dir_norm) < 1e-12:
        # target_dir is parallel to x axis
        target_dir_norm = gear2.transform.x_axis
    else:
        target_dir_norm = normalize_vector(target_dir_norm)

    if gear1.cone.cone_angle == 0 and gear2.cone.cone_angle == 0:
        if gear2.tooth_param.inside_teeth:
            distance_ref = gear2.rp - gear1.rp + offset
        elif gear1.tooth_param.inside_teeth:
            distance_ref = gear2.rp - gear1.rp - offset
        else:
            distance_ref = gear1.rp + gear2.rp + offset
        return gear2.center + distance_ref * target_dir_norm
    else:
        if gear2.tooth_param.inside_teeth:
            angle_ref = gear2.cone.gamma - gear1.cone.gamma + offset / gear1.R
        elif gear1.tooth_param.inside_teeth:
            angle_ref = gear2.cone.gamma - gear1.cone.gamma - offset / gear1.R
        else:
            angle_ref = gear1.cone.gamma + gear2.cone.gamma + offset / gear1.R
        rot_ax = normalize_vector(np.cross(target_dir_norm, gear2.transform.z_axis))
        rot1 = scp_Rotation.from_rotvec(rot_ax * angle_ref)
        center_h = gear1.cone.R * gear1.module * np.cos(gear1.cone.gamma)
        diff_vector = rot1.apply(-center_h * gear2.transform.z_axis)
        return gear2.center_sphere + diff_vector


def calc_mesh_orientation(
    gear1: "Gear", gear2: "Gear", target_dir: np.ndarray = RIGHT, offset: float = 0
):
    target_dir_norm = (
        target_dir - np.dot(target_dir, gear2.transform.z_axis) * gear2.transform.z_axis
    )
    if np.linalg.norm(target_dir_norm) < 1e-12:
        # target_dir is parallel to x axis
        target_dir_norm = gear2.transform.x_axis
    else:
        target_dir_norm = normalize_vector(target_dir_norm)
    if gear1.cone.cone_angle == 0 and gear2.cone.cone_angle == 0:
        return gear2.transform.orientation
    else:
        if gear2.tooth_param.inside_teeth:
            angle_ref = gear2.cone.gamma - gear1.cone.gamma + offset / gear1.R
        elif gear1.tooth_param.inside_teeth:
            angle_ref = gear2.cone.gamma - gear1.cone.gamma - offset / gear1.R
        else:
            angle_ref = gear1.cone.gamma + gear2.cone.gamma + offset / gear1.R
        rot_ax = normalize_vector(np.cross(target_dir_norm, gear2.transform.z_axis))
        rot1 = scp_Rotation.from_rotvec(rot_ax * angle_ref)
        return rot1.as_matrix() @ gear2.transform.orientation
