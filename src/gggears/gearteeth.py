from gggears.gggears_base_classes import *
from gggears.function_generators import *
from scipy.optimize import root


class InvoluteTooth(GearToothConicGenerator):
    def __init__(
        self,
        pitch_intersect_angle: float = PI / 16,
        pitch_radius: float = 1.0,
        cone_angle: float = PI / 4,
        pressure_angle: float = 20 * PI / 180,
    ):
        self.pitch_intersect_angle = pitch_intersect_angle
        self.pitch_radius = pitch_radius
        self.cone_angle = cone_angle
        self.pressure_angle = pressure_angle

    def generate_tooth_curve(self) -> crv.CurveChain:
        return self.generate_involute_curve()

    def generate_involute_curve(self) -> crv.CurveChain:
        """Generate involute curve for a gear tooth.

        Returns
        -------
        crv.CurveChain
            A curve chain containing the involute curve and the connector line extending
            it at the root. Curve starts at the connector bottom and moves towards the
            tip.
        """
        rp = self.pitch_radius
        alpha = self.pressure_angle
        gamma = self.cone_angle / 2

        if self.cone_angle == 0:

            involute_curve = crv.InvoluteCurve(
                r=self.pitch_radius * np.cos(alpha), angle=0
            )

            pitch_circle = crv.ArcCurve(radius=rp, angle=2 * PI)

            sol2 = crv.find_curve_intersect(
                involute_curve, pitch_circle, guess=[0.5, 0]
            )
            involute_angle_0 = angle_between_vectors(RIGHT, involute_curve(sol2.x[0]))

            involute_curve.angle = -(self.pitch_intersect_angle + involute_angle_0)
            sol1 = crv.find_curve_plane_intersect(
                involute_curve, plane_normal=UP, guess=1
            )
            involute_curve.set_end_on(sol1.x[0])
            connector_line = crv.LineCurve(
                p0=0.5 * involute_curve(0), p1=involute_curve(0)
            )

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
            angle_offset = -(self.pitch_intersect_angle + angle_0)
            involute_curve.angle = angle_offset
            involute_curve.z_offs = -involute_sphere(
                base_res.x[0], base_res.x[1], C=C_sph
            )[2]
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


class InvoluteUndercutTooth(InvoluteTooth):
    def __init__(
        self,
        pitch_intersect_angle: float = PI / 16,
        pitch_radius: float = 1.0,
        cone_angle: float = 0,
        pressure_angle: float = 20 * PI / 180,
        ref_limits: ToothLimitParam = None,
    ):
        self.pitch_intersect_angle = pitch_intersect_angle
        self.pitch_radius = pitch_radius
        self.cone_angle = cone_angle
        self.pressure_angle = pressure_angle
        if ref_limits is None:
            self.ref_limits = ToothLimitParam(h_a=1, h_d=1.2)
        self.ref_limits = ref_limits

    def generate_tooth_curve(self) -> crv.CurveChain:
        tooth_curve = self.generate_involute_curve()
        undercut_ref_point = self.get_default_undercut_ref_point()
        undercut_curve = generate_undercut_curve(
            pitch_radius=self.pitch_radius,
            cone_angle=self.cone_angle,
            undercut_ref_point=undercut_ref_point,
        )
        return trim_involute_undercut(tooth_curve, undercut_curve)

    def get_default_undercut_ref_point(
        self,
    ) -> np.ndarray:
        return generate_involute_rack_curve(
            self.pitch_radius,
            self.pitch_intersect_angle,
            ref_limits=self.ref_limits,
            pressure_angle=self.pressure_angle,
            cone_angle=self.cone_angle,
        )(0)

    def set_default_undercut_ref_point(self):
        self.undercut_ref_point = self.get_default_undercut_ref_point(self.ref_limits)
        return self


def generate_involute_rack_curve(
    pitch_radius: float,
    pitch_intersect_angle: float,
    ref_limits: ToothLimitParam,
    pressure_angle: float = PI / 9,
    cone_angle: float = 0,
) -> crv.Curve:
    """Generate trapezoid reference rack curve for involute teeth.

    Generates spherical rack curve for conical gears.
    Genereates only 1 flank of the rack curve.

    Returns
    -------
    crv.Curve
        The flank of the rack curve."""

    if cone_angle == 0:
        rp = pitch_radius
        pitch_len_ref = rp * pitch_intersect_angle
        p0 = RIGHT * rp + DOWN * pitch_len_ref
        direction = rotate_vector(RIGHT, pressure_angle)
        p1 = p0 + direction * ref_limits.h_a / direction[0]
        p2 = p0 - direction * ref_limits.h_d / direction[0]
        return crv.LineCurve(p0=p2, p1=p1)
    else:
        alpha = pressure_angle
        conic_transform = ConicData(cone_angle=cone_angle, base_radius=pitch_radius)

        def rack_flank_func(t, a):
            axis1 = scp_Rotation.from_euler("x", -alpha).apply(OUT)
            v0 = conic_transform.R * RIGHT
            an1 = t * np.cos(alpha)
            v1 = scp_Rotation.from_rotvec(-axis1 * an1).apply(v0)
            v2 = scp_Rotation.from_euler("z", t + a).apply(v1)
            return v2

        an_tooth_sph = conic_transform.r / conic_transform.R * pitch_intersect_angle
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


def generate_undercut_curve(
    pitch_radius: float,
    cone_angle: float,
    undercut_ref_point: np.ndarray,
) -> crv.Curve:
    """Generate undercut curve for involute teeth."""
    if cone_angle == 0:
        undercut_curve = crv.InvoluteCurve(
            r=pitch_radius,
            angle=0,
            v_offs=undercut_ref_point - RIGHT * pitch_radius,
            t0=0,
            t1=-1,
        )

    else:
        gamma = cone_angle / 2
        R = pitch_radius / np.sin(gamma)
        C_sph = 1 / R  # spherical curvature
        v_offs = scp_Rotation.from_euler("y", PI / 2 * np.sign(C_sph)).apply(
            undercut_ref_point - R * RIGHT
        )
        undercut_curve = crv.SphericalInvoluteCurve(
            r=pitch_radius,
            c_sphere=C_sph,
            v_offs=v_offs,
            t0=0,
            t1=-1,
        )

    sol1 = root(
        lambda t: np.dot(undercut_curve(t[0]), undercut_curve(t[0])) - pitch_radius**2,
        [0.5],
    )
    undercut_curve.set_end_on(sol1.x[0])
    return undercut_curve


def trim_involute_undercut(
    tooth_curve, undercut_curve, guess=(0.5, 1)
) -> crv.CurveChain:
    """Find the intersection and trim the tooth curve (involute curve)
    with undercut curve."""
    sol = crv.find_curve_intersect(tooth_curve, undercut_curve, guess=guess)
    # solcheck = np.linalg.norm(tooth_curve(sol.x[0]) - undercut_curve(sol.x[1]))

    tooth_curve.set_start_on(sol.x[0])
    undercut_curve.set_end_on(sol.x[1])
    return crv.CurveChain(undercut_curve, tooth_curve)


class CycloidTooth(GearToothConicGenerator):
    def __init__(
        self,
        pitch_intersect_angle: float = PI / 16,
        pitch_radius: float = 1.0,
        cone_angle: float = PI / 4,
        rc_in_coeff: float = 0.5,
        rc_out_coeff: float = 0.5,
    ):
        self.pitch_intersect_angle = pitch_intersect_angle
        self.pitch_radius = pitch_radius
        self.cone_angle = cone_angle
        self.rc_in_coeff = rc_in_coeff
        self.rc_out_coeff = rc_out_coeff

    def generate_tooth_curve(self) -> crv.CurveChain:
        return self.generate_cycloid_curve()

    def generate_cycloid_curve(self) -> crv.CurveChain:
        if self.cone_angle == 0:
            lower_curve = crv.CycloidCurve(
                rb=self.pitch_radius,
                rc=-self.rc_in_coeff * self.pitch_radius,
                angle=-self.pitch_intersect_angle,
                t0=-self.rc_in_coeff * PI / 2,
                t1=0,
            )
            upper_curve = crv.CycloidCurve(
                rb=self.pitch_radius,
                rc=self.rc_out_coeff * self.pitch_radius,
                angle=-self.pitch_intersect_angle,
                t1=self.rc_out_coeff * PI / 2,
                t0=0,
            )

        else:
            R = self.pitch_radius / np.sin(self.cone_angle / 2)
            lower_curve = crv.CycloidConicCurve(
                rb=self.pitch_radius,
                rc=-self.rc_in_coeff * self.pitch_radius,
                C=1 / R,
                angle=-self.pitch_intersect_angle,
                t0=-self.rc_in_coeff * PI / 2,
                t1=0,
            )
            upper_curve = crv.CycloidConicCurve(
                rb=self.pitch_radius,
                rc=self.rc_out_coeff * self.pitch_radius,
                C=1 / R,
                angle=-self.pitch_intersect_angle,
                t1=self.rc_out_coeff * PI / 2,
                t0=0,
            )
        sol = crv.find_curve_plane_intersect(upper_curve, plane_normal=UP, guess=1)
        upper_curve.set_end_on(sol.x[0])

        return crv.CurveChain(lower_curve, upper_curve)
