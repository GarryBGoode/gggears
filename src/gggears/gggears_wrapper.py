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

import numpy as np
from gggears.gggears_core import *
from gggears.gggears_build123d import *
from build123d import Part
from gggears.gearteeth import *


class GearInfoMixin:
    """Mixin class for gear information properties."""

    def __init__(self, gear: Gear):
        self.gearcore = gear

    @property
    def number_of_teeth(self):
        """Number of teeth of the gear."""
        return self.gearcore.tooth_param.num_teeth

    @property
    def inside_teeth(self):
        """Indicator of an inside-ring gear."""
        return self.gearcore.tooth_param.inside_teeth

    @property
    def pitch_radius(self):
        """Nominal pitch radius of the gear."""
        return self.gearcore.tooth_param.num_teeth * self.gearcore.module / 2

    @property
    def rp(self):
        """Shorthand of pitch_radius."""
        return self.pitch_radius

    @property
    def module(self):
        """Module of the gear."""
        return self.gearcore.module

    @module.setter
    def module(self, value):
        self.gearcore.module = value

    @property
    def cone_data(self):
        """Info about the cone of the gear in a ConicData object."""
        cone_loc = ConicData(
            cone_angle=self.gearcore.cone.cone_angle, base_radius=self.rp
        )
        return cone_loc

    @property
    def cone_angle(self):
        """Cone angle of the gear."""
        return self.gearcore.cone.cone_angle

    @cone_angle.setter
    def cone_angle(self, value):
        self.gearcore.cone.cone_angle = value

    @property
    def center(self):
        """Nominal center (reference point) of the gear."""
        return self.gearcore.transform.center

    @center.setter
    def center(self, value):
        self.gearcore.transform.center = value

    @property
    def center_spherical(self):
        """Center of the gear-cone for bevel gears.

        Spherical refers to the spherical gear-boundary curves."""
        center_sph = self.cone_data.center / self.module
        return self.gearcore.transform(center_sph)

    def center_transform_at_z(self, z) -> GearTransform:
        """Returns the transform-data that represents the center point of a gear-slice
        at a given z-value."""
        tf1 = self.gearcore.shape_recipe(z).transform
        tf2 = self.gearcore.transform
        return tf2 * tf1

    def center_point_at_z(self, z):
        """Returns the center of a gear-slice at a given z-value."""
        return self.center_transform_at_z(z).center

    def center_location_at_z(self, z):
        """Returns the build123d Location object of the center of a gear-slice at a
        given z-value."""
        return transform2Location(self.center_transform_at_z(z))

    @property
    def center_location_bottom(self):
        """Build123d Location object of the reference center of the bottom of the gear.

        Note that for bevel gears this aligns with the pitch circle, and is not coplanar
        with the bottom face."""
        return self.center_location_at_z(self.gearcore.z_vals[0])

    @property
    def center_location_middle(self):
        """Build123d Location object of the reference center of the (axially) middle of
        the gear."""
        return self.center_location_at_z(
            0.5 * (self.gearcore.z_vals[0] + self.gearcore.z_vals[-1])
        )

    @property
    def center_location_top(self):
        """Build123d Location object of the reference center of the top of the gear.

        Note that for bevel gears this aligns with the pitch circle, and is not coplanar
        with the top face."""
        return self.center_location_at_z(self.gearcore.z_vals[-1])

    @property
    def face_location_top(self):
        """Build123d Location object of the top face of the gear."""
        point = self.radii_data_gen(self.gearcore.z_vals[-1]).r_o_curve.center
        loc = self.center_location_top
        loc.position = np2v(point)
        return loc

    @property
    def face_location_bottom(self):
        """Build123d Location object of the bottom face of the gear."""
        point = self.radii_data_gen(self.gearcore.z_vals[0]).r_o_curve.center
        loc = self.center_location_bottom
        loc.position = np2v(point)
        return loc

    @property
    def center_point_bottom(self):
        """Center point of the gear at the bottom as a numpy array."""
        return self.center_point_at_z(self.gearcore.z_vals[0])

    @property
    def center_point_middle(self):
        """Center point of the gear at the middle as a numpy array."""
        return self.center_point_at_z(
            0.5 * (self.gearcore.z_vals[0] + self.gearcore.z_vals[-1])
        )

    @property
    def center_point_top(self):
        """Center point of the gear at the top as a numpy array."""
        return self.center_point_at_z(self.gearcore.z_vals[-1])

    @property
    def pitch_angle(self):
        """Nominal pitch angle of the gear."""
        return self.gearcore.tooth_param.pitch_angle

    @property
    def angle(self):
        """Rotation progress of the gear."""
        return self.gearcore.transform.angle

    @property
    def z_height(self):
        """Height of the gear along its axis.

        Note that the height input parameter controls rather the width of the gearteeth
        which is different than the height for bevel gears."""
        # the height parameter is rather the width of the gear teeth,
        # which is the height for cylindrical gears but not for bevel gears
        # that is why it is reverse engineered from the shape recipe
        z0, z1 = self.gearcore.z_vals[1], self.gearcore.z_vals[0]
        c0, c1 = self.gearcore.shape_recipe.transform.center(
            z0
        ), self.gearcore.shape_recipe.transform.center(z1)
        return np.linalg.norm((self.gearcore.transform(c0 - c1)))

    def radii_data_gen(self, z) -> GearRefCircles:
        """Generates the reference circles of the gear at a given z-value."""
        profile = self.gearcore.curve_gen_at_z(z)
        trf = self.gearcore.transform * profile.transform

        r_a = crv.ArcCurve(
            radius=profile.ra_curve.radius, center=profile.ra_curve.center, angle=2 * PI
        ).transform(trf)
        r_d = crv.ArcCurve(
            radius=profile.rd_curve.radius, center=profile.rd_curve.center, angle=2 * PI
        ).transform(trf)
        r_o = crv.ArcCurve(
            radius=profile.ro_curve.radius, center=profile.ro_curve.center, angle=2 * PI
        ).transform(trf)
        r_p = crv.ArcCurve(
            radius=self.gearcore.tooth_param.num_teeth / 2, center=ORIGIN, angle=2 * PI
        ).transform(trf)

        return GearRefCircles(
            r_a_curve=r_a, r_d_curve=r_d, r_o_curve=r_o, r_p_curve=r_p
        )

    @property
    def radii_data_array(self):
        """Array of the reference circles of the gear at reference z-values."""
        return [self.radii_data_gen(z) for z in self.gearcore.z_vals]

    @property
    def radii_data_bottom(self):
        """Reference circles of the gear at the bottom."""
        return self.radii_data_gen(self.gearcore.z_vals[0])

    @property
    def radii_data_top(self):
        """Reference circles of the gear at the top."""
        return self.radii_data_gen(self.gearcore.z_vals[-1])

    def proportion_to_z(self, prop):
        return prop * self.gearcore.z_vals[1] + (1 - prop) * self.gearcore.z_vals[0]

    def p2z(self, prop):
        return self.proportion_to_z(prop)

    @property
    def addendum_radius(self):
        """Nominal addendum radius of the gear, after calculating all modifiers."""
        return self.radii_data_gen(self.gearcore.z_vals[0]).r_a_curve.radius

    @property
    def dedendum_radius(self):
        """Nominal dedendum radius of the gear, after calculating all modifiers."""
        return self.radii_data_gen(self.gearcore.z_vals[0]).r_d_curve.radius

    @property
    def max_outside_radius(self):
        """Maximum outside radius of the gear."""
        r_os = [elem.r_o_curve.radius for elem in self.radii_data_array]
        r_as = [elem.r_a_curve.radius for elem in self.radii_data_array]
        return np.max(r_os + r_as)

    def cone_angles_at_z(self, z):
        """Returns relevant cone angles for reference radii at a given z-value."""
        # this is convoluted due to preparing for hypoid gears,
        # where the cone angle is not constant
        radiidata = self.radii_data_gen(z)
        R0 = self.gearcore.shape_recipe(z).cone.R
        tf0 = self.gearcore.shape_recipe(z).transform
        tf1 = self.gearcore.transform
        R = R0 * tf0.scale * tf1.scale
        cone_angles = [
            np.arcsin(r / R)
            for r in [
                radiidata.r_o_curve.radius,
                radiidata.r_d_curve.radius,
                radiidata.r_p_curve.radius,
                radiidata.r_a_curve.radius,
            ]
        ]
        return cone_angles

    @property
    def cone_angle_limits_bottom(self):
        """Cone angles of reference radii at the bottom of the gear."""
        return self.cone_angles_at_z(self.gearcore.z_vals[0])

    @property
    def cone_angle_limits_top(self):
        """Cone angles of reference radii at the top of the gear."""
        return self.cone_angles_at_z(self.gearcore.z_vals[-1])


@dataclasses.dataclass
class InvoluteInputParam:
    number_of_teeth: int
    height: float = 1.0
    helix_angle: float = 0
    cone_angle: float = 0
    center: np.ndarray = dataclasses.field(default_factory=lambda: ORIGIN)
    angle: float = 0
    module: float = 1.0
    enable_undercut: bool = True
    root_fillet: float = 0.0
    tip_fillet: float = 0.0
    tip_truncation: float = 0.1
    profile_shift: float = 0.0
    addendum_coefficient: float = 1.0
    dedendum_coefficient: float = 1.2
    pressure_angle: float = 20 * PI / 180
    backlash: float = 0
    crowning: float = 0
    inside_teeth: bool = False
    z_anchor: float = 0


class InvoluteGear(GearInfoMixin):
    """Class for a generic involute gears.

    Parameters
    ----------
    number_of_teeth: int
        Number of teeth of the gear.
    height: float, optional
        Height of the gear. Default is 1.0.
    helix_angle: float
        Helix angle of the gear in radians. Default is 0.
    cone_angle: float
        Cone angle of the gear in radians. Default is 0.
    center: np.ndarray, optional
        Center reference-point of the gear. Default is ORIGIN.
    angle: float, optional
        Angle (rotation progress) of the gear. Default is 0.
    module: float, optional
        Module of the gear. Default is 1.0.
    enable_undercut: bool, optional
        Enables calcuation of undercut. Default is True.
        When True, root_fillet parameter is ignored.
        When False, root is connected with a straight line, or root fillet is applied.
    root_fillet: float, optional
        Root fillet radius coefficient. Default is 0.0.
    tip_fillet: float, optional
        Tip fillet radius coefficient. Default is 0.0.
    tip_truncation: float, optional
        Tip truncation coefficient. Default is 0.1. This parameter is used to truncate
        the tip of the gear, should it reach a sharp point due to profile shift or
        addendum parameter.
    profile_shift: float, optional
        Profile shift coefficient. Default is 0.0.
    addendum_coefficient: float, optional
        Addendum height coefficient. Default is 1.0.
    dedendum_coefficient: float, optional
        Dedendum height coefficient. Default is 1.2.
    pressure_angle: float, optional
        Pressure angle in radians. Default is 20 * PI / 180 (20deg in radians).
    backlash: float, optional
        Backlash coefficient. Default is 0.
    crowning: float, optional
        Crowning coefficient. Default is 0.
        Crowning reduces tooth width near the top and bottom face, resulting in a
        barrel-like tooth shape. It is used to reduce axial alignment errors.
        A value of 100 will result in a tooth flank displacement
        of 0.1 module, or a reduction of tooth width by 0.2 module near the top and
        bottom face, while tooth width remains nominal in the middle of the tooth.
    inside_teeth: bool, optional
        If True, inside-ring gear is created. Default is False.
    z_anchor: float, optional
        Determines where the reference zero-level of the gear should be placed relative
        to its height. The reference level contains the gear center and is related to
        placement of the gear with the mesh_to() function. Also the reference level is
        initially in the XY plane.
        Value 0 places reference at the bottom, value 0.5 in the middle, value 1 at the
        top of the gear. Default is 0.

    Methods
    -------
    mesh_to(other, target_dir=RIGHT)
        Aligns this gear to another gear object. The target_dir parameter specifies
        where this gear should be placed in relation to the other gear.
    build_part()
        Builds and returns a build123d Part object of the gear.
    update_part()
        Updates and returns the build123d Part object of the gear based on the current
        gear objects center position and angle.
        This method is useful if mesh_to() was called after build_part().


    """

    def __init__(
        self,
        number_of_teeth: int,
        height: float = 1.0,
        helix_angle: float = 0,
        cone_angle: float = 0,
        center: np.ndarray = ORIGIN,
        angle: float = 0,
        module: float = 1.0,
        enable_undercut=True,
        root_fillet: float = 0.0,
        tip_fillet: float = 0.0,
        tip_truncation: float = 0.1,
        profile_shift: float = 0.0,
        addendum_coefficient: float = 1.0,
        dedendum_coefficient: float = 1.2,
        pressure_angle: float = 20 * PI / 180,
        backlash: float = 0,
        crowning: float = 0,
        inside_teeth: bool = False,
        z_anchor: float = 0,
    ):
        self.inputparam = InvoluteInputParam(
            number_of_teeth=number_of_teeth,
            height=height,
            helix_angle=helix_angle,
            cone_angle=cone_angle,
            center=center,
            angle=angle,
            module=module,
            enable_undercut=enable_undercut,
            root_fillet=root_fillet,
            tip_fillet=tip_fillet,
            tip_truncation=tip_truncation,
            profile_shift=profile_shift,
            addendum_coefficient=addendum_coefficient,
            dedendum_coefficient=dedendum_coefficient,
            pressure_angle=pressure_angle,
            backlash=backlash,
            crowning=crowning,
            inside_teeth=inside_teeth,
            z_anchor=z_anchor,
        )
        if self.inputparam.number_of_teeth < 0:
            self.inputparam.number_of_teeth = -self.number_of_teeth
            self.inputparam.inside_teeth = not self.inputparam.inside_teeth

        self.builder: GearBuilder = None
        self.gearcore: Gear = None
        self.calc_params()

    @property
    def gamma(self):
        """Half of the cone angle."""
        return self.cone_angle / 2

    @property
    def beta(self):
        """Helix angle in radians."""
        return self.inputparam.helix_angle

    @property
    def helix_angle(self):
        """Helix angle in radians."""
        return self.inputparam.helix_angle

    @property
    def r_base(self):
        """Base radius of the gear."""
        circle_base = self.circle_involute_base(0)
        return circle_base.radius

    def circle_involute_base(self, z_ratio: float = 0) -> crv.ArcCurve:
        """Generates the base circle of the involute at a given z-ratio.

        Arguments
        ---------
        z_ratio: float, optional
            Ratio of the height of the gear where the slice should be taken.
            0 means at the bottom, 1 at the top. Default is 0.
        """
        z = z_ratio * self.gearcore.z_vals[1] + (1 - z_ratio) * self.gearcore.z_vals[0]
        r = self.gearcore.shape_recipe(z).tooth_generator.get_base_radius()
        trf = self.gearcore.transform * self.gearcore.shape_recipe(z).transform
        circle_ref = crv.ArcCurve(radius=r, center=ORIGIN, angle=2 * PI).transform(trf)
        return circle_ref

    def update_tooth_param(self):
        """Updates the tooth parameters for the gear. (pitch angle calculated here)"""
        return GearToothParam(
            self.inputparam.number_of_teeth, inside_teeth=self.inputparam.inside_teeth
        )

    def calc_params(self):
        """Sets up the internal construction recipe for the gear based on the
        parameters."""
        # reference pitch radius with module 1
        rp_ref = self.inputparam.number_of_teeth / 2
        pitch_angle = 2 * PI / self.inputparam.number_of_teeth
        gamma = self.inputparam.cone_angle / 2

        def crowning_func(z, offset=0):
            return (
                offset
                - (z * 2 / self.inputparam.height - 1) ** 2
                * self.inputparam.crowning
                / rp_ref
                * 1e-3
            )

        tooth_angle = (
            pitch_angle / 4
            + self.inputparam.profile_shift
            * np.tan(self.inputparam.pressure_angle)
            / rp_ref
            - self.inputparam.backlash / rp_ref
        )

        spiral_coeff = np.tan(self.beta) / rp_ref

        def angle_func(z, coeff=spiral_coeff):
            return z * coeff

        limits = ToothLimitParamRecipe(
            h_d=self.inputparam.dedendum_coefficient - self.inputparam.profile_shift,
            h_a=self.inputparam.addendum_coefficient + self.inputparam.profile_shift,
            h_o=-2 if self.inputparam.inside_teeth else 2,
        )

        if self.inputparam.enable_undercut:
            tooth_generator = InvoluteUndercutTooth(
                pressure_angle=self.inputparam.pressure_angle,
                pitch_radius=rp_ref,
                pitch_intersect_angle=lambda z: crowning_func(z, tooth_angle),
                ref_limits=limits,
                cone_angle=self.inputparam.cone_angle,
                pitch_angle=pitch_angle,
            )
        else:
            tooth_generator = InvoluteTooth(
                pressure_angle=self.inputparam.pressure_angle,
                pitch_radius=rp_ref,
                pitch_intersect_angle=lambda z: crowning_func(z, tooth_angle),
                cone_angle=self.inputparam.cone_angle,
            )
        z_h = self.inputparam.height / self.inputparam.module
        self.gearcore = Gear(
            tooth_param=self.update_tooth_param(),
            z_vals=np.array(
                [-z_h * self.inputparam.z_anchor, z_h * (1 - self.inputparam.z_anchor)]
            ),
            module=self.inputparam.module,
            cone=ConicData(cone_angle=self.inputparam.cone_angle),
            shape_recipe=GearProfileRecipe(
                tooth_generator=tooth_generator,
                limits=limits,
                fillet=FilletDataRecipe(
                    root_fillet=self.inputparam.root_fillet,
                    tip_fillet=self.inputparam.tip_fillet,
                    tip_reduction=self.inputparam.tip_truncation,
                ),
                cone=ConicData(
                    cone_angle=self.inputparam.cone_angle, base_radius=rp_ref
                ),
                pitch_angle=pitch_angle,
                transform=GearTransformRecipe(
                    scale=lambda z: (1 - z * 2 * np.sin(gamma) / self.number_of_teeth),
                    center=lambda z: 1 * z * OUT * np.cos(gamma),
                    angle=angle_func,
                ),
            ),
            transform=GearTransform(
                center=self.inputparam.center,
                angle=self.inputparam.angle,
                scale=self.inputparam.module,
            ),
        )

    def build_part(self) -> Part:
        """Creates the build123d Part object of the gear. This may take several seconds.

        Returns
        -------
        Part"""
        max_angle = np.max(
            self.gearcore.shape_recipe.transform.angle(
                np.linspace(self.gearcore.z_vals[0], self.gearcore.z_vals[1], 20)
            )
        )
        min_angle = np.min(
            self.gearcore.shape_recipe.transform.angle(
                np.linspace(self.gearcore.z_vals[0], self.gearcore.z_vals[1], 20)
            )
        )

        twist_angle = np.abs(max_angle - min_angle)

        if self.inputparam.crowning == 0 and self.beta == 0:
            n_vert = 2
        elif twist_angle > PI / 6:
            n_vert = 3 + int(twist_angle / (PI / 6))
        else:
            if self.cone_angle == 0:
                n_vert = 3
            else:
                n_vert = 4

        self.builder = GearBuilder(
            self.gearcore,
            n_points_hz=4,
            n_points_vert=n_vert,
            method="slow",
        )
        return self.builder.part_transformed

    def update_part(self):
        """Updates the build123d Part object accordingly if
        the current gear was moved or rotated"""
        if self.builder is None:
            self.build_part()
        self.builder.part_transformed = apply_transform_part(
            self.builder.solid, self.gearcore.transform
        )
        return self.builder.part_transformed

    def mesh_to(self, other: "InvoluteGear", target_dir: np.ndarray = RIGHT):
        """Aligns this gear to the other gear object.

        Arguments
        ---------
        other: InvoluteGear
            The other gear object to align to.
        target_dir: np.ndarray
            The direction in which the gear should be placed in relation to the other
            gear.
            Should be a unit vector. Default is RIGHT (x).
        """
        if self.inside_teeth:
            ps_mult_1 = -1
        else:
            ps_mult_1 = 1
        if other.inside_teeth:
            ps_mult_2 = -1
        else:
            ps_mult_2 = 1
        self.gearcore.mesh_to(
            other.gearcore,
            target_dir=target_dir,
            distance_offset=(
                self.inputparam.profile_shift * ps_mult_1 * self.module
                + other.inputparam.profile_shift * other.module * ps_mult_2
            ),
        )

    def build_boundary_wire(self, z_ratio: float = 0):
        """Generates a build123d Wire object of the gear-tooth slice at a given z-ratio.

        Arguments
        ----------
        z_ratio: float, optional
            Ratio of the height of the gear where the slice should be taken.
            0 means at the bottom, 1 at the top. Default is 0.

        Returns
        -------
        Wire
        """
        z = z_ratio * self.gearcore.z_vals[1] + (1 - z_ratio) * self.gearcore.z_vals[0]
        profile = self.gearcore.curve_gen_at_z(z)
        edges = generate_boundary_edges(profile, self.gearcore.transform)
        return bd.Wire(edges)

    def copy(self):
        return copy.deepcopy(self)


class SpurGear(InvoluteGear):
    """Class for a basic spur gear.

    Parameters
    ----------
    number_of_teeth: int
        Number of teeth of the gear.
    height: float, optional
        Height of the gear. Default is 1.0.
    center: np.ndarray, optional
        Center reference-point of the gear. Default is ORIGIN.
    angle: float, optional
        Angle (rotation progress) of the gear. Default is 0.
    module: float, optional
        Module of the gear. Default is 1.0.
    enable_undercut: bool, optional
        Enables calcuation of undercut. Default is True.
        When True, root_fillet parameter is ignored.
        When False, root is connected with a straight line, or root fillet is applied.
    root_fillet: float, optional
        Root fillet radius coefficient. Default is 0.0.
    tip_fillet: float, optional
        Tip fillet radius coefficient. Default is 0.0.
    tip_truncation: float, optional
        Tip truncation coefficient. Default is 0.1. This parameter is used to truncate
        the tip of the gear, should it reach a sharp point due to profile shift or
        addendum parameter.
    profile_shift: float, optional
        Profile shift coefficient. Default is 0.0.
    addendum_coefficient: float, optional
        Addendum height coefficient. Default is 1.0.
    dedendum_coefficient: float, optional
        Dedendum height coefficient. Default is 1.2.
    pressure_angle: float, optional
        Pressure angle in radians. Default is 20 degrees (converted to radians).
    backlash: float, optional
        Backlash coefficient. Default is 0.
    crowning: float, optional
        Crowning coefficient. Default is 0.
        Crowning reduces tooth width near the top and bottom face, resulting in a
        barrel-like tooth shape. It is used to reduce axial alignment errors.
        A value of 100 will result in a tooth flank displacement
        of 0.1 module, or a reduction of tooth width by 0.2 module near the top and
        bottom face, while tooth width remains nominal in the middle of the tooth.
    z_anchor: float, optional
        Determines where the reference zero-level of the gear should be placed relative
        to its height. The reference level contains the gear center and is related to
        placement of the gear with the mesh_to() function. Also the reference level is
        initially in the XY plane.
        Value 0 places reference at the bottom, value 0.5 in the middle, value 1 at the
        top of the gear. Default is 0.

    Methods
    -------
    mesh_to(other, target_dir=RIGHT)
        Aligns this gear to another gear object. The target_dir parameter specifies
        where this gear should be placed in relation to the other gear.
    build_part()
        Builds and returns a build123d Part object of the gear.
    update_part()
        Updates and returns the build123d Part object of the gear based on the current
        gear objects center position and angle.
        This method is useful if mesh_to() was called after build_part().

    Examples
    --------
    >>> gear1 = SpurGear(number_of_teeth=12)
    >>> gear2 = SpurGear(number_of_teeth=24)
    >>> gear1.mesh_to(gear2, target_dir=UP)
    >>> gear_part_1 = gear1.build_part()
    >>> gear_part_2 = gear2.build_part()
    >>> isinstance(gear_part_1, Part)
    True
    >>> isinstance(gear_part_2, Part)
    True

    """

    def __init__(
        self,
        number_of_teeth: int,
        height: float = 1.0,
        center: np.ndarray = ORIGIN,
        angle: float = 0,
        module: float = 1.0,
        enable_undercut=True,
        root_fillet: float = 0.0,
        tip_fillet: float = 0.0,
        tip_truncation: float = 0.1,
        profile_shift: float = 0.0,
        addendum_coefficient: float = 1.0,
        dedendum_coefficient: float = 1.2,
        pressure_angle: float = 20 * PI / 180,
        backlash: float = 0,
        crowning: float = 0,
        z_anchor: float = 0,
    ):
        super().__init__(
            number_of_teeth=number_of_teeth,
            height=height,
            helix_angle=0,
            cone_angle=0,
            center=center,
            angle=angle,
            module=module,
            enable_undercut=enable_undercut,
            root_fillet=root_fillet,
            tip_fillet=tip_fillet,
            tip_truncation=tip_truncation,
            profile_shift=profile_shift,
            addendum_coefficient=addendum_coefficient,
            dedendum_coefficient=dedendum_coefficient,
            pressure_angle=pressure_angle,
            backlash=backlash,
            crowning=crowning,
            inside_teeth=False,
            z_anchor=z_anchor,
        )


class SpurRingGear(InvoluteGear):
    """
    A class representing a spur ring gear, which is a type of gear with teeth on the
    inner circumference.

    Parameters
    ----------
    number_of_teeth: int
        Number of teeth of the gear.
    height: float
        Height of the gear. Default is 1.0.
    center: np.ndarray
        Center reference-point of the gear. Default is ORIGIN.
    angle: float
        Angle (rotation progress) of the gear. Default is 0.
    module: float
        Module of the gear. Default is 1.0.
    enable_undercut: bool
        Enables calculation of undercut. Default is False.
        Not recommended for ring gears.
    root_fillet: float
        Root fillet radius coefficient. Default is 0.2.
    tip_fillet: float
        Tip fillet radius coefficient. Default is 0.0.
    tip_truncation: float
        Tip truncation coefficient. Default is 0.
    profile_shift: float
        Profile shift coefficient. Default is 0.0.
    addendum_coefficient: float
        Addendum height coefficient. Default is 1.2.
    dedendum_coefficient: float
        Dedendum height coefficient. Default is 1.0.
    pressure_angle: float
        Pressure angle in radians. Default is 20 degrees (converted to radians).
    backlash: float
        Backlash coefficient. Default is 0.
    crowning: float
        Crowning coefficient. Default is 0.
    z_anchor: float, optional
        Determines where the reference zero-level of the gear should be placed relative
        to its height. The reference level contains the gear center and is related to
        placement of the gear with the mesh_to() function. Also the reference level is
        initially in the XY plane.
        Value 0 places reference at the bottom, value 0.5 in the middle, value 1 at the
        top of the gear. Default is 0.

    Notes
    -----
    Ring gear geometry is generated by subtracting a spur gear from a cylinder.
    The parameters and conventions are not inverted, e.g. increasing addendum
    coefficient will make deeper cuts in the ring. Only default values are updated
    to reflect the inversion.

    Methods
    -------
    update_tooth_param()
        Updates the tooth parameters for the gear.
    build_part()
        Builds the gear part using the specified parameters.
    mesh_to(other, target_dir=RIGHT)
        Aligns this gear to another gear object. The target_dir parameter specifies
        where this gear should be placed in relation to the other gear.

    Examples
    --------
    >>> gear1 = SpurGear(number_of_teeth=12)
    >>> gear2 = SpurRingGear(number_of_teeth=24)
    >>> gear1.mesh_to(gear2, target_dir=UP)
    >>> gear_part_1 = gear1.build_part()
    >>> gear_part_2 = gear2.build_part()
    >>> isinstance(gear_part_1, Part)
    True
    >>> isinstance(gear_part_2, Part)
    True
    """

    def __init__(
        self,
        number_of_teeth: int,
        height: float = 1.0,
        center: np.ndarray = ORIGIN,
        angle: float = 0,
        module: float = 1.0,
        enable_undercut=False,
        root_fillet: float = 0.2,
        tip_fillet: float = 0.0,
        tip_truncation: float = 0,
        profile_shift: float = 0.0,
        addendum_coefficient: float = 1.2,
        dedendum_coefficient: float = 1.0,
        outside_ring_coefficient: float = 2.5,
        pressure_angle: float = 20 * PI / 180,
        backlash: float = 0,
        crowning: float = 0,
        z_anchor: float = 0,
    ):
        super().__init__(
            number_of_teeth=number_of_teeth,
            height=height,
            helix_angle=0,
            cone_angle=0,
            center=center,
            angle=angle,
            module=module,
            enable_undercut=enable_undercut,
            root_fillet=root_fillet,
            tip_fillet=tip_fillet,
            tip_truncation=tip_truncation,
            profile_shift=profile_shift,
            addendum_coefficient=addendum_coefficient,
            dedendum_coefficient=dedendum_coefficient,
            pressure_angle=pressure_angle,
            backlash=backlash,
            crowning=crowning,
            inside_teeth=True,
            z_anchor=z_anchor,
        )
        self.gearcore.shape_recipe.limits.h_o = -outside_ring_coefficient


class HelicalGear(InvoluteGear):
    """Class for a helical gear.

    Parameters
    ----------
    number_of_teeth: int
        Number of teeth of the gear.
    helix_angle: float, optional
        Helix angle of the gear in radians. Default is 30 degrees (in radians).
        Two meshing helical gears should have opposite helix angles
        (one positive, one negative).
    herringbone: bool, optional
        If True, creates a double helical (herringbone) gear. Default is False.
    height: float, optional
        Height of the gear. Default is 1.0.
    center: np.ndarray, optional
        Center reference-point of the gear. Default is ORIGIN.
    angle: float, optional
        Angle (rotation progress) of the gear. Default is 0.
    module: float, optional
        Module of the gear. Default is 1.0.
    enable_undercut: bool, optional
        Enables calculation of undercut. Default is True.
        When True, root_fillet parameter is ignored.
        When False, root is connected with a straight line, or root fillet is applied.
    root_fillet: float, optional
        Root fillet radius coefficient. Default is 0.0.
    tip_fillet: float, optional
        Tip fillet radius coefficient. Default is 0.0.
    tip_truncation: float, optional
        Tip truncation coefficient. Default is 0.1. This parameter is used to truncate
        the tip of the gear, should it reach a sharp point due to profile shift or
        addendum parameter.
    profile_shift: float, optional
        Profile shift coefficient. Default is 0.0.
    addendum_coefficient: float, optional
        Addendum height coefficient. Default is 1.0.
    dedendum_coefficient: float, optional
        Dedendum height coefficient. Default is 1.2.
    pressure_angle: float, optional
        Pressure angle in radians. Default is 20 degrees (converted to radians).
    backlash: float, optional
        Backlash coefficient. Default is 0.
    crowning: float, optional
        Crowning coefficient. Default is 0.
    z_anchor: float, optional
        Determines where the reference zero-level of the gear should be placed relative
        to its height. The reference level contains the gear center and is related to
        placement of the gear with the mesh_to() function. Also the reference level is
        initially in the XY plane.
        Value 0 places reference at the bottom, value 0.5 in the middle, value 1 at the
        top of the gear. Default is 0.

    Notes
    -----
    Helical gear geometry is generated with normal system, meaning the pressure angle
    and pitch are calculated in the normal direction to the tooth surface. This allows
    HelicalGears to mesh with different helix angles, the mesh_to() function will adjust
    the shaft orientation. Use PI/4 (45 degrees) for both gears to produce a
    perpendicular axis combination. It is suggested to use z_anchor=0.5 for this kind of
    drive.

    Methods
    -------
    mesh_to(other, target_dir=RIGHT)
        Aligns this gear to another gear object. The target_dir parameter specifies
        where this gear should be placed in relation to the other gear.
    build_part()
        Builds and returns a build123d Part object of the gear.
    update_part()
        Updates and returns the build123d Part object of the gear based on the current
        gear objects center position and angle.
        This method is useful if mesh_to() was called after build_part().

    Examples
    --------
    >>> gear1 = HelicalGear(number_of_teeth=12, helix_angle=30 * PI / 180)
    >>> gear2 = HelicalGear(number_of_teeth=24, helix_angle=-30 * PI / 180)
    >>> gear1.mesh_to(gear2, target_dir=UP)
    >>> gear_part_1 = gear1.build_part()
    >>> gear_part_2 = gear2.build_part()
    >>> isinstance(gear_part_1, Part)
    True
    >>> isinstance(gear_part_2, Part)
    True

    """

    def __init__(
        self,
        number_of_teeth: int,
        helix_angle: float = 30 * PI / 180,
        herringbone=False,
        height: float = 1,
        center: np.ndarray = ORIGIN,
        angle: float = 0,
        module: float = 1,
        enable_undercut=True,
        root_fillet: float = 0,
        tip_fillet: float = 0,
        tip_truncation: float = 0.1,
        profile_shift: float = 0,
        addendum_coefficient: float = 1,
        dedendum_coefficient: float = 1.2,
        pressure_angle: float = 20 * PI / 180,
        backlash: float = 0,
        crowning: float = 0,
        z_anchor: float = 0,
    ):
        # self.helix_angle = helix_angle
        self.herringbone = herringbone
        beta = helix_angle
        super().__init__(
            number_of_teeth=number_of_teeth,
            height=height,
            helix_angle=helix_angle,
            cone_angle=0,
            center=center,
            angle=angle,
            module=module / np.cos(beta),
            enable_undercut=enable_undercut,
            root_fillet=root_fillet,
            tip_fillet=tip_fillet,
            tip_truncation=tip_truncation,
            profile_shift=profile_shift * np.cos(beta),
            addendum_coefficient=addendum_coefficient * np.cos(beta),
            dedendum_coefficient=dedendum_coefficient * np.cos(beta),
            pressure_angle=np.arctan(np.tan(pressure_angle) / np.cos(beta)),
            backlash=backlash,
            crowning=crowning,
            z_anchor=z_anchor,
        )

        # correct for herringbone design
        if herringbone:

            zmax = self.gearcore.z_vals[-1]
            zmin = self.gearcore.z_vals[0]
            zmid = (zmax + zmin) / 2

            self.gearcore.z_vals = np.insert(
                self.gearcore.z_vals,
                np.searchsorted(self.gearcore.z_vals, zmid),
                zmid,
            )

            def herringbone_mod(
                z,
                original: Callable = self.gearcore.shape_recipe.transform.angle,
                midpoint=zmid,
            ):
                return original(midpoint - np.abs(midpoint - z))

            self.gearcore.shape_recipe.transform.angle = herringbone_mod

        # correct for too much twist
        max_angle = np.max(
            self.gearcore.shape_recipe.transform.angle(
                np.linspace(self.gearcore.z_vals[0], self.gearcore.z_vals[1], 20)
            )
        )
        min_angle = np.min(
            self.gearcore.shape_recipe.transform.angle(
                np.linspace(self.gearcore.z_vals[0], self.gearcore.z_vals[1], 20)
            )
        )

        twist_angle = np.abs(max_angle - min_angle)
        if twist_angle > 2 * PI:
            z_add = int(twist_angle // (2 * PI) * (len(self.gearcore.z_vals) - 1))
            new_z_vals = np.linspace(
                self.gearcore.z_vals[0],
                self.gearcore.z_vals[-1],
                z_add + len(self.gearcore.z_vals),
            )
            new_z_vals = np.unique((np.concatenate((new_z_vals, self.gearcore.z_vals))))
            self.gearcore.z_vals = new_z_vals

    @property
    def beta(self):
        """Beta = helix angle of the gear."""
        return self.helix_angle

    def mesh_to(self, other, target_dir=RIGHT):
        # basic meshing
        super().mesh_to(other, target_dir)
        # orientation correction
        rot_axis = normalize_vector(other.gearcore.center - self.gearcore.center)
        angle_axis = self.beta + other.beta
        rot = scp_Rotation.from_rotvec(rot_axis * angle_axis)
        self.gearcore.transform.orientation = (
            rot.as_matrix() @ self.gearcore.transform.orientation
        )


class HelicalRingGear(InvoluteGear):
    """A class representing a helical ring gear, which is a type of gear with teeth on
    the inner circumference and a helical angle.

    Parameters
    ----------
    number_of_teeth: int
        Number of teeth of the gear.
    helix_angle: float
        Helix angle of the gear in radians. Default is 30 degrees (in radians).
        A helical gear and a helical ring gear should have the same helix angles.
    herringbone: bool, optional
        If True, creates a double helical (herringbone) gear. Default is False.
    height: float
        Height of the gear. Default is 1.0.
    center: np.ndarray
        Center reference-point of the gear. Default is ORIGIN.
    angle: float
        Angle (rotation progress) of the gear. Default is 0.
    module: float
        Module of the gear. Default is 1.0.
    enable_undercut: bool
        Enables calculation of undercut. Default is False.
        Not recommended for ring gears.
    root_fillet: float
        Root fillet radius coefficient. Default is 0.2.
    tip_fillet: float
        Tip fillet radius coefficient. Default is 0.0.
    tip_truncation: float
        Tip truncation coefficient. Default is 0.1.
    profile_shift: float
        Profile shift coefficient. Default is 0.0.
    addendum_coefficient: float
        Addendum height coefficient. Default is 1.2.
    dedendum_coefficient: float
        Dedendum height coefficient. Default is 1.0.
    pressure_angle: float
        Pressure angle in radians. Default is 20 degrees (converted to radians).
    backlash: float
        Backlash coefficient. Default is 0.
    crowning: float
        Crowning coefficient. Default is 0.
    z_anchor: float, optional
        Determines where the reference zero-level of the gear should be placed relative
        to its height. The reference level contains the gear center and is related to
        placement of the gear with the mesh_to() function. Also the reference level is
        initially in the XY plane.
        Value 0 places reference at the bottom, value 0.5 in the middle, value 1 at the
        top of the gear. Default is 0.

    Notes
    -----
    Ring gear geometry is generated by subtracting a helical gear from a cylinder.
    The parameters and conventions are not inverted, e.g. increasing addendum
    coefficient will make deeper cuts in the ring. Only default values are updated
    to reflect the inversion.
    HelicalRingGear does not support orientation adjustment with mesh_to() function if
    helix angles are different.

    Methods
    -------
    update_tooth_param()
        Updates the tooth parameters for the gear.
    build_part()
        Builds the gear part using the specified parameters.
    mesh_to(other, target_dir=RIGHT)
        Aligns this gear to another gear object. The target_dir parameter specifies
        where this gear should be placed in relation to the other gear.

    Examples
    --------
    >>> gear1 = HelicalGear(number_of_teeth=12, helix_angle=30 * PI / 180)
    >>> gear2 = HelicalRingGear(number_of_teeth=24, helix_angle=30 * PI / 180)
    >>> gear1.mesh_to(gear2, target_dir=UP)
    >>> gear_part_1 = gear1.build_part()
    >>> gear_part_2 = gear2.build_part()
    >>> isinstance(gear_part_1, Part)
    True
    >>> isinstance(gear_part_2, Part)
    True
    """

    def __init__(
        self,
        number_of_teeth: int,
        helix_angle: float = 30 * PI / 180,
        herringbone=False,
        height: float = 1,
        center: np.ndarray = ORIGIN,
        angle: float = 0,
        module: float = 1,
        enable_undercut=False,
        root_fillet: float = 0,
        tip_fillet: float = 0,
        tip_truncation: float = 0.1,
        profile_shift: float = 0,
        addendum_coefficient: float = 1.2,
        dedendum_coefficient: float = 1.0,
        pressure_angle: float = 20 * PI / 180,
        backlash: float = 0,
        crowning: float = 0,
        z_anchor: float = 0,
    ):
        beta = helix_angle
        super().__init__(
            number_of_teeth=number_of_teeth,
            helix_angle=beta,
            height=height,
            center=center,
            angle=angle,
            module=module / np.cos(beta),
            enable_undercut=enable_undercut,
            root_fillet=root_fillet,
            tip_fillet=tip_fillet,
            tip_truncation=tip_truncation,
            profile_shift=profile_shift * np.cos(beta),
            addendum_coefficient=addendum_coefficient * np.cos(beta),
            dedendum_coefficient=dedendum_coefficient * np.cos(beta),
            pressure_angle=np.arctan(np.tan(pressure_angle) / np.cos(beta)),
            backlash=backlash,
            crowning=crowning,
            inside_teeth=True,
            z_anchor=z_anchor,
        )

        # correct for herringbone design
        if herringbone:

            zmax = self.gearcore.z_vals[-1]
            zmin = self.gearcore.z_vals[0]
            zmid = (zmax + zmin) / 2

            self.gearcore.z_vals = np.insert(
                self.gearcore.z_vals,
                np.searchsorted(self.gearcore.z_vals, zmid),
                zmid,
            )

            def herringbone_mod(
                z,
                original: Callable = self.gearcore.shape_recipe.transform.angle,
                midpoint=zmid,
            ):
                return original(midpoint - np.abs(midpoint - z))

            self.gearcore.shape_recipe.transform.angle = herringbone_mod

    def update_tooth_param(self):
        """:no-index:"""
        return GearToothParam(self.inputparam.number_of_teeth, inside_teeth=True)

    def build_part(self):
        """:no-index:"""
        max_zval = np.max(self.gearcore.z_vals[1:] - self.gearcore.z_vals[:-1])
        twist_angle = np.abs(self.gearcore.shape_recipe.transform.angle(max_zval))
        if twist_angle < PI / 16:
            n_vert = 3
            method = "fast"
        elif twist_angle < PI / 2:
            n_vert = 4
            method = "slow"
        else:
            # 5 points tend to crash the OCT fuse with the slow option
            n_vert = 5
            method = "fast"
        self.builder = GearBuilder(
            self.gearcore,
            n_points_hz=4,
            n_points_vert=n_vert,
            method=method,
        )
        return self.builder.part_transformed


class BevelGear(InvoluteGear):
    """
    A class representing a bevel gear, which is a type of gear with teeth on a conical
    surface.

    Parameters
    ----------
    number_of_teeth: int
        Number of teeth of the gear.
    cone_angle: float
        Cone angle of the gear in radians. Default is PI / 2.
    helix_angle: float
        Spiral coefficient or angle for the gear. Default is 0.
        This parameter can be used to create a spiral bevel gears, but it is not
        dimensionally accurate.
        Two meshing spiral bevel gears should have opposite spiral coefficients.
    height: float
        Height parameter of the gear. Default is 1.0.
        Height is not the dimension along the z axis, but rather the length of
        the tooth surface. This is to ensure different cone angles but same height
        parameters result in equal tooth surface heights.
    center: np.ndarray
        Center reference-point of the gear. Default is ORIGIN.
    angle: float
        Angle (rotation progress) of the gear. Default is 0.
    module: float
        Module of the gear. Default is 1.0.
    enable_undercut: bool
        Enables calculation of undercut. Default is True.
    root_fillet: float
        Root fillet radius coefficient. Default is 0.0.
    tip_fillet: float
        Tip fillet radius coefficient. Default is 0.0.
    tip_truncation: float
        Tip truncation coefficient. Default is 0.1.
    profile_shift: float
        Profile shift coefficient. Default is 0.0..
        Underlying method is not exact, but approximates the behavior of spur gears
        with profile shift.
    addendum_coefficient: float
        Addendum height coefficient. Default is 1.0.
    dedendum_coefficient: float
        Dedendum height coefficient. Default is 1.2.
    pressure_angle: float
        Pressure angle in radians. Default is 20 degrees (converted to radians).
    backlash: float
        Backlash coefficient. Default is 0.
    crowning: float
        Crowning coefficient. Default is 0.
    z_anchor: float, optional
        Determines where the reference zero-level of the gear should be placed relative
        to its height. The reference level contains the gear center and is related to
        placement of the gear with the mesh_to() function. Also the reference level is
        initially in the XY plane.
        Value 0 places reference at the bottom, value 0.5 in the middle, value 1 at the
        top of the gear. Default is 0.

    Notes
    -----
    Bevel gears are generated with spherical involute profiles. The end-faces of teeth
    are (approximate) spherical surface elements.

    Undercut is calculated similarly to spur gears, based on the trochoid of the
    spherical involute.

    Height is not the dimension along the z axis, but rather the length of the tooth.
    This is to ensure different cone angles but same height parameters result in equal
    tooth surface heights.

    The size of the gear is determined by the module, cone angle, and number of teeth.
    The pitch cone's radius on the XY plane is the pitch radius, and it is calculated
    from the module and number of teeth as usual. r_p = module * number_of_teeth / 2

    By default the gear is positioned such that the pitch circle is in the XY plane.
    This forces some protion of the gear under the XY plane.

    Profile shift is not recommended for bevel gears, the meshing function doesn't
    support it yet. Use complementary values (eg. +0.3 and -0.3) of shift on the pair
    of bevels if you really need it.

    Methods
    -------
    mesh_to(other, target_dir=RIGHT)
        Aligns this gear to another gear object. The target_dir parameter specifies
        where this gear should be placed in relation to the other gear.
    build_part()
        Builds and returns a build123d Part object of the gear.
    update_part()
        Updates and returns the build123d Part object of the gear based on the current
        gear objects center position and angle.
        This method is useful if mesh_to() was called after build_part().

    Examples
    --------
    >>> num_teeth_1 = 16
    >>> num_teeth_2 = 31
    >>> beta = 0.5
    >>> gamma = np.arctan2(num_teeth_1, num_teeth_2)
    >>> gamma2 = np.pi / 2 - gamma
    >>> height = 5
    >>> m = 2
    >>> gear1 = BevelGear(number_of_teeth=num_teeth_1,module=m,height=height,cone_angle=gamma * 2, helix_angle=beta)
    >>> gear2 = BevelGear(number_of_teeth=num_teeth_2,module=m,height=height,cone_angle=gamma2 * 2,helix_angle=-beta)
    >>> gear1.mesh_to(gear2, target_dir=UP)
    >>> gear_part_1 = gear1.build_part()
    >>> gear_part_2 = gear2.build_part()
    >>> isinstance(gear_part_1, Part)
    True
    >>> isinstance(gear_part_2, Part)
    True
    """

    def __init__(
        self,
        number_of_teeth: int,
        cone_angle: float = PI / 2,
        helix_angle: float = 0,
        height: float = 1.0,
        center: np.ndarray = ORIGIN,
        angle: float = 0,
        module: float = 1.0,
        enable_undercut=True,
        root_fillet: float = 0.0,
        tip_fillet: float = 0.0,
        tip_truncation: float = 0.1,
        profile_shift: float = 0,
        addendum_coefficient: float = 1.0,
        dedendum_coefficient: float = 1.2,
        pressure_angle: float = 20 * PI / 180,
        backlash: float = 0,
        crowning: float = 0,
        z_anchor: float = 0,
    ):
        # Note: this bevel is not much different from the generic involute gear
        super().__init__(
            number_of_teeth=number_of_teeth,
            height=height,
            helix_angle=helix_angle,
            cone_angle=cone_angle,
            center=center,
            angle=angle,
            module=module,
            enable_undercut=enable_undercut,
            root_fillet=root_fillet,
            tip_fillet=tip_fillet,
            tip_truncation=tip_truncation,
            profile_shift=profile_shift,
            addendum_coefficient=addendum_coefficient,
            dedendum_coefficient=dedendum_coefficient,
            pressure_angle=pressure_angle,
            backlash=backlash,
            crowning=crowning,
            inside_teeth=False,
            z_anchor=z_anchor,
        )


@dataclasses.dataclass
class CycloidInputParam:
    number_of_teeth: int
    height: float = 1.0
    helix_angle: float = 0
    cone_angle: float = 0
    center: np.ndarray = dataclasses.field(default_factory=lambda: ORIGIN)
    angle: float = 0
    module: float = 1.0
    enable_undercut = True
    root_fillet: float = 0.0
    tip_fillet: float = 0.0
    tip_truncation: float = 0.1
    profile_shift: float = 0.0
    addendum_coefficient: float = 1.0
    dedendum_coefficient: float = 1.2
    inside_cycloid_coefficient: float = 0.5
    outside_cycloid_coefficient: float = 0.5
    backlash: float = 0
    crowning: float = 0
    inside_teeth: bool = False
    z_anchor: float = 0


class CycloidGear(GearInfoMixin):
    """Class for a cycloid gear.

    Parameters
    ----------
    number_of_teeth: int
        Number of teeth of the gear.
    height: float, optional
        Height of the gear. Default is 1.0.
    cone_angle: float, optional
        Cone angle of the gear in radians. Default is 0.
    center: np.ndarray, optional
        Center reference-point of the gear. Default is ORIGIN.
    angle: float, optional
        Angle (rotation progress) of the gear. Default is 0.
    module: float, optional
        Module of the gear. Default is 1.0.
    root_fillet: float, optional
        Root fillet radius coefficient. Default is 0.0.
    tip_fillet: float, optional
        Tip fillet radius coefficient. Default is 0.0.
    tip_truncation: float, optional
        Tip truncation coefficient. Default is 0.1. This parameter is used to truncate
        the tip of the gear, should it reach a sharp point due to profile shift or
        addendum parameter.
    addendum_coefficient: float, optional
        Addendum height coefficient. Default is 1.0.
    dedendum_coefficient: float, optional
        Dedendum height coefficient. Default is 1.2.
    inside_cycloid_coefficient: float, optional
        Ratio of inside rolling circle vs pitch circle radius. Default is 0.5.
    outside_cycloid_coefficient: float, optional
        Ratio of outside rolling circle vs pitch circle radius. Default is 0.5.
    helix_angle: float
        Helix angle of the gear in radians. Default is 0.
    backlash: float, optional
        Backlash coefficient. Default is 0.
    crowning: float, optional
        Crowning coefficient. Default is 0.
        Crowning reduces tooth width near the top and bottom face, resulting in a
        barrel-like tooth shape. It is used to reduce axial alignment errors.
        A value of 100 will result in a tooth flank displacement
        of 0.1 module, or a reduction of tooth width by 0.2 module near the top and
        bottom face, while tooth width remains nominal in the middle of the tooth.
    inside_teeth: bool, optional
        When true, creates a ring-gear. Default is False.

    Methods
    -------
    mesh_to(other, target_dir=RIGHT)
        Aligns this gear to another gear object. The target_dir parameter specifies
        where this gear should be placed in relation to the other gear.
    build_part()
        Builds and returns a build123d Part object of the gear.
    update_part()
        Updates and returns the build123d Part object of the gear based on the current
        gear objects center position and angle.
        This method is useful if mesh_to() was called after build_part().
    adapt_cycloid_radii(other)
        Adapts the cycloid generator radii for both this and other gear to ensure
        proper meshing. It is done by adapting the 'outside' radii while keeping the
        'inside' radii unchanged.

    Examples
    --------
    >>> gear1 = CycloidGear(number_of_teeth=12)
    >>> gear2 = CycloidGear(number_of_teeth=24)
    >>> gear1.mesh_to(gear2, target_dir=UP)
    >>> gear1.adapt_cycloid_radii(gear2)
    >>> gear_part_1 = gear1.build_part()
    >>> gear_part_2 = gear2.build_part()
    >>> isinstance(gear_part_1, Part)
    True
    >>> isinstance(gear_part_2, Part)
    True

    """

    def __init__(
        self,
        number_of_teeth: int,
        height: float = 1.0,
        cone_angle=0,
        center: np.ndarray = ORIGIN,
        angle: float = 0,
        module: float = 1.0,
        root_fillet: float = 0.0,
        tip_fillet: float = 0.0,
        tip_truncation: float = 0.1,
        addendum_coefficient: float = 1.0,
        dedendum_coefficient: float = 1.2,
        inside_cycloid_coefficient: float = 0.5,
        outside_cycloid_coefficient: float = 0.5,
        helix_angle: float = 0,
        backlash: float = 0,
        crowning: float = 0,
        inside_teeth=False,
        z_anchor: float = 0,
    ):
        self.inputparam = CycloidInputParam(
            number_of_teeth=number_of_teeth,
            height=height,
            cone_angle=cone_angle,
            center=center,
            angle=angle,
            module=module,
            root_fillet=root_fillet,
            tip_fillet=tip_fillet,
            tip_truncation=tip_truncation,
            addendum_coefficient=addendum_coefficient,
            dedendum_coefficient=dedendum_coefficient,
            inside_cycloid_coefficient=inside_cycloid_coefficient,
            outside_cycloid_coefficient=outside_cycloid_coefficient,
            helix_angle=helix_angle,
            backlash=backlash,
            crowning=crowning,
            inside_teeth=inside_teeth,
            z_anchor=z_anchor,
        )
        self.builder: GearBuilder = None
        self.gearcore: Gear = None
        self.calc_params()

    @property
    def inside_cycloid_coefficient(self):
        return self.inputparam.inside_cycloid_coefficient

    @inside_cycloid_coefficient.setter
    def inside_cycloid_coefficient(self, value):
        self.inputparam.inside_cycloid_coefficient = value

    @property
    def outside_cycloid_coefficient(self):
        return self.inputparam.outside_cycloid_coefficient

    @outside_cycloid_coefficient.setter
    def outside_cycloid_coefficient(self, value):
        self.inputparam.outside_cycloid_coefficient = value

    def update_tooth_param(self):
        """Updates the tooth parameters for the gear. (pitch angle calculated here)"""
        return GearToothParam(
            self.inputparam.number_of_teeth, inside_teeth=self.inputparam.inside_teeth
        )

    def calc_params(self):
        """Sets up the internal construction recipe for the gear based on the
        parameters."""
        # reference pitch radius with module 1
        rp_ref = self.inputparam.number_of_teeth / 2
        gamma = self.inputparam.cone_angle / 2
        pitch_angle = 2 * PI / self.inputparam.number_of_teeth

        def crowning_func(z, offset=0):
            return (
                offset
                - (z * 2 / self.inputparam.height - 1) ** 2
                * self.inputparam.crowning
                / rp_ref
                * 1e-3
            )

        tooth_angle = pitch_angle / 4 - self.inputparam.backlash / rp_ref
        spiral_coeff = np.tan(self.inputparam.helix_angle) / rp_ref

        def angle_func(z, coeff=spiral_coeff):
            return z * coeff

        if not self.inputparam.inside_teeth:
            limits = ToothLimitParamRecipe(
                h_d=self.inputparam.dedendum_coefficient,
                h_a=self.inputparam.addendum_coefficient,
            )
        else:
            limits = ToothLimitParamRecipe(
                h_d=self.inputparam.dedendum_coefficient,
                h_a=self.inputparam.addendum_coefficient,
                h_o=-2,
            )

        tooth_generator = CycloidTooth(
            pitch_radius=rp_ref,
            pitch_intersect_angle=lambda z: crowning_func(z, tooth_angle),
            cone_angle=gamma * 2,
            rc_in_coeff=self.inputparam.inside_cycloid_coefficient,
            rc_out_coeff=self.inputparam.outside_cycloid_coefficient,
        )
        z_h = self.inputparam.height / self.inputparam.module
        self.gearcore = Gear(
            tooth_param=self.update_tooth_param(),
            z_vals=np.array(
                [-z_h * self.inputparam.z_anchor, z_h * (1 - self.inputparam.z_anchor)]
            ),
            module=self.inputparam.module,
            cone=ConicData(cone_angle=self.inputparam.cone_angle),
            shape_recipe=GearProfileRecipe(
                tooth_generator=tooth_generator,
                limits=limits,
                fillet=FilletDataRecipe(
                    root_fillet=self.inputparam.root_fillet,
                    tip_fillet=self.inputparam.tip_fillet,
                    tip_reduction=self.inputparam.tip_truncation,
                ),
                cone=ConicData(
                    cone_angle=self.inputparam.cone_angle, base_radius=rp_ref
                ),
                pitch_angle=pitch_angle,
                transform=GearTransformRecipe(
                    scale=lambda z: 1
                    * (1 - z * 2 * np.sin(gamma) / self.inputparam.number_of_teeth),
                    center=lambda z: 1 * z * OUT * np.cos(gamma),
                    angle=angle_func,
                ),
            ),
            transform=GearTransform(
                center=self.inputparam.center,
                angle=self.inputparam.angle,
                scale=self.inputparam.module,
            ),
        )

    def build_part(self) -> Part:
        """Creates the build123d Part object of the gear. This may take several seconds.

        Returns
        -------
        Part"""

        max_angle = np.max(
            self.gearcore.shape_recipe.transform.angle(
                np.linspace(self.gearcore.z_vals[0], self.gearcore.z_vals[1], 20)
            )
        )
        min_angle = np.min(
            self.gearcore.shape_recipe.transform.angle(
                np.linspace(self.gearcore.z_vals[0], self.gearcore.z_vals[1], 20)
            )
        )

        twist_angle = np.abs(max_angle - min_angle)

        if (
            self.inputparam.helix_angle == 0
            and self.inputparam.cone_angle == 0
            and self.inputparam.crowning == 0
        ):
            n_vert = 2
        elif twist_angle > PI / 4:
            n_vert = 3 + int(twist_angle / (PI / 4))
        else:
            n_vert = 3
        self.builder = GearBuilder(
            self.gearcore,
            n_points_hz=4,
            n_points_vert=n_vert,
            method="slow",
        )
        return self.builder.part_transformed

    def update_part(self):
        """Updates the build123d Part object accordingly if
        the current gear was moved or rotated"""
        if self.builder is None:
            self.build_part()
        self.builder.part_transformed = apply_transform_part(
            self.builder.solid, self.gearcore.transform
        )
        return self.builder.part_transformed

    def mesh_to(self, other: "CycloidGear", target_dir: np.ndarray = RIGHT):
        """Aligns this gear to another gear object.

        Arguments
        ---------
        other: CycloidGear
            The other gear object to align to.
        target_dir: np.ndarray
            The direction in which the gear should be placed in relation to the other
            gear.
            Should be a unit vector. Default is RIGHT (x).
        """
        self.gearcore.mesh_to(
            other.gearcore,
            target_dir=target_dir,
            distance_offset=0,
        )

    def adapt_cycloid_radii(self, other: "CycloidGear"):
        """Adapts the radii of the 2 gears to enable meshing. The inside radii will
        remain the same while the outside radii are adjusted.

        The role of inside-outside is reversed for ring gears.

        Arguments
        ---------
        other: CycloidGear
            The other gear object to adapt to.
        """
        if self.inside_teeth:
            self.inside_cycloid_coefficient = (
                other.inside_cycloid_coefficient
                / self.number_of_teeth
                * other.number_of_teeth
            )

            other.outside_cycloid_coefficient = (
                self.outside_cycloid_coefficient
                / other.number_of_teeth
                * self.number_of_teeth
            )
        elif other.inside_teeth:

            self.outside_cycloid_coefficient = (
                other.outside_cycloid_coefficient
                / self.number_of_teeth
                * other.number_of_teeth
            )

            other.inside_cycloid_coefficient = (
                self.inside_cycloid_coefficient
                / other.number_of_teeth
                * self.number_of_teeth
            )

        else:
            self.outside_cycloid_coefficient = (
                other.inside_cycloid_coefficient
                / self.number_of_teeth
                * other.number_of_teeth
            )

            other.outside_cycloid_coefficient = (
                self.inside_cycloid_coefficient
                / other.number_of_teeth
                * self.number_of_teeth
            )
        self.calc_params()
        other.calc_params()

    def build_boundary_wire(self, z_ratio: float = 0):
        """Generates a build123d Wire object of the gear-tooth slice at a given z-ratio.

        Arguments
        ----------
        z_ratio: float, optional
            Ratio of the height of the gear where the slice should be taken.
            0 means at the bottom, 1 at the top. Default is 0.

        Returns
        -------
        Wire
        """
        z = z_ratio * self.gearcore.z_vals[1] + (1 - z_ratio) * self.gearcore.z_vals[0]
        profile = self.gearcore.curve_gen_at_z(z)
        edges = generate_boundary_edges(profile, self.gearcore.transform)
        return bd.Wire(edges)

    def copy(self):
        """:no-index:"""
        return copy.deepcopy(self)


class LineOfAction:
    """Class for generating the line of action of two meshing gears.

    Parameters
    ----------
    gear1: InvoluteGear
        The first gear object.
    gear2: InvoluteGear
        The second gear object.
    z_ratio: float, optional
        Ratio of the height of the gears where the line of action should be calculated.
        0 means at the bottom, 1 at the top. Default is 0.
    """

    def __init__(self, gear1: InvoluteGear, gear2: InvoluteGear, z_ratio=0):
        self.gear1: InvoluteGear = gear1
        self.gear2: InvoluteGear = gear2
        self.z_ratio = z_ratio

    @property
    def centerdiff(self):
        return self.gear2.center_point_at_z(
            self.gear2.p2z(self.z_ratio)
        ) - self.gear1.center_point_at_z(self.gear1.p2z(self.z_ratio))

    def LOA_gen(self):
        diff_vector = self.centerdiff
        distance = np.linalg.norm(diff_vector)
        diff_vector_unit = diff_vector / distance

        rb1 = self.gear1.circle_involute_base(self.z_ratio).radius
        rb2 = self.gear2.circle_involute_base(self.z_ratio).radius
        center1 = self.gear1.center_point_at_z(self.gear1.p2z(self.z_ratio))
        center2 = self.gear2.center_point_at_z(self.gear2.p2z(self.z_ratio))

        def mirror_diffline(p):
            p1 = p - center1
            p_norm = p1 - np.dot(p1, diff_vector_unit) * diff_vector_unit
            return p - 2 * p_norm

        if self.gear1.inside_teeth:
            if np.abs(rb2 - rb1) / distance > 1:
                raise ValueError(
                    "Unable to calculate line of action for this gear pair."
                )
            alpha = np.arccos((rb2 - rb1) / distance)
            len_loa = np.sqrt(self.gear2.addendum_radius**2 - rb2**2)
            startpoint1 = center2 + rotate_vector(-diff_vector_unit, alpha) * rb2
            endpoint1 = (
                startpoint1 + rotate_vector(diff_vector_unit, -PI / 2 + alpha) * len_loa
            )
            line1 = crv.LineCurve(p0=startpoint1, p1=endpoint1)
            startpoint2 = mirror_diffline(startpoint1)
            endpoint2 = mirror_diffline(endpoint1)
            line2 = crv.LineCurve(p0=startpoint2, p1=endpoint2)
            return line1, line2
        elif self.gear2.inside_teeth:
            if np.abs(rb2 - rb1) / distance > 1:
                raise ValueError(
                    "Unable to calculate line of action for this gear pair."
                )
            alpha = np.arccos((rb2 - rb1) / distance)
            len_loa = np.sqrt(self.gear1.addendum_radius**2 - rb1**2)
            startpoint1 = center1 + rotate_vector(-diff_vector_unit, alpha) * rb1
            endpoint1 = (
                startpoint1 - rotate_vector(diff_vector_unit, -PI / 2 + alpha) * len_loa
            )
            line1 = crv.LineCurve(p0=startpoint1, p1=endpoint1)
            startpoint2 = mirror_diffline(startpoint1)
            endpoint2 = mirror_diffline(endpoint1)
            line2 = crv.LineCurve(p0=startpoint2, p1=endpoint2)
            return line1, line2
        else:
            if np.abs(rb2 + rb1) / distance > 1:
                raise ValueError(
                    "Unable to calculate line of action for this gear pair."
                )
            alpha = np.arccos((rb1 + rb2) / distance)
            line1 = crv.LineCurve(
                p0=center1 + rotate_vector(diff_vector_unit, alpha) * rb1,
                p1=center2 + rotate_vector(-diff_vector_unit, alpha) * rb2,
            )
            line2 = crv.LineCurve(
                p0=center1 + rotate_vector(diff_vector_unit, -alpha) * rb1,
                p1=center2 + rotate_vector(-diff_vector_unit, -alpha) * rb2,
            )
            return line1, line2
