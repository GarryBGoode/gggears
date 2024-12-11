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


class InvoluteGear:
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
    inside_teeht: bool, optional
        If True, inside-ring gear is created. Default is False.

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
    ):
        self.number_of_teeth = number_of_teeth
        self.height = height
        self.helix_angle = helix_angle
        self.cone_angle = cone_angle
        self.center = center
        self.angle = angle
        self.module = module
        self.enable_undercut = enable_undercut
        self.root_fillet = root_fillet
        self.tip_fillet = tip_fillet
        self.tip_truncation = tip_truncation
        self.profile_shift = profile_shift
        self.addendum_coefficient = addendum_coefficient
        self.dedendum_coefficient = dedendum_coefficient
        self.pressure_angle = pressure_angle
        self.backlash = backlash
        self.crowning = crowning
        self.inside_teeth = inside_teeth

        if self.number_of_teeth < 0:
            self.number_of_teeth = -self.number_of_teeth
            self.inside_teeth = not self.inside_teeth

        self.builder: GearBuilder = None
        self.gearcore: Gear = None
        self.calc_params()

    @property
    def rp(self):
        """Pitch radius of the gear at z=0."""
        return self.module * self.number_of_teeth / 2

    @property
    def gamma(self):
        return self.cone_angle / 2

    @property
    def beta(self):
        return self.helix_angle

    @property
    def pitch_angle(self):
        """Pitch angle of the gear."""
        return 2 * PI / self.number_of_teeth

    def update_tooth_param(self):
        """Updates the tooth parameters for the gear. (pitch angle calculated here)"""
        return GearToothParam(self.number_of_teeth, inside_teeth=self.inside_teeth)

    def calc_params(self):
        """Sets up the internal construction recipe for the gear based on the
        parameters."""
        # reference pitch radius with module 1
        rp_ref = self.rp / self.module

        def crowning_func(z, offset=0):
            return (
                offset - (z * 2 / self.height - 1) ** 2 * self.crowning / rp_ref * 1e-3
            )

        tooth_angle = (
            self.pitch_angle / 4
            + self.profile_shift * np.tan(self.pressure_angle) / rp_ref
            - self.backlash / rp_ref
        )

        spiral_coeff = np.sin(self.beta) / rp_ref

        def angle_func(z, coeff=spiral_coeff):
            return z * coeff

        limits = ToothLimitParamRecipe(
            h_d=self.dedendum_coefficient - self.profile_shift,
            h_a=self.addendum_coefficient + self.profile_shift,
            h_o=-2 if self.inside_teeth else 2,
        )

        if self.enable_undercut:
            tooth_generator = InvoluteUndercutTooth(
                pressure_angle=self.pressure_angle,
                pitch_radius=rp_ref,
                pitch_intersect_angle=lambda z: crowning_func(z, tooth_angle),
                ref_limits=limits,
                cone_angle=self.cone_angle,
            )
        else:
            tooth_generator = InvoluteTooth(
                pressure_angle=self.pressure_angle,
                pitch_radius=rp_ref,
                pitch_intersect_angle=lambda z: crowning_func(z, tooth_angle),
                cone_angle=self.cone_angle,
            )

        self.gearcore = Gear(
            tooth_param=self.update_tooth_param(),
            z_vals=np.array([0, self.height / self.module]),
            module=self.module,
            cone=ConicData(cone_angle=self.cone_angle),
            shape_recipe=GearProfileRecipe(
                tooth_generator=tooth_generator,
                limits=limits,
                fillet=FilletDataRecipe(
                    root_fillet=self.root_fillet,
                    tip_fillet=self.tip_fillet,
                    tip_reduction=self.tip_truncation,
                ),
                cone=ConicData(cone_angle=self.cone_angle, base_radius=rp_ref),
                pitch_angle=self.pitch_angle,
                transform=GearTransformRecipe(
                    scale=lambda z: (
                        1 - z * 2 * np.sin(self.gamma) / self.number_of_teeth
                    ),
                    center=lambda z: 1 * z * OUT * np.cos(self.gamma),
                    angle=angle_func,
                ),
            ),
            transform=GearTransform(
                center=self.center, angle=self.angle, scale=self.module
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
        if self.crowning == 0 and self.beta == 0:
            n_vert = 2
        elif twist_angle > PI / 4:
            n_vert = 3 + int(twist_angle / (PI / 4))
        else:
            if self.cone_angle == 0:
                n_vert = 3
            else:
                n_vert = 4

        self.builder = GearBuilder(
            self.gearcore,
            n_points_hz=4,
            n_points_vert=n_vert,
            add_plug=not self.inside_teeth,
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
        """Aligns this gear to another gear object.

        Arguments
        ---------
        other: InvoluteGear
            The other gear object to align to.
        target_dir: np.ndarray
            The direction in which the gear should be placed in relation to the other gear.
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
                self.profile_shift * ps_mult_1 * self.module
                + other.profile_shift * other.module * ps_mult_2
            ),
        )

    def copy(self):
        """:no-index:"""
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
        pressure_angle: float = 20 * PI / 180,
        backlash: float = 0,
        crowning: float = 0,
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
        )


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
    ):
        self.helix_angle = helix_angle
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
            profile_shift=profile_shift,
            addendum_coefficient=addendum_coefficient,
            dedendum_coefficient=dedendum_coefficient,
            pressure_angle=np.arctan(np.tan(pressure_angle) / np.cos(beta)),
            backlash=backlash,
            crowning=crowning,
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

    @property
    def beta(self):
        """Beta = helix angle of the gear."""
        return self.helix_angle


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

    Notes
    -----
    Ring gear geometry is generated by subtracting a helical gear from a cylinder.
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
    ):
        super().__init__(
            number_of_teeth=number_of_teeth,
            helix_angle=helix_angle,
            height=height,
            center=center,
            angle=angle,
            module=module / np.cos(helix_angle),
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
        )
        # correct for herringbone design
        if herringbone:
            self.gearcore.z_vals = np.array([0, self.height / 2, self.height])

            def herringbone_mod(
                z, original: Callable = self.gearcore.shape_recipe.transform.angle
            ):
                return original(self.height / 2 - np.abs(self.height / 2 - z))

            self.gearcore.shape_recipe.transform.angle = herringbone_mod

    def update_tooth_param(self):
        """:no-index:"""
        return GearToothParam(self.number_of_teeth, inside_teeth=True)

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
            add_plug=False,
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
        )


class CycloidGear:
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
    ):
        self.number_of_teeth = number_of_teeth
        self.height = height
        self.cone_angle = cone_angle
        self.center = center
        self.angle = angle
        self.module = module
        self.root_fillet = root_fillet
        self.tip_fillet = tip_fillet
        self.tip_truncation = tip_truncation
        self.addendum_coefficient = addendum_coefficient
        self.dedendum_coefficient = dedendum_coefficient
        self.inside_cycloid_coefficient = inside_cycloid_coefficient
        self.outside_cycloid_coefficient = outside_cycloid_coefficient
        self.helix_angle = helix_angle
        self.backlash = backlash
        self.crowning = crowning
        self.builder: GearBuilder = None
        self.gearcore: Gear = None
        self.inside_teeth = inside_teeth
        self.calc_params()

    @property
    def rp(self):
        """Pitch radius of the gear at z=0."""
        return self.module * self.number_of_teeth / 2

    @property
    def pitch_angle(self):
        """Pitch angle of the gear."""
        return 2 * PI / self.number_of_teeth

    def update_tooth_param(self):
        """Updates the tooth parameters for the gear. (pitch angle calculated here)"""
        return GearToothParam(self.number_of_teeth, inside_teeth=self.inside_teeth)

    def calc_params(self):
        """Sets up the internal construction recipe for the gear based on the
        parameters."""
        # reference pitch radius with module 1
        rp_ref = self.rp / self.module
        gamma = self.cone_angle / 2

        def crowning_func(z, offset=0):
            return (
                offset - (z * 2 / self.height - 1) ** 2 * self.crowning / rp_ref * 1e-3
            )

        tooth_angle = self.pitch_angle / 4 - self.backlash / rp_ref
        spiral_coeff = np.sin(self.helix_angle) / rp_ref

        def angle_func(z, coeff=spiral_coeff):
            return z * coeff

        if not self.inside_teeth:
            limits = ToothLimitParamRecipe(
                h_d=self.dedendum_coefficient, h_a=self.addendum_coefficient
            )
        else:
            limits = ToothLimitParamRecipe(
                h_d=self.dedendum_coefficient, h_a=self.addendum_coefficient, h_o=-2
            )

        tooth_generator = CycloidTooth(
            pitch_radius=rp_ref,
            pitch_intersect_angle=lambda z: crowning_func(z, tooth_angle),
            cone_angle=gamma * 2,
            rc_in_coeff=self.inside_cycloid_coefficient,
            rc_out_coeff=self.outside_cycloid_coefficient,
        )

        self.gearcore = Gear(
            tooth_param=self.update_tooth_param(),
            z_vals=np.array([0, self.height / self.module]),
            module=self.module,
            cone=ConicData(cone_angle=self.cone_angle),
            shape_recipe=GearProfileRecipe(
                tooth_generator=tooth_generator,
                limits=limits,
                fillet=FilletDataRecipe(
                    root_fillet=self.root_fillet,
                    tip_fillet=self.tip_fillet,
                    tip_reduction=self.tip_truncation,
                ),
                cone=ConicData(cone_angle=self.cone_angle, base_radius=rp_ref),
                pitch_angle=self.pitch_angle,
                transform=GearTransformRecipe(
                    scale=lambda z: 1
                    * (1 - z * 2 * np.sin(gamma) / self.number_of_teeth),
                    center=lambda z: 1 * z * OUT * np.cos(gamma),
                    angle=angle_func,
                ),
            ),
            transform=GearTransform(
                center=self.center, angle=self.angle, scale=self.module
            ),
        )

    def build_part(self) -> Part:
        """Creates the build123d Part object of the gear. This may take several seconds.

        Returns
        -------
        Part"""
        if self.helix_angle == 0 and self.cone_angle == 0 and self.crowning == 0:
            n_vert = 2
        else:
            n_vert = 3
        if self.inside_teeth:
            plug = False
        else:
            plug = True
        self.builder = GearBuilder(
            self.gearcore,
            n_points_hz=4,
            n_points_vert=n_vert,
            add_plug=plug,
            method="fast",
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
            The direction in which the gear should be placed in relation to the other gear.
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

    def copy(self):
        """:no-index:"""
        return copy.deepcopy(self)
