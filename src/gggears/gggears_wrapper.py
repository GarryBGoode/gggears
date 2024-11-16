import numpy as np
from gggears.gggears_core import *
from gggears.gggears_build123d import *


class SpurGear:
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
        self.number_of_teeth = number_of_teeth
        self.height = height
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
        self.builder: GearBuilder = None
        self.gearcore: InvoluteGear = None
        self.calc_params()

    @property
    def rp(self):
        return self.module * self.number_of_teeth / 2

    @property
    def pitch_angle(self):
        return 2 * PI / self.number_of_teeth

    def update_tooth_param(self):
        return GearToothParam(self.number_of_teeth)

    def calc_params(self):

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

        self.gearcore = InvoluteGear(
            tooth_param=self.update_tooth_param(),
            z_vals=np.array([0, self.height]),
            module=self.module,
            cone=ConicData(cone_angle=0),
            shape_recipe=InvoluteToothRecipe(
                involute=InvoluteProfileParamRecipe(
                    pressure_angle=self.pressure_angle,
                    pitch_radius=rp_ref,
                    angle_pitch_ref=lambda z: crowning_func(z, tooth_angle),
                ),
                limits=ToothLimitParamRecipe(
                    h_d=self.dedendum_coefficient - self.profile_shift,
                    h_a=self.addendum_coefficient + self.profile_shift,
                ),
                fillet=FilletDataRecipe(
                    root_fillet=self.root_fillet,
                    tip_fillet=self.tip_fillet,
                    tip_reduction=self.tip_truncation,
                ),
                cone=ConicData(cone_angle=0, base_radius=self.rp),
                pitch_angle=self.pitch_angle,
                transform=GearTransformRecipe(
                    scale=1.0,
                    center=lambda z: OUT * z,
                ),
            ),
            transform=GearTransform(
                center=self.center, angle=self.angle, scale=self.module
            ),
        )

    def build_part(self):
        if self.crowning == 0:
            n_vert = 2
        else:
            n_vert = 3
        self.builder = GearBuilder(
            self.gearcore,
            n_points_hz=4,
            n_points_vert=n_vert,
            add_plug=True,
            method="fast",
        )
        return self.builder.solid_transformed

    def update_part(self):
        if self.builder is None:
            self.build_part()
        self.builder.solid_transformed = apply_transform_part(
            self.builder.solid, self.gearcore.transform
        )
        return self.builder.solid_transformed

    def mesh_to(self, other: "SpurGear", target_dir: np.ndarray = RIGHT):
        if (
            self.gearcore.tooth_param.inside_teeth
            or other.gearcore.tooth_param.inside_teeth
        ):
            ps_mult = -1
        else:
            ps_mult = 1
        self.gearcore.mesh_to(
            other.gearcore,
            target_dir=target_dir,
            distance_offset=ps_mult
            * (self.profile_shift * self.module + other.profile_shift * other.module),
        )


class SpurRingGear(SpurGear):
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
            number_of_teeth,
            height,
            center,
            angle,
            module,
            enable_undercut,
            root_fillet,
            tip_fillet,
            tip_truncation,
            profile_shift,
            addendum_coefficient,
            dedendum_coefficient,
            pressure_angle,
            backlash,
            crowning,
        )
        if self.gearcore.shape_recipe.limits.h_o > 0:
            self.gearcore.shape_recipe.limits.h_o *= -1

    def update_tooth_param(self):
        return GearToothParam(self.number_of_teeth, inside_teeth=True)

    def build_part(self):
        if self.crowning == 0:
            n_vert = 2
        else:
            n_vert = 3
        self.builder = GearBuilder(
            self.gearcore,
            n_points_hz=4,
            n_points_vert=n_vert,
            add_plug=False,
            method="fast",
        )
        return self.builder.solid_transformed


class HelicalGear(SpurGear):
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
            number_of_teeth,
            height,
            center,
            angle,
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
        angle_coeff = np.sin(self.beta) / self.gearcore.rp
        if herringbone:
            self.gearcore.shape_recipe.transform.angle = (
                lambda z, coeff=angle_coeff: (
                    -np.abs(self.height / 2 - z) + self.height / 2
                )
                * coeff
            )
            self.gearcore.z_vals = np.array([0, self.height / 2, self.height])
        else:
            self.gearcore.shape_recipe.transform.angle = (
                lambda z, coeff=angle_coeff: z * coeff
            )
        self.gearcore.shape_recipe.transform.center = (
            lambda z, coeff=np.cos(beta): OUT * z * coeff
        )

    @property
    def beta(self):
        return self.helix_angle

    def build_part(self):
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
            add_plug=True,
            method=method,
        )
        return self.builder.solid_transformed


class HelicalRingGear(HelicalGear):
    """
    A class representing a helical ring gear, which is a type of gear with teeth on the
    inner circumference and a helical angle.

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
            number_of_teeth,
            helix_angle,
            herringbone,
            height,
            center,
            angle,
            module,
            enable_undercut,
            root_fillet,
            tip_fillet,
            tip_truncation,
            profile_shift,
            addendum_coefficient,
            dedendum_coefficient,
            pressure_angle,
            backlash,
            crowning,
        )
        if self.gearcore.shape_recipe.limits.h_o > 0:
            self.gearcore.shape_recipe.limits.h_o *= -1

    def update_tooth_param(self):
        return GearToothParam(self.number_of_teeth, inside_teeth=True)

    def build_part(self):
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
        return self.builder.solid_transformed


class BevelGear(SpurGear):
    """
    A class representing a bevel gear, which is a type of gear with teeth on a conical
    surface.

    Parameters
    ----------
    number_of_teeth: int
        Number of teeth of the gear.
    cone_angle: float
        Cone angle of the gear in radians. Default is PI / 2.
    spiral_coefficient: float
        Spiral coefficient for the gear. Default is 0.
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

    Profile shift is not implemented for bevel gears. A bevel-gear equivalent of profile
    shift would cause a change in the axial angle, which is often required to be 90 degrees.
    Reverse-optimizing around this constraint would be tedious.
    If undercut is causing issues, changing the pressure angle is recommended instead.

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
    >>> n1 = 12
    >>> n2 = 24
    >>> gamma = np.arctan2(n1, n2)
    >>> gamma2 = np.pi/2-gamma
    >>> gear1 = BevelGear(number_of_teeth=12, cone_angle=gamma, spiral_coefficient=0.5)
    >>> gear2 = BevelGear(number_of_teeth=24, cone_angle=gamma2, spiral_coefficient=-0.5)
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
        spiral_coefficient: float = 0,
        height: float = 1.0,
        center: np.ndarray = ORIGIN,
        angle: float = 0,
        module: float = 1.0,
        enable_undercut=True,
        root_fillet: float = 0.0,
        tip_fillet: float = 0.0,
        tip_truncation: float = 0.1,
        addendum_coefficient: float = 1.0,
        dedendum_coefficient: float = 1.2,
        pressure_angle: float = 20 * PI / 180,
        backlash: float = 0,
        crowning: float = 0,
    ):
        self.cone_angle = cone_angle
        self.spiral_coefficient = spiral_coefficient
        super().__init__(
            number_of_teeth=number_of_teeth,
            height=height,
            center=center,
            angle=angle,
            module=module,
            enable_undercut=enable_undercut,
            root_fillet=root_fillet,
            tip_fillet=tip_fillet,
            tip_truncation=tip_truncation,
            addendum_coefficient=addendum_coefficient,
            dedendum_coefficient=dedendum_coefficient,
            pressure_angle=pressure_angle,
            backlash=backlash,
            crowning=crowning,
            profile_shift=0,
        )
        # self.gearcore.shape_recipe.cone = ConicData(cone_angle=PI / 4)

    def calc_params(self):

        # reference pitch radius with module 1
        rp_ref = self.rp / self.module
        gamma = self.cone_angle / 2

        def crowning_func(z, offset=0):
            return (
                offset - (z * 2 / self.height - 1) ** 2 * self.crowning / rp_ref * 1e-3
            )

        tooth_angle = self.pitch_angle / 4 - self.backlash / rp_ref

        self.gearcore = InvoluteGear(
            tooth_param=self.update_tooth_param(),
            z_vals=np.array([0, self.height]),
            module=self.module,
            cone=ConicData(cone_angle=self.cone_angle),
            shape_recipe=InvoluteToothRecipe(
                involute=InvoluteProfileParamRecipe(
                    pressure_angle=self.pressure_angle,
                    pitch_radius=rp_ref,
                    angle_pitch_ref=lambda z: crowning_func(z, tooth_angle),
                ),
                limits=ToothLimitParamRecipe(
                    h_d=self.dedendum_coefficient - self.profile_shift,
                    h_a=self.addendum_coefficient + self.profile_shift,
                ),
                fillet=FilletDataRecipe(
                    root_fillet=self.root_fillet,
                    tip_fillet=self.tip_fillet,
                    tip_reduction=self.tip_truncation,
                ),
                cone=ConicData(cone_angle=self.cone_angle, base_radius=self.rp),
                pitch_angle=self.pitch_angle,
                transform=GearTransformRecipe(
                    scale=lambda z: 1
                    * (1 - z * 2 * np.sin(gamma) / self.number_of_teeth),
                    center=lambda z: 1 * z * OUT * np.cos(gamma),
                    angle=lambda z: self.spiral_coefficient / rp_ref * z,
                ),
            ),
            transform=GearTransform(
                center=self.center, angle=self.angle, scale=self.module
            ),
        )

    def update_tooth_param(self):
        return GearToothParam(self.number_of_teeth, inside_teeth=False)

    def build_part(self):
        n_vert = 3
        self.builder = GearBuilder(
            self.gearcore,
            n_points_hz=4,
            n_points_vert=n_vert,
            add_plug=True,
            method="fast",
        )
        return self.builder.solid_transformed
