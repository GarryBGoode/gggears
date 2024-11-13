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

from gggears.gggears_build123d import *


# These examples are meant to showcase the functionality of the library,
# and serve as manual testing templates for the developer.


def spur_helical_gear():

    # param = InvoluteGearParamManager(
    #     z_vals=[0, 4],  # combined with the center function this defines gear height
    #     num_teeth=8,  # number of teeth
    #     module=2,  # module
    #     cone_angle=0,  # cone angle for bevel gears
    #     center=ORIGIN,
    #     enable_undercut=True,  # enable undercut - don't use with root fillet at the same time
    #     inside_teeth=False,  # inside teeth for ring gears
    # )
    # param.z_params.angle = 0  # progress angle, can be used for helical
    # param.z_params.h_d = 1.2  # dedendum height coefficient
    # param.z_params.h_a = 1.0  # addendum height coefficient
    # param.z_params.h_o = 2.5  # outer ring (or inner ring) coefficient
    # param.z_params.root_fillet = 0.0  # root fillet radius coefficient
    # param.z_params.tip_fillet = 0.0  # tip fillet radius coefficient
    # param.z_params.tip_reduction = 0.2  # tip reduction (truncation) coefficient
    # param.z_params.profile_reduction = 0  # profile reduction coefficient (for backlash)
    # param.z_params.profile_shift = 0.8  # profile shift coefficient  param.z_params.
    # param.z_params.fix_callables()

    # gear_spur = InvoluteGear(param)
    gear_spur = InvoluteGear2(
        z_vals=np.array([0, 1]), module=1, tooth_param=GearTeethData(8)
    )
    gear_cad = GearBuilder(
        gear=gear_spur,
        n_points_vert=2,  # nurb spline points in vertical direction
        n_points_hz=4,  # nurb spline points in horizontal direction (for involute)
        add_plug=True,  # without plug the gear will have a center hole that depends on h_o
        method="fast",  # switch for slightly faster or slower nurb generation method
    )

    # being lazy with the parameters here, but a deepcopy might be better
    # param.num_teeth = 21
    # param.z_params.profile_shift = lambda z: 0.0
    # making the angle a function of z will make the gear helical
    # proper calculation of helix angle is not implemented yet
    # param.z_params.angle = lambda z: z * 0.03
    # gear_helical = InvoluteGear(param)
    # gear_cad_helical = GearBuilder(
    #     gear=gear_helical,
    #     n_points_vert=3,  # spiral curves need at least 3 points
    #     n_points_hz=4,
    #     add_plug=False,
    #     method="fast",
    # )
    # # using build123d translate method to move them apart
    # return (
    #     gear_cad.solid.translate((-20, 0, 0)),
    #     gear_cad_helical.solid.translate((25, 0, 0)),
    # )

    return gear_cad.solid


def planetary_gear():
    m = 3
    # ring gear convention: geometry is same as spur gear but cut out from ring block
    # care must be taken with addendum and dedendum values, spur gear defaults may not work
    param_ring = InvoluteGearParamManager(
        z_vals=[0, 12],
        num_teeth=90,
        module=m,
        h_d=1.0,  # addendum coeff larger than dedendum for clearence
        h_a=1.2,
        h_o=2.5,  # this controls the outer ring diameter relative to pitch diameter
        # ring gears can use undercut geometry but it most likely causes interference
        # root fillet is preferred for ring gears
        root_fillet=0.2,
        enable_undercut=False,
        tip_reduction=0.2,
        inside_teeth=True,
    )

    param_sun = InvoluteGearParamManager(
        z_vals=[0, 12],
        num_teeth=12,
        module=m,
        h_d=1.2,
        h_a=1.0,
        h_o=2.5,
        root_fillet=0.0,
        tip_reduction=0.2,
        enable_undercut=True,
        inside_teeth=False,
    )

    param_planet = InvoluteGearParamManager(
        z_vals=[0, 12],
        num_teeth=39,
        module=m,
        h_d=1.2,
        h_a=1.0,
        h_o=2.5,
        root_fillet=0.0,
        tip_reduction=0.2,
        enable_undercut=True,
        inside_teeth=False,
    )

    gear_ring = InvoluteGear(param_ring)
    gear_sun = InvoluteGear(param_sun)
    gear_planet1 = InvoluteGear(param_planet)
    gear_planet2 = InvoluteGear(param_planet)
    gear_planet3 = InvoluteGear(param_planet)

    # using the mesh_to function to align planets with the sun
    gear_planet1.mesh_to(
        gear_sun, target_dir=RIGHT
    )  # planet is moved to the right of the sun
    gear_planet2.mesh_to(gear_sun, target_dir=rotate_vector(RIGHT, 2 * PI / 3))
    gear_planet3.mesh_to(gear_sun, target_dir=rotate_vector(RIGHT, -2 * PI / 3))

    gear_planet1_cad = GearBuilder(
        gear=gear_planet1, n_points_vert=2, n_points_hz=3, add_plug=True, method="fast"
    )
    gear_planet2_cad = GearBuilder(
        gear=gear_planet2, n_points_vert=2, n_points_hz=3, add_plug=True, method="fast"
    )
    gear_planet3_cad = GearBuilder(
        gear=gear_planet3, n_points_vert=2, n_points_hz=3, add_plug=True, method="fast"
    )

    gear_ring_cad = GearBuilder(
        gear=gear_ring, n_points_vert=2, n_points_hz=3, add_plug=False, method="fast"
    )
    gear_sun_cad = GearBuilder(
        gear=gear_sun, n_points_vert=2, n_points_hz=3, add_plug=True, method="fast"
    )

    return (
        gear_ring_cad.solid,
        gear_sun_cad.solid,
        gear_planet1_cad.solid,
        gear_planet2_cad.solid,
        gear_planet3_cad.solid,
    )


def bevel_gear():

    num_teeth_1 = 26
    num_teeth_2 = 13
    # module
    m = 4
    # half cone angle
    # this calculation ensures that bevels will generate a 90 degree axial angle
    gamma = np.arctan2(num_teeth_1, num_teeth_2)
    axis = OUT

    beta = 0.05

    param1 = InvoluteGearParamManager(
        z_vals=[0, 5],
        num_teeth=num_teeth_1,
        module=m,
        # this is the half cone angle
        cone_angle=gamma * 2,
        # note the abs function that corresponds to the breakpoint in z_vals
        h_d=1.2,
        h_a=1.0,
        h_o=2.5,
        root_fillet=0.2,
        tip_fillet=0.2,
        tip_reduction=0.2,
        profile_reduction=0,
        profile_shift=0.0,
        enable_undercut=False,
        inside_teeth=False,
    )
    param1.z_params.angle = lambda z: z * beta
    param1.z_params.cone_angle = lambda z: gamma * 2

    gamma2 = PI / 2 - gamma
    param2 = InvoluteGearParamManager(
        z_vals=[0, 5],
        num_teeth=num_teeth_2,
        module=m,
        cone_angle=gamma2 * 2,
        # note the abs function that corresponds to the breakpoint in z_vals
        h_d=1.2,
        h_a=1.0,
        h_o=2.5,
        root_fillet=0.0,
        tip_fillet=0.2,
        tip_reduction=0.2,
        profile_reduction=0,
        profile_shift=0.0,
        enable_undercut=True,
        inside_teeth=False,
    )
    param2.z_params.angle = lambda z: -z * beta * num_teeth_1 / num_teeth_2
    param2.z_params.cone_angle = lambda z: gamma2 * 2
    gear_1 = InvoluteGear(param1)
    gear_2 = InvoluteGear(param2)
    gear_2.mesh_to(gear_1, target_dir=normalize_vector(LEFT + UP * 0.1))

    gear_cad1 = GearBuilder(
        gear=gear_1, n_points_vert=3, n_points_hz=4, add_plug=False, method="fast"
    )
    gear_cad2 = GearBuilder(
        gear=gear_2, n_points_vert=3, n_points_hz=4, add_plug=False, method="fast"
    )
    return (gear_cad1.solid, gear_cad2.solid)


def fishbone_bevels():
    # it is a bit slow to build the gear so time measurements are thrown in here and there
    start = time.time()

    num_teeth = 9
    # module
    m = 4
    # half cone angle
    gamma = PI / 4

    param = InvoluteGearParamManager(
        # the middle value is used for enabling a breakpoint for the angle funciton
        z_vals=[0, 2, 4],
        num_teeth=num_teeth,
        module=m,
        # gamma is the half cone angle
        cone_angle=gamma * 2,
        h_d=1.4,
        h_a=1.35,
        h_o=2.5,
        root_fillet=0.0,
        tip_fillet=0.0,
        tip_reduction=0.2,
        # some backlash is generated by the profile reduction
        profile_reduction=0.01,
        profile_shift=0.0,
        # yes, bevel gears can have undercuts
        enable_undercut=True,
        inside_teeth=False,
    )
    # note the abs function that corresponds to the breakpoint in z_vals
    param.z_params.angle = lambda z: np.abs(z - param.z_vals[1]) * 0.2
    param.z_params.cone_angle = lambda z: gamma * 2

    gear_base = InvoluteGear(param)

    gear_cad = GearBuilder(
        gear=gear_base, n_points_vert=3, n_points_hz=3, add_plug=True, method="slow"
    )

    print(f"gear build time: {time.time()-start}")

    solid1 = Part(
        gear_cad.solid.translate(
            nppoint2Vector(-gear_cad.gear_generator_ref.center_sphere)
        )
    )
    solid1 = solid1 - Hole(radius=4, depth=50)
    solid2 = (
        solid1.rotate(Axis.Y, 90)
        .mirror(Plane.XY)
        .rotate(Axis.X, gear_cad.gear_generator_ref.pitch_angle * RAD2DEG * 0.5)
    )
    solid3 = solid2.rotate(Axis.Z, 120)
    solid4 = solid2.rotate(Axis.Z, -120)
    solid5 = solid1.rotate(Axis.Y, 180)
    # gears can be exported to step files
    # export_step(solid1,"fishbone_bevel_left.step")
    # solid1b = solid1.mirror(Plane.XZ)
    # export_step(solid1b,"fishbone_bevel_right.step")
    return (solid1, solid2, solid3, solid4, solid5)


if __name__ == "__main__":
    spur_helical_gear()
