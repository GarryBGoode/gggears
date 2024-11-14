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
from ocp_vscode import show, set_port
import time

# These examples are meant to showcase the functionality of the library,
# and serve as manual testing templates for the developer.


def spur_helical_gear():

    gear_spur = InvoluteGear(
        z_vals=np.array([0, 4]),
        module=2,
        tooth_param=GearToothParam(num_teeth=8),
        enable_undercut=True,
    )

    gear_spur.shape_recipe.transform.angle = (
        0  # progress angle, can be used for helical
    )
    gear_spur.shape_recipe.limits.h_d = 1.2  # dedendum height coefficient
    gear_spur.shape_recipe.limits.h_a = 1.0  # addendum height coefficient
    gear_spur.shape_recipe.limits.h_o = 2.5  # outer ring (or inner ring) coefficient
    gear_spur.shape_recipe.fillet.root_fillet = 0.0  # root fillet radius coefficient
    gear_spur.shape_recipe.fillet.tip_fillet = 0.0  # tip fillet radius coefficient
    # tip reduction (truncation) coefficient
    gear_spur.shape_recipe.fillet.tip_reduction = 0.2

    gear_cad = GearBuilder(
        gear=gear_spur,
        n_points_vert=2,  # nurb spline points in vertical direction
        n_points_hz=4,  # nurb spline points in horizontal direction (for involute)
        add_plug=True,  # without plug the gear will have a center hole that depends on h_o
        method="fast",  # switch for slightly faster or slower nurb generation method
    )

    gear_helical = gear_spur.copy()
    gear_helical.tooth_param.num_teeth = 21
    gear_helical.shape_recipe = default_gear_recipe(
        teeth_data=gear_helical.tooth_param, module=2, cone_angle=0
    )
    # making the angle a function of z will make the gear helical
    # proper calculation of helix angle is not implemented yet
    gear_helical.shape_recipe.transform.angle = lambda z: 0.1 * z

    # being lazy with the parameters here, but a deepcopy might be better

    gear_cad_helical = GearBuilder(
        gear=gear_helical,
        n_points_vert=3,  # spiral curves need at least 3 points
        n_points_hz=4,
        add_plug=False,
        method="fast",
    )
    # using build123d translate method to move them apart
    return (
        gear_cad.solid.translate((-20, 0, 0)),
        gear_cad_helical.solid.translate((25, 0, 0)),
    )


def planetary_gear():
    m = 3
    # ring gear convention: geometry is same as spur gear but cut out from ring block
    # care must be taken with addendum and dedendum values, spur gear defaults may not work
    gear_ring = InvoluteGear(
        z_vals=np.array([0, 12]),
        module=m,
        tooth_param=GearToothParam(num_teeth=90, inside_teeth=True),
        enable_undercut=False,
    )

    gear_ring.shape_recipe.transform.angle = (
        0  # progress angle, can be used for helical
    )
    gear_ring.shape_recipe.limits.h_d = 1.0  # dedendum height coefficient
    gear_ring.shape_recipe.limits.h_a = 1.2  # addendum height coefficient
    gear_ring.shape_recipe.limits.h_o = -2.5  # outer ring (or inner ring) coefficient
    gear_ring.shape_recipe.fillet.root_fillet = 0.3  # root fillet radius coefficient
    gear_ring.shape_recipe.fillet.tip_fillet = 0.0  # tip fillet radius coefficient

    gear_sun = InvoluteGear(
        z_vals=np.array([0, 12]),
        module=m,
        tooth_param=GearToothParam(num_teeth=12),
        enable_undercut=True,
    )

    gear_planet1 = InvoluteGear(
        z_vals=np.array([0, 12]),
        module=m,
        tooth_param=GearToothParam(num_teeth=39),
        enable_undercut=True,
    )

    gear_planet2 = gear_planet1.copy()
    gear_planet3 = gear_planet1.copy()

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
        gear_ring_cad.solid_transformed,
        gear_sun_cad.solid_transformed,
        gear_planet1_cad.solid_transformed,
        gear_planet2_cad.solid_transformed,
        gear_planet3_cad.solid_transformed,
    )


def bevel_gear():

    num_teeth_1 = 26
    num_teeth_2 = 9
    beta = 0.05
    # module
    m = 4
    # half cone angle
    # this calculation ensures that bevels will generate a 90 degree axial angle
    gamma = np.arctan2(num_teeth_1, num_teeth_2)

    gear_1 = InvoluteGear(
        z_vals=[0, 5],
        tooth_param=GearToothParam(num_teeth=num_teeth_1),
        cone=ConicData(cone_angle=gamma * 2),
        module=m,
        enable_undercut=True,
    )
    gear_1.shape_recipe.transform.angle = lambda z: z * beta

    gamma2 = PI / 2 - gamma
    gear_2 = InvoluteGear(
        z_vals=[0, 5],
        tooth_param=GearToothParam(num_teeth=num_teeth_2),
        cone=ConicData(cone_angle=gamma2 * 2),
        module=m,
        enable_undercut=True,
    )
    gear_2.shape_recipe.transform.angle = (
        lambda z: -z * beta * num_teeth_1 / num_teeth_2
    )

    gear_2.mesh_to(gear_1, target_dir=rotate_vector(RIGHT, 3 * PI / 4))

    gear_cad1 = GearBuilder(
        gear=gear_1, n_points_vert=3, n_points_hz=4, add_plug=False, method="fast"
    )
    gear_cad2 = GearBuilder(
        gear=gear_2, n_points_vert=3, n_points_hz=4, add_plug=False, method="fast"
    )
    return (gear_cad1.solid_transformed, gear_cad2.solid_transformed)


def fishbone_bevels():
    # This example was meant to stress the library a bit, and to generate
    # interlocking bevel gears. In theory it should be possible to design them in a way
    # that they form a 'ball' that mechanically locks together.
    #
    # It is a bit slow to build the gear so time measurements are thrown in here
    start = time.time()

    num_teeth = 9
    # module
    m = 4
    # half cone angle
    gamma = PI / 4
    beta = 0.65

    gear_base = InvoluteGear(
        z_vals=[0, 2, 4],
        tooth_param=GearToothParam(num_teeth=num_teeth),
        cone=ConicData(cone_angle=gamma * 2),
        module=m,
        enable_undercut=True,
    )
    gear_base.shape_recipe.involute.pressure_angle = 35 * PI / 180
    gear_base.shape_recipe.limits.h_a = 1.0
    gear_base.shape_recipe.limits.h_d = 1.1
    gear_base.shape_recipe.limits.h_o = 1.6
    gear_base.shape_recipe.fillet.tip_reduction = 0.0
    gear_base.shape_recipe.fillet.tip_fillet = 0.1
    gear_base.shape_recipe.transform.angle = lambda z: np.abs(z - 2) * beta

    gear_cad = GearBuilder(
        gear=gear_base, n_points_vert=5, n_points_hz=4, add_plug=False, method="fast"
    )

    print(f"gear build time: {time.time()-start}")

    gear2 = gear_base.copy()
    gear2.mesh_to(gear_base, target_dir=rotate_vector(RIGHT, 0))
    gear3 = gear_base.copy()
    gear3.mesh_to(
        gear_base,
        target_dir=rotate_vector(
            RIGHT,
            np.round(2 * PI / 3 / gear_base.pitch_angle) * gear_base.pitch_angle,
        ),
    )
    gear4 = gear_base.copy()
    gear4.mesh_to(
        gear_base,
        target_dir=rotate_vector(
            RIGHT, np.round(4 * PI / 3 / gear_base.pitch_angle) * gear_base.pitch_angle
        ),
    )

    solid1 = gear_cad.solid_transformed
    solid2 = apply_transform_part(gear_cad.solid.mirror(Plane.XZ), gear2.transform)
    solid3 = apply_transform_part(gear_cad.solid.mirror(Plane.XZ), gear3.transform)
    solid4 = apply_transform_part(gear_cad.solid.mirror(Plane.XZ), gear4.transform)
    solid5 = solid1.rotate(Axis((0, 0, gear_base.center_sphere[2]), (0, 1, 0)), 180)
    # export_step(solid1,"fishbone_bevel_left.step")
    # solid1b = solid1.mirror(Plane.XZ)
    # export_step(solid1b,"fishbone_bevel_right.step")
    return (solid1, solid2, solid3, solid4, solid5)


if __name__ == "__main__":
    set_port(3939)
    show(fishbone_bevels())
