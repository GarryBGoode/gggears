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

from gggears import *
from ocp_vscode import show, set_port
import time
import logging

# These examples are meant to showcase the functionality of the library,
# and serve as manual testing templates for the developer.

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def spur_gears():

    gear1 = SpurGear(number_of_teeth=12, profile_shift=0.0)
    gear2 = SpurGear(number_of_teeth=24, enable_undercut=True, root_fillet=0.0)
    gear1.mesh_to(gear2, target_dir=UP)
    gear_part_1 = gear1.build_part()
    gear_part_2 = gear2.build_part()
    return (gear_part_1, gear_part_2)


def helical_gears():
    # a test of helical gears meshing with different helix angles
    # z_anchor=0.5 places the gear symmetrically on the XY plane, makes meshing easier
    gear0 = HelicalGear(number_of_teeth=12, helix_angle=0, height=15, z_anchor=0.5)
    gear1 = HelicalGear(number_of_teeth=12, helix_angle=PI / 4, height=15, z_anchor=0.5)
    gear2 = HelicalGear(number_of_teeth=24, helix_angle=0, height=15, z_anchor=0.5)
    gear3 = HelicalGear(number_of_teeth=12, helix_angle=PI / 4, height=15, z_anchor=0.5)
    gear4 = HelicalGear(number_of_teeth=24, helix_angle=PI / 4, height=15, z_anchor=0.5)
    gear1.mesh_to(gear2, target_dir=LEFT)
    gear0.mesh_to(gear1, target_dir=LEFT)
    gear3.mesh_to(gear2, target_dir=RIGHT)
    gear4.mesh_to(gear3, target_dir=RIGHT)
    gear_part_0 = gear0.build_part()
    gear_part_1 = gear1.build_part()
    gear_part_2 = gear2.build_part()
    gear_part_3 = gear3.build_part()
    gear_part_4 = gear4.build_part()
    return [gear_part_0, gear_part_1, gear_part_2, gear_part_3, gear_part_4]


def worm_approx():
    # this example is stressing the helical gear geometry a bit, and approximates
    # a worm drive
    gear0 = HelicalGear(
        number_of_teeth=3, helix_angle=PI / 2 * 0.85, height=20, z_anchor=0.5
    )
    gear1 = HelicalGear(
        number_of_teeth=45, helix_angle=PI / 2 * 0.15, height=10, z_anchor=0.5
    )
    gear0.mesh_to(gear1, target_dir=RIGHT)
    gear_part_0 = gear0.build_part()
    gear_part_1 = gear1.build_part()
    return [gear_part_0, gear_part_1]


def planetary_helical_gear():
    m = 2

    n_ring = 97
    n_sun = 11
    n_planet = int(np.floor((n_ring - n_sun) / 2))

    beta = 15 * PI / 180
    herringbone = True

    height = 15
    # this hacky correction needs a better treatment later
    angle_correction = PI / n_ring * ((n_planet + 1) % 2)

    gear_ring = HelicalRingGear(
        number_of_teeth=n_ring,
        module=m,
        height=height,
        helix_angle=beta,
        angle=angle_correction,
        herringbone=herringbone,
    )
    gear_sun = HelicalGear(
        number_of_teeth=n_sun,
        module=m,
        height=height,
        helix_angle=-beta,
        herringbone=herringbone,
    )
    gear_planet1 = HelicalGear(
        number_of_teeth=n_planet,
        module=m,
        height=height,
        helix_angle=beta,
        herringbone=herringbone,
    )

    gear_planet2 = gear_planet1.copy()
    gear_planet3 = gear_planet1.copy()

    dir1 = RIGHT
    # If the sun and ring number of teeth are not divisible by 3,
    # the location of the planets is not trivial.
    angle2 = root(lambda x: (x * n_sun + x * n_ring) % 1, 1.0 / 3).x[0] * 2 * PI
    angle3 = root(lambda x: (x * n_sun + x * n_ring) % 1, 2.0 / 3).x[0] * 2 * PI
    dir2 = rotate_vector(RIGHT, angle2)
    dir3 = rotate_vector(RIGHT, angle3)

    # using the mesh_to function to align planets with the sun
    gear_planet1.mesh_to(gear_sun, target_dir=dir1)
    gear_planet2.mesh_to(gear_sun, target_dir=dir2)
    gear_planet3.mesh_to(gear_sun, target_dir=dir3)

    start = time.time()
    gear_planet1_cad = gear_planet1.build_part()
    gear_planet2_cad = gear_planet2.build_part()
    gear_planet3_cad = gear_planet3.build_part()
    gear_sun_cad = gear_sun.build_part()
    gear_ring_cad = gear_ring.build_part()
    print(f"gear build time: {time.time()-start}")

    return (
        gear_ring_cad,
        gear_sun_cad,
        gear_planet1_cad,
        gear_planet2_cad,
        gear_planet3_cad,
    )


def bevel_gear():

    num_teeth_1 = 8
    num_teeth_2 = 21
    beta = PI / 6
    # module
    m = 2
    # half cone angle
    # this calculation ensures that bevels will generate a 90 degree axial angle

    gamma = np.arctan2(num_teeth_1, num_teeth_2)
    gamma2 = np.pi / 2 - gamma

    height = 10
    gear1 = BevelGear(
        number_of_teeth=num_teeth_1,
        module=m,
        height=height,
        cone_angle=gamma * 2,
        helix_angle=beta,
        profile_shift=0.25,
    )
    gear2 = BevelGear(
        number_of_teeth=num_teeth_2,
        module=m,
        height=height,
        cone_angle=gamma2 * 2,
        helix_angle=-beta,
        profile_shift=-0.25,
    )

    gear1.mesh_to(gear2, target_dir=LEFT)
    with BuildPart() as gear1_builder:
        gear1.build_part()
        with Locations([gear1.face_location_top]):
            Cylinder(2, 2, align=(Align.CENTER, Align.CENTER, Align.MIN))

        with Locations([gear1.face_location_bottom]):
            cone_angle_bottom = gear1.cone_angle_limits_bottom[0]
            # Rhe r_o is the radius where the spherical surface ends and transitions to
            # a flat circle.
            # Index [0] in the radii_data_array represents the bottom.
            r0 = gear1.radii_data_array[0].r_o
            h = 0.2
            r1 = r0 - h / np.tan(cone_angle_bottom)
            # bottom location still points up, aligned with the gear,
            # so the cone1 is positioned under the local XY plane
            Cone(r1, r0, h, align=(Align.CENTER, Align.CENTER, Align.MAX))
            Hole(1, depth=None)

    with BuildPart() as gear2_builder:
        gear2.build_part()
        with Locations([gear2.face_location_top]):
            Cylinder(4, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
            Hole(1, depth=None)

    gear_part_1 = gear1_builder.part
    gear_part_2 = gear2_builder.part
    return (
        gear_part_1,
        gear_part_2,
        arc_to_b123d(gear1.radii_data_gen(gear1.p2z(1)).r_p_curve),
        arc_to_b123d(gear2.radii_data_gen(gear2.p2z(1)).r_p_curve),
        arc_to_b123d(gear1.radii_data_gen(gear1.p2z(1)).r_a_curve),
        arc_to_b123d(gear2.radii_data_gen(gear2.p2z(1)).r_a_curve),
        arc_to_b123d(gear1.radii_data_gen(gear1.p2z(1)).r_d_curve),
        arc_to_b123d(gear2.radii_data_gen(gear2.p2z(1)).r_d_curve),
    )


def bevel_chain():
    # This example is meant to showcase and test the mesh_to function for bevel gears.
    gear = BevelGear(
        number_of_teeth=10,
        cone_angle=PI / 4,
        profile_shift=0.5,
        height=5,
        module=1.5,
    )
    n_gears = 6
    gears = [gear.copy() for i in range(n_gears)]
    angles = np.linspace(0, PI, n_gears)
    for i in range(1, n_gears):
        gears[i].mesh_to(gears[i - 1], target_dir=rotate_vector(RIGHT, angles[i]))
    gear_parts = [gear.build_part() for gear in gears]
    return gear_parts


def fishbone_bevels():
    # This example was meant to stress the library a bit, and to generate
    # interlocking bevel gears. In theory it should be possible to design them in a way
    # that they form a 'ball' that mechanically locks together.
    #
    # It is a bit slow to build the gear so time measurements are thrown in here
    start = time.time()

    num_teeth = 9
    # module
    m = 1
    # half cone angle
    gamma = PI / 4
    beta = 0.65

    gear_base = Gear(
        z_vals=[0, 2, 4],
        tooth_param=GearToothParam(num_teeth=num_teeth),
        cone=ConicData(cone_angle=gamma * 2),
        module=m,
    )

    gear_base.shape_recipe.limits.h_a = 1.0
    gear_base.shape_recipe.limits.h_d = 1.1
    gear_base.shape_recipe.limits.h_o = 1.6
    gear_base.shape_recipe.fillet.tip_reduction = 0.0
    gear_base.shape_recipe.fillet.tip_fillet = 0.1
    gear_base.shape_recipe.transform.angle = lambda z: np.abs(z - 2) * beta

    tooth_generator = InvoluteUndercutTooth(
        pressure_angle=35 * PI / 180,
        pitch_radius=gear_base.shape_recipe.tooth_generator.pitch_radius,
        pitch_intersect_angle=gear_base.shape_recipe.tooth_generator.pitch_intersect_angle,
        cone_angle=gamma * 2,
        ref_limits=gear_base.shape_recipe.limits,
        pitch_angle=gear_base.pitch_angle,
    )

    gear_base.shape_recipe.tooth_generator = tooth_generator

    gear_cad = GearBuilder(
        gear=gear_base,
        n_points_vert=4,
        n_points_hz=4,
        method="slow",
        oversampling_ratio=2.5,
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

    solid1 = gear_cad.part_transformed
    solid2 = apply_transform_part(gear_cad.solid.mirror(Plane.XZ), gear2.transform)
    solid3 = apply_transform_part(gear_cad.solid.mirror(Plane.XZ), gear3.transform)
    solid4 = apply_transform_part(gear_cad.solid.mirror(Plane.XZ), gear4.transform)
    solid5 = solid1.rotate(Axis((0, 0, gear_base.center_sphere[2]), (0, 1, 0)), 180)
    # export_step(solid1,"fishbone_bevel_left.step")
    # solid1b = solid1.mirror(Plane.XZ)
    # export_step(solid1b,"fishbone_bevel_right.step")
    return (solid1, solid2, solid3, solid4, solid5)


def cycloid_gear():
    # The cycloid coefficients determine the radius of the rolling circle
    # relative to the pitch circle.
    # The inside coefficient of 0.5 means the rolling circle is half the pitch circle.
    # This is special for cycloidal gears, for an insider rolling circle with half the
    # radius results in a straight line.
    gear1 = CycloidGear(
        number_of_teeth=12,
        inside_cycloid_coefficient=0.5,
        height=4,
    )
    gear2 = CycloidGear(
        number_of_teeth=22,
        inside_cycloid_coefficient=0.5,
        height=4,
    )
    # Cycloid gears need to have the same rolling radii to mesh properly.
    # This function adapts the outside rolling circle of both gears to match.
    gear1.adapt_cycloid_radii(gear2)
    gear1.mesh_to(gear2, target_dir=UP)
    gear_part_1 = gear1.build_part()
    gear_part_2 = gear2.build_part()
    return (gear_part_1, gear_part_2)


def cycloid_drive():
    # This is a kind of experimental setup to test cycloids when the
    # addendum / dedendum limits cannot apply and the teeth are entirely cycloid curves.
    n = 17
    diff = 1
    beta = 3 * PI / 8 * 1
    h = 10
    c1 = 1 / 2 / (n - diff)
    c2 = 1 / 2 / (n)
    gear1 = CycloidGear(
        number_of_teeth=n - diff,
        inside_cycloid_coefficient=c1,
        outside_cycloid_coefficient=c1,
        tip_truncation=0.0,
        addendum_coefficient=c1 * n * 1.25,
        dedendum_coefficient=c1 * n * 1.25,
        cone_angle=0 * PI / 2,
        height=h,
        helix_angle=beta,
    )
    gear2 = CycloidGear(
        number_of_teeth=n,
        module=1.000,  # adding a little bit of clearance
        inside_cycloid_coefficient=c2,
        outside_cycloid_coefficient=c2,
        addendum_coefficient=c2 * n * 1.25,
        dedendum_coefficient=c2 * n * 1.25,
        tip_truncation=0.0,
        cone_angle=0 * PI / 2,
        inside_teeth=True,
        height=h,
        helix_angle=beta,
    )
    gear2.adapt_cycloid_radii(gear1)
    gear2.mesh_to(gear1, target_dir=RIGHT)
    gear_part_1 = gear1.build_part()
    gear_part_2 = gear2.build_part()
    return (gear_part_1, gear_part_2)


if __name__ == "__main__":
    set_port(3939)
    # default deviation is 0.1, default angular tolerance is 0.2.
    # Lower values result in higher resulution.
    show(bevel_gear(), deviation=0.05, angular_tolerance=0.1)
