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

    gear1 = SpurGear(number_of_teeth=12, profile_shift=0.5)
    gear2 = SpurGear(number_of_teeth=24, enable_undercut=False, root_fillet=0.2)
    gear1.mesh_to(gear2, target_dir=UP)
    gear_part_1 = gear1.build_part()
    gear_part_2 = gear2.build_part()
    return (gear_part_1, gear_part_2)


def planetary_helical_gear():
    m = 1

    n_ring = 97
    n_sun = 11
    n_planet = int(np.floor((n_ring - n_sun) / 2))

    beta = 15 * PI / 180

    height = 15
    # this hacky correction needs a better treatment later
    angle_correction = PI / n_ring * ((n_planet + 1) % 2)

    gear_ring = HelicalRingGear(
        number_of_teeth=n_ring,
        module=m,
        height=height,
        helix_angle=beta,
        angle=angle_correction,
    )
    gear_sun = HelicalGear(
        number_of_teeth=n_sun, module=m, height=height, helix_angle=-beta
    )
    gear_planet1 = HelicalGear(
        number_of_teeth=n_planet, module=m, height=height, helix_angle=beta
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

    num_teeth_1 = 16
    num_teeth_2 = 31
    beta = 0.5
    # module
    m = 2
    # half cone angle
    # this calculation ensures that bevels will generate a 90 degree axial angle

    gamma = np.arctan2(num_teeth_1, num_teeth_2)
    gamma2 = np.pi / 2 - gamma

    height = 5
    gear1 = BevelGear(
        number_of_teeth=num_teeth_1,
        module=m,
        height=height,
        cone_angle=gamma * 2,
        spiral_coefficient=beta,
    )
    gear2 = BevelGear(
        number_of_teeth=num_teeth_2,
        module=m,
        height=height,
        cone_angle=gamma2 * 2,
        spiral_coefficient=-beta,
    )
    gear1.mesh_to(gear2, target_dir=UP)
    gear_part_1 = gear1.build_part()
    gear_part_2 = gear2.build_part()
    return (gear_part_1, gear_part_2)


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
    )

    gear_base.shape_recipe.tooth_generator = tooth_generator

    gear_cad = GearBuilder(
        gear=gear_base,
        n_points_vert=4,
        n_points_hz=4,
        add_plug=False,
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

    solid1 = gear_cad.solid_transformed
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
    gear1 = CycloidGear(
        number_of_teeth=n - diff,
        inside_cycloid_coefficient=1 / 2 / (n - diff),
        outside_cycloid_coefficient=1 / 2 / (n - diff),
        tip_truncation=0.0,
        addendum_coefficient=1.5,
        dedendum_coefficient=1.5,
        cone_angle=0 * PI / 2,
    )
    gear2 = CycloidGear(
        number_of_teeth=n,
        module=1.001,  # adding a little bit of clearance
        inside_cycloid_coefficient=1 / 2 / n,
        outside_cycloid_coefficient=1 / 2 / n,
        addendum_coefficient=1.5,
        dedendum_coefficient=1.5,
        tip_truncation=0.0,
        cone_angle=0 * PI / 2,
        inside_teeth=True,
    )
    gear2.adapt_cycloid_radii(gear1)
    gear1.mesh_to(gear2, target_dir=UP)
    gear_part_1 = gear1.build_part()
    gear_part_2 = gear2.build_part()
    return (gear_part_1, gear_part_2)


if __name__ == "__main__":
    set_port(3939)

    show(fishbone_bevels())
