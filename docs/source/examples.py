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

from build123d import *
from gggears import *
from ocp_vscode import show, set_port

# set_port(3939)


def write_svg(part, viewpos=(-100, -100, 70)):
    """Save an image of the BuildPart object as SVG"""
    global example_counter
    try:
        example_counter += 1
    except:
        example_counter = 1

    # builder: BuildPart = BuildPart._get_context()

    visible, hidden = part.project_to_viewport(viewpos)
    max_dimension = max(*Compound(children=visible + hidden).bounding_box().size)
    exporter = ExportSVG(scale=100 / max_dimension)
    exporter.add_layer("Visible")
    # exporter.add_layer("Hidden", line_color=(99, 99, 99), line_type=LineType.ISO_DOT)
    exporter.add_shape(visible, layer="Visible")
    # exporter.add_shape(hidden, layer="Hidden")
    exporter.write(f"assets/general_ex{example_counter}.svg")


##########################################
# 1. Simple Spur Gears
# [Ex. 1]

gear1 = SpurGear(number_of_teeth=12, module=1.2, height=5)
gear2 = SpurGear(number_of_teeth=24, module=1.2, height=5)
gear1.mesh_to(gear2, target_dir=RIGHT)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()

# [Ex. 1]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(0, -40, 70))

# show_object(gear_part_1, gear_part_2)


##########################################
# 2. Internal Ring Gear
# [Ex. 2]

gear1 = SpurGear(number_of_teeth=12)
gear2 = SpurRingGear(number_of_teeth=24)
gear1.mesh_to(gear2, target_dir=RIGHT + UP)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()

# [Ex. 2]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(0, -40, 70))

# show_object(gear_part_1, gear_part_2)

##########################################
# 3. Profile Shifts
# [Ex. 3]

gear1 = SpurGear(number_of_teeth=8)
gear2 = SpurGear(number_of_teeth=8, profile_shift=0.7, tip_truncation=0)
gear3 = SpurGear(number_of_teeth=8, profile_shift=0.7, tip_truncation=0.2)
gear1.mesh_to(gear2, target_dir=LEFT)
gear3.mesh_to(gear2, target_dir=RIGHT)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()
gear_part_3 = gear3.build_part()

# [Ex. 3]
write_svg(Compound([gear_part_1, gear_part_2, gear_part_3]), viewpos=(0, -40, 70))

# show(gear_part_1, gear_part_2, gear_part_3)

##########################################
# 4. Helical Gears
# [Ex. 4]

gear1 = HelicalGear(number_of_teeth=12, height=5, helix_angle=PI / 6)
gear2 = HelicalGear(number_of_teeth=24, height=5, helix_angle=-PI / 6)
gear1.mesh_to(gear2, target_dir=RIGHT)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()

# [Ex. 4]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(0, -40, 20))

# show_object(gear_part_1, gear_part_2)


##########################################
# 5. Crowning
# [Ex. 5]

gear1 = SpurGear(number_of_teeth=24, height=10, crowning=200)
gear_part_1 = gear1.build_part()

# [Ex. 5]
write_svg(gear_part_1, viewpos=(0, -40, 20))

# show_object(gear_part_1, gear_part_2)


##########################################
# 6. 90° Bevel Gears
# [Ex. 6]

n1 = 13
n2 = 32
cone_angle1, cone_angle2 = cone_angle_from_teeth(n1, n2, axis_angle=PI / 2)
gear1 = BevelGear(number_of_teeth=n1, cone_angle=cone_angle1, height=5)
gear2 = BevelGear(number_of_teeth=n2, cone_angle=cone_angle2, height=5)
gear1.mesh_to(gear2, target_dir=RIGHT)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()

# [Ex. 6]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(-50, -100, 30))

# show_object(gear_part_1, gear_part_2)

##########################################
# 7. Spiral Bevel Gears
# [Ex. 7]

n1 = 12
n2 = 31

cone_angle1, cone_angle2 = cone_angle_from_teeth(n1, n2)
gear1 = BevelGear(number_of_teeth=n1, cone_angle=cone_angle1, height=5, helix_angle=0.5)
gear2 = BevelGear(
    number_of_teeth=n2, cone_angle=cone_angle2, height=5, helix_angle=-0.5
)
gear1.mesh_to(gear2, target_dir=RIGHT + UP)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()

# [Ex. 7]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(0, -100, 70))


##########################################
# 8. Cycloid Gears
# [Ex. 8]

n1 = 12
n2 = 31

# Cycloid coefficient refers the radius of the cycloid generator rolling circle
# as a fraction of the pitch circle radius.
gear1 = CycloidGear(number_of_teeth=n1, height=5, inside_cycloid_coefficient=0.25)
gear2 = CycloidGear(number_of_teeth=n2, height=5, inside_cycloid_coefficient=0.5)
# Cycloid gears need to have the same rolling radii to mesh properly.
# This function adapts the outside rolling circle of both gears to match.
gear1.adapt_cycloid_radii(gear2)
gear1.mesh_to(gear2, target_dir=RIGHT)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()

# [Ex. 8]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(0, -20, 70))


##########################################
# 9. Rack and Pinion
# [Ex. 9]

n_gear = 12
n_rack = 40

gear = SpurGear(number_of_teeth=n_gear, module=2, height=5, profile_shift=0.25)
rack = InvoluteRack(number_of_teeth=n_rack, module=2, height=5)
rack.mesh_to(gear, target_dir=LEFT, offset=19)
gear_part = gear.build_part()
rack_part = rack.build_part()
# [Ex. 9]
write_svg(Compound([gear_part, rack_part]), viewpos=(0, -20, 70))


##########################################
# 10. Helical, Herringbone  Rack and Pinion
# [Ex. 10]

n_gear = 12
n_rack = 40

gear = HelicalGear(
    number_of_teeth=n_gear,
    module=2,
    height=25,
    helix_angle=PI / 3,
    herringbone=True,
)
rack = HelicalRack(
    number_of_teeth=n_rack,
    module=2,
    height=25,
    helix_angle=PI / 3,
    herringbone=True,
)
rack.mesh_to(gear, target_dir=LEFT, offset=0)
gear_part = gear.build_part()
rack_part = rack.build_part()
# [Ex. 10]
write_svg(Compound([gear_part, rack_part]), viewpos=(0, -45, 70))
# show(gear_part_1, gear_part_2)


# 11. Controlling backlash
# [Ex. 11]

gear1 = SpurGear(24, module=2)
gear2 = SpurGear(33, module=2)
gear2.center = np.array([-33, 0, 0])
gear1.mesh_to(gear2, target_dir=RIGHT, backlash=0.4, angle_bias=1)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()
# [Ex. 11]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(0, 0, 100))
# show(gear_part_1, gear_part_2)
