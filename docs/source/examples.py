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
    exporter.add_layer("Hidden", line_color=(99, 99, 99), line_type=LineType.ISO_DOT)
    exporter.add_shape(visible, layer="Visible")
    exporter.add_shape(hidden, layer="Hidden")
    exporter.write(f"assets/general_ex{example_counter}.svg")


##########################################
# 1. Simple Spur Gears
# [Ex. 1]

gear1 = SpurGear(number_of_teeth=12)
gear2 = SpurGear(number_of_teeth=24)
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

gear1 = BevelGear(number_of_teeth=24, cone_angle=PI / 2, height=5)
gear2 = BevelGear(number_of_teeth=24, cone_angle=PI / 2, height=5)
gear1.mesh_to(gear2, target_dir=RIGHT)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()

# [Ex. 6]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(0, -100, 70))

# show_object(gear_part_1, gear_part_2)

##########################################
# 7. Spiral Bevel Gears
# [Ex. 7]

n1 = 12
n2 = 31

gamma1 = np.arctan2(n1, n2)
gamma2 = PI / 2 - gamma1

gear1 = BevelGear(
    number_of_teeth=n1, cone_angle=gamma1 * 2, height=5, spiral_coefficient=0.5
)
gear2 = BevelGear(
    number_of_teeth=n2, cone_angle=gamma2 * 2, height=5, spiral_coefficient=-0.5
)
gear1.mesh_to(gear2, target_dir=RIGHT + UP)
gear_part_1 = gear1.build_part()
gear_part_2 = gear2.build_part()

# [Ex. 7]
write_svg(Compound([gear_part_1, gear_part_2]), viewpos=(0, -100, 70))

# show(gear_part_1, gear_part_2)
