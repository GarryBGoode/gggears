from gggears import *
from ocp_vscode import show

# thi code is meant to approximate a Hirth joint with a straight tooth 180deg bevel gear
# it is also an example and test of the power of the gear-generator recipe engine

num_teeth = 34
cone_angle = PI
# Gear is not from the wrapper, but from the core, kind of a base-class
# Uses a default simple straigth tooth generator GearToothConicGenerator
testgear = Gear(
    z_vals=[0, 10],
    module=1,
    tooth_param=GearToothParam(num_teeth=num_teeth),
    cone=ConicData(cone_angle=cone_angle),
    tooth_generator=GearToothConicGenerator(
        pitch_intersect_angle=2 * PI / num_teeth,
        pitch_radius=num_teeth / 2,
        cone_angle=cone_angle,
        tooth_angle=PI / 4,
    ),
)
testgear.shape_recipe.fillet.tip_fillet = 0.3
testgear.shape_recipe.fillet.root_fillet = 0.3
# this probably could be done with math to ensure flat top and equal tooth height,
# now just guessing the coefficient
testgear.shape_recipe.tooth_generator.tooth_angle = lambda z: PI / 4 - 0.04 * z

# large values to ensure rounded tips from the fillet
# if addendum cutoff would affect only part of the tooth length, it leads to errors
# changing topology (disappearing face due to tooth sides merging near top)
# is not supported

testgear.shape_recipe.limits.h_a = 3
testgear.shape_recipe.limits.h_d = 3

builder = GearBuilder(testgear, n_points_hz=3, n_points_vert=4)
part = builder.part_transformed
part2 = builder.part_transformed.copy()
part2 = part2.rotate(Axis.X, 180)
part2 = part2.rotate(Axis.Z, 360 / num_teeth / 2)


show(part, part2)
