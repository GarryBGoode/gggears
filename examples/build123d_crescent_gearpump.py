import gggears as gg
from build123d import *
from ocp_vscode import *

set_port(3939)

gearheight = 10
axis_diameter = 6
port_diameter = 5
gearmodule = 2
wall_thickness = 3

gear1 = gg.SpurGear(
    number_of_teeth=17,
    module=gearmodule,
    height=gearheight,
    addendum_coefficient=1.0,
    z_anchor=0.5,
)
gear2 = gg.SpurRingGear(
    number_of_teeth=23,
    module=gearmodule,
    height=gearheight,
    addendum_coefficient=1.2,
    dedendum_coefficient=0.6,
    outside_ring_coefficient=2,
    z_anchor=0.5,
    # I used the angle kwarg to iteratively check for interference
    angle=0.135,
)
gear1.mesh_to(gear2)
gear1.center += gg.LEFT * 0.1

with BuildPart() as gearpart1:
    gear1.build_part()
    with Locations((gear1.center_location_bottom)):
        # notch
        # a rectangular hole on the radius in Y direction
        with Locations((0, axis_diameter / 2, 0)):
            Box(
                length=3,
                width=2,
                height=gearheight,
                mode=Mode.SUBTRACT,
                # location is on the bottom of gear, need to align Z to with MIN
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
        # axle hole
        Hole(radius=axis_diameter / 2)


# ring gear needs no modifications
with BuildPart() as gearpart2:
    gear2.build_part()

# set up rendering colors
gearpart1.part.color = (0.75, 0.75, 0.75)
gearpart2.part.color = (0.6, 0.6, 0.6)


with BuildPart() as housing_base:
    r_outer_gear2 = gear2.max_outside_radius
    r_outer_wall = r_outer_gear2 + wall_thickness
    # External housing with even-ish wall thickness
    Cylinder(radius=r_outer_wall, height=gearheight + wall_thickness * 2, mode=Mode.ADD)
    Cylinder(radius=r_outer_gear2, height=gearheight, mode=Mode.SUBTRACT)


with BuildPart() as housing_bottom:
    add(housing_base.part.split(tool=Plane.XY, keep=Keep.BOTTOM))
    with Locations((gear1.center_location_bottom)):
        Hole(radius=axis_diameter / 2)


with BuildPart() as crescent:
    with BuildSketch():
        # crescent constructed from ring gear inner (dedendum) circle and
        # gear1 outer (addendum) circle.
        with Locations((gear2.center_location_bottom)):
            Circle(radius=gear2.dedendum_radius, mode=Mode.ADD)
        with Locations((gear1.center_location_bottom)):
            Circle(radius=gear1.addendum_radius, mode=Mode.SUBTRACT)
        # cut off the right side sharp tips of crescent
        Rectangle(
            width=gear2.addendum_radius,
            height=2 * gear2.addendum_radius,
            mode=Mode.SUBTRACT,
            align=(Align.MIN, Align.CENTER),
        )
        # fillet for good measure
        fillet(vertices(), radius=1)
    extrude(amount=gearheight / 2, both=True)

crescent.part.color = (0.5, 0.5, 0.8)


# indicator sketches
addendum_circle_1 = gg.arc_to_b123d(gear1.radii_data_top.r_a_curve)
addendum_circle_2 = gg.arc_to_b123d(gear2.radii_data_top.r_a_curve)

# involute base circle is not in the radii data
# because radii data was meant to be generic and apply to other gears
base_circle_1 = gg.arc_to_b123d(gear1.circle_involute_base(z_ratio=1))
base_circle_2 = gg.arc_to_b123d(gear2.circle_involute_base(z_ratio=1))

loa1, loa2 = gg.LineOfAction(gear2, gear1, z_ratio=1).LOA_gen()
line_of_action_1 = gg.line_to_b123d(loa1)
line_of_action_2 = gg.line_to_b123d(loa2)

# coloring
line_of_action_1.color = (1, 0.2, 0.2)
line_of_action_2.color = (1, 0.2, 0.2)
base_circle_1.color = (0, 0, 0)
addendum_circle_1.color = (0, 0, 0)

# construction of the housing top with channel volumes for oil-flow
channel_thickness = 3
# blocker width is aligned with the distance between the ends of the 2 lines of action
# this is not official pump design advice
blocker_width = (line_of_action_1 @ 1 - line_of_action_2 @ 1).length

with BuildPart() as housing_top:
    add(housing_base.part.split(tool=Plane.XY, keep=Keep.TOP))

    # main cavity + horizontal blocker
    # the top_position rotates with the gear, but we only need the position here
    with Locations(Location(gear2.center_location_top.position)):
        Cylinder(
            radius=r_outer_wall,
            height=channel_thickness + wall_thickness,
            mode=Mode.ADD,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        Cylinder(
            radius=gear2.addendum_radius,
            height=channel_thickness,
            mode=Mode.SUBTRACT,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        # horizontal blocker
        Box(
            length=gear2.addendum_radius * 2,
            width=blocker_width,
            height=channel_thickness,
            mode=Mode.ADD,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    # axle support
    # top location is aligned with gear rotation, but only using position here
    with Locations(Location(gear1.center_location_top.position)):
        r = axis_diameter / 2 + wall_thickness
        Cylinder(
            radius=r,
            height=channel_thickness,
            mode=Mode.ADD,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        Box(
            length=2 * r,
            width=2 * r,
            height=channel_thickness,
            mode=Mode.ADD,
            align=(Align.MAX, Align.CENTER, Align.MIN),
        )
    with BuildSketch(Location(gear2.center_location_top.position)):
        Circle(radius=gear2.addendum_radius, mode=Mode.ADD)
        Rectangle(
            width=gear2.addendum_radius,
            height=gear2.addendum_radius * 2,
            align=(Align.MAX, Align.CENTER),
            mode=Mode.INTERSECT,
        )
    extrude(amount=channel_thickness)
    r_hole = (gear1.addendum_radius + gear2.dedendum_radius) / 2
    ax_offs = (gear1.center - gear2.center)[0]
    with Locations([(ax_offs / 2, r_hole, 0), (ax_offs / 2, -r_hole, 0)]):
        Hole(radius=port_diameter / 2)

    with Locations(Location(gear1.center_location_top)):
        # axle pocket, should not go all the way through
        Cylinder(
            radius=axis_diameter / 2,
            height=channel_thickness,
            mode=Mode.SUBTRACT,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

show_all()
