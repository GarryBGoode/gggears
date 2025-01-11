import gggears as gg
from build123d import *
from ocp_vscode import *

set_port(3939)

# this example demonstrates how to create a simple pair of gears and
# add additional features to them, and then assemble them on a baseplate
gearmodule = 2
gearheight = 4
bore_diameter = 5
pin_diamaeter = 2
sleeve_height = 7
sleeve_thickness = 1

gear1 = gg.HelicalGear(
    number_of_teeth=13, module=gearmodule, height=gearheight, helix_angle=gg.PI / 12
)
gear2 = gg.HelicalGear(
    number_of_teeth=31, module=gearmodule, height=gearheight, helix_angle=-gg.PI / 12
)
gear1.mesh_to(gear2, target_dir=gg.DOWN)

# gggears uses numpy arrays for vectors, build123d uses its own Vector class
# np2v() is shorthand for nppoint2Vector(), which makes the conversion
gear1_center_vector = gg.np2v(gear1.center)
gear2_center_vector = gg.np2v(gear2.center)
axial_distance_vector = gear1_center_vector - gear2_center_vector

with BuildPart() as gear1_part:
    # creating gear part
    gear1.build_part()
    # note: gear1 is moved and rotated to be meshed with gear2 by the mesh_to() method
    # the alignment of the sleeve and pinhole may need to be adjusted
    with Locations((gear1.center_location_top)):
        # note: location of top-center is aligned with tooth no. 0 of the gear
        # the angle is changed from the mesh_to() method and the helix angle as well
        sleeve = Cylinder(
            radius=bore_diameter / 2 + sleeve_thickness,
            height=sleeve_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        loc_pin_hole = Location(
            Vector(0, 0, sleeve_height - pin_diamaeter * 3 / 2),
            (0, 90, 0),
        )
        # Holes with depth=None mean through all the way
        Hole(bore_diameter / 2, depth=None)
        with Locations([loc_pin_hole]):
            Hole(pin_diamaeter / 2, depth=None)
    # revolute joint seems fitting, but rigid could be used as well,
    # since gear rotation animation or simulation is not implemented
    RevoluteJoint(
        "gear_axis",
        axis=Axis(gear1_center_vector, (0, 0, 1)),
        angular_range=(-360, 360),
    )

with BuildPart() as gear2_part:
    gear2.build_part()
    with Locations((gear2.center_location_top)):
        # note: location of top-center is aligned with tooth no. 0 of the gear
        # the angle is changed from the helix angle
        Cylinder(
            radius=bore_diameter / 2 + sleeve_thickness,
            height=sleeve_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        loc_pin_hole = Location(
            Vector(0, 0, sleeve_height - pin_diamaeter * 3 / 2),
            (0, 90, 0),
        )
        # Holes with depth=None mean through all the way
        Hole(bore_diameter / 2, depth=None)
        with Locations([loc_pin_hole]):
            Hole(pin_diamaeter / 2, depth=None)

    RevoluteJoint(
        "gear_axis",
        axis=Axis(gear2_center_vector, (0, 0, 1)),
        angular_range=(-360, 360),
    )


with BuildPart() as baseplate:
    box = Box(100, 10, 50)
    face = box.faces().sort_by(Axis.Y)[0]
    # note: the orientation of the face is such that the local Y aligns with global X
    loc = face.center_location
    # mult operation on locations means locate 2nd location within 1st location
    loc_g1 = loc * Location(axial_distance_vector * 0.5)
    loc_g2 = loc * Location(-axial_distance_vector * 0.5)
    with Locations([loc_g1, loc_g2]):
        Hole(bore_diameter / 2, depth=50)
    # joints don't seem to work well with Locations context manager
    # so they are created outside of it with joint_location specified as kwarg

    # build123d joint system needs pairs of rigid-revolute joints,
    # revolute-revolute pair does not work
    RigidJoint("gear1_axis", joint_location=loc_g1)
    RigidJoint("gear2_axis", joint_location=loc_g2)


baseplate.joints["gear1_axis"].connect_to(gear1_part.joints["gear_axis"])
baseplate.joints["gear2_axis"].connect_to(gear2_part.joints["gear_axis"])

show_all(render_joints=True)
