########
Examples
########

The examples on this page showcase the available functionality of gggears package and help learning the workflow.

.. note::

    Some important lines are omitted below to save space, so you will most likely need to add 1 & 2 to the provided code below for them to work:

        1. ``from gggears import *``
        2. To view the created objects, you can use the following commands:

            - in *ocp_vscode* simply use e.g. ``show(gear_part_1,gear_part_2)`` or ``show_all()`` can be used to automatically show all objects with their variable names as labels.
            - in *CQ-editor* add e.g. ``show_object(gear_part_1)``
        3. To export parts, use build123d's export functions, e.g. ``export_stl(gear_part_1)``


.. contents:: List of Examples
    :backlinks: entry

Basic Examples
==============

.. _ex 1:

1. Basic Spur Gears
-------------------

This example demonstrates the creation of two spur gears with different number of teeth.

.. note::

    The default (unspecified) values are :py:attr:`module=1` and :py:attr:`height=1` .

.. image:: assets/general_ex1.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 1]
    :end-before: [Ex. 1]

.. _ex 2:

2. Inside Ring Gears
--------------------

Example with a ring-gear for planetary drive construction.

.. image:: assets/general_ex2.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 2]
    :end-before: [Ex. 2]


3. Profile shifts
--------------------

Create profile shifted gears. Use the :py:class:`tip_truncation <gggears.gggears_wrapper.InvoluteGear>` parameter to avoid sharp tips.

.. note::

    The :py:meth:`mesh_to() <gggears.gggears_wrapper.InvoluteGear.mesh_to>` function does not yet handle accurately the ``profile_shift`` parameter, so you get increased backlash with shifted spur gears.

.. image:: assets/general_ex3.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 3]
    :end-before: [Ex. 3]


4. Helical Gears
--------------------

Create Helical Gears. Positive helix angle is right-handed, negative is left-handed.
Use positive and negative values to create a common helical pair. The gears are calculated with the 'normal' (tool-parameter) system,
as opposed to the 'transverse' system. The :py:meth:`mesh_to() <gggears.gggears_wrapper.InvoluteGear.mesh_to>` function can account for different helix angles. Use the value ``PI/4`` for 90° crossed helicals.
The class :py:class:`HelicalRingGear <gggears.gggears_wrapper.HelicalRingGear>` is also available for planetary drives with helical gears.

.. image:: assets/general_ex4.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 4]
    :end-before: [Ex. 4]

5. Crowning
--------------------

Crowning is a feature that gradually reduces tooth width from the middile towards the top/bottom face, resulting in a barrel-like side profile.
Crowning can help against axial alignment errors, ensures gears don't make first contact on the edges.
The parameter ``crowning`` has a 1E-3 conversion factor, so values in the range of 100-200 have visible effect.
The value of 1 corresponds to 0.001 module arc length reduction of tooth flank on both left-right sides.

.. image:: assets/general_ex5.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 5]
    :end-before: [Ex. 5]

6. Bevel Gears at 90°
---------------------

Create simple bevel gears at 90° angle.

.. image:: assets/general_ex6.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 6]
    :end-before: [Ex. 6]

.. note ::
    Bevel gears are implemented with ideal spherical involute geometry, which does not represent real bevel gears due to manufacturing constraints.

7. Spiral Bevel Gears
---------------------
Spiral bevel gears are under development, but you can already create them with the following code.

.. image:: assets/general_ex7.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 7]
    :end-before: [Ex. 7]

8. Cycloid Gears
---------------------

Create gears wit cycloid geometry. Cycloids have no pressure angle as a parameter, but are rather defined by the radii of the generator rolling circles.
These generator circles are controlled by the ``inside_cycloid_coefficient`` and ``outside_cycloid_coefficient`` parameters.
The rolling circles need to match for meshing gears.
The :py:meth:`adapt_cycloid_radii <gggears.gggears_wrapper.CycloidGear.adapt_cycloid_radii>` function can be used to adjust the outside rolling circles of gears for appropriate meshing.

.. image:: assets/general_ex8.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 8]
    :end-before: [Ex. 8]

Build123d Workflow Examples
===========================
Various properties and methods are made available in the class :py:class:`GearInfoMixin <gggears.gggears_wrapper.GearInfoMixin>`. 
The following example demonstrates the creation of a gear-pair and attaching them to a base-plate: ::

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
        gearpart = gear2.build_part()
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

