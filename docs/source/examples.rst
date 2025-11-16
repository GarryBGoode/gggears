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

    The :py:meth:`mesh_to() <gggears.gggears_wrapper.InvoluteGear.mesh_to>` function calculates accurate zero-backlash meshing of gears with ``profile_shift`` parameter.
    This is slightly shorter distance than the nominal center distance, and may lead to interference at the root.

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
The function :py:meth:`cone_angle_from_teeth() <gggears.gearmath.cone_angle_from_teeth>` can be used to calculate appropriate cone angles for given number of teeth and given angle between axes.

.. image:: assets/general_ex6.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 6]
    :end-before: [Ex. 6]

.. note ::
    Bevel gears are implemented with octoid tooth profile. Undercuts are supported for bevel gears.
    Top and bottom face of bevel gears follow spherical surface (not conic).

7. Spiral Bevel Gears
---------------------
Spiral bevel gears are under development, but you can already create them with the following code.
The spiral geometry follows an euclidean spiral, which is mathematically simple, but is not realistic.
Axis offset (hypoid) geometry is not yet supported.

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


9. Racks
---------------------

Create straight racks with the :py:class:`InvoluteRack <gggears.gggears_wrapper.InvoluteRack>` class.
Racks are developed separately from the gear class structures, so not all features of gears are available for racks.
Racks can use :py:meth:`mesh_to() <gggears.gggears_wrapper.InvoluteRack.mesh_to>` function for positioning next to gears, but gears can't mesh_to() racks (missing feature for  now).

.. image:: assets/general_ex9.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 9]
    :end-before: [Ex. 9]


10. Helical Racks
---------------------

The :py:class:`HelicalRack <gggears.gggears_wrapper.HelicalRack>` class can be used to create racks matching to :py:class:`HelicalGear <gggears.gggears_wrapper.HelicalGear>`  class.
While :py:class:`InvoluteRack <gggears.gggears_wrapper.InvoluteRack>` class also has beta angle input which corresponds to helical angle,
the HelicalRack class also accounts for normal-transverse system conversions to directly match the :py:class:`HelicalGear <gggears.gggears_wrapper.HelicalGear>` class.

.. image:: assets/general_ex10.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 10]
    :end-before: [Ex. 10]

11. Controlling backlash
---------------------
The :py:meth:`mesh_to() <gggears.gggears_wrapper.InvoluteGear.mesh_to>` function has a ``backlash`` parameter to control the distance between meshing gears.
The backlash value is given in mm, it is measured as the distance along the line of action, between the inactive flanks of gearteeth.

.. image:: assets/general_ex11.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 11]
    :end-before: [Ex. 11]
