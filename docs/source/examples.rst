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

    The default values are ``module=1`` and ``height=1`` .

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

Create profile shifted gears. Use the ``tip_truncation`` parameter to avoid sharp tips.

.. note::

    The ``mesh_to()`` function does not yet handle fully accurately the ``profile_shift`` parameter, so you get increased backlash with shifted spur gears.

.. image:: assets/general_ex3.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 3]
    :end-before: [Ex. 3]


4. Helical Gears
--------------------

Create Helical Gears.

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

7. Spiral Bevel Gears
---------------------
Spiral bevel gears are under development, but you can already create them with the following code.

.. image:: assets/general_ex7.svg
    :align: center

.. literalinclude:: examples.py
    :start-after: [Ex. 7]
    :end-before: [Ex. 7]
