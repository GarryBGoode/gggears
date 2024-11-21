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
