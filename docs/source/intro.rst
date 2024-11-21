Introduction
============
``gggears`` is a python module for accurate and feature-rich gear geometry generation.
It is designed to be used with build123d python package, but can be used with any CAD workflow via STEP export.

Motivation
==========
``gggears`` is built on geometric calculations via numpy and scipy, and it is designed to circumvent the common limitations of CAD programs when designing gear surfaces.
It is also designed to be easily extensible via object oriented design, so that new gear types can be added with minimal effort.

Warning
=======
``gggears`` is in early development. While functionality is checked for every merge on main branch, there is no stability yet in the API.

Installation
============
This project is not yet registered on pypi. You can install it from git via pip:

```pip install git+https://github.com/GarryBGoode/gggears.git@main```

Or download repository and install locally. Navigate to the repository folder and:

```pip install .```