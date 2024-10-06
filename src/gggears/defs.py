import numpy as np
# constants taken over from Manim
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi
PI = np.pi




# numerical differentiation global 'small step'
DELTA = 1E-6

# Dimension and shape conventions
# Vectors of 3D coordinates: row vectors, shape(3)
# I used to prefer column vectors, shape(3,1) but it was inconvenient
# Changing this does not immediately work, the code is not that abstract!
VSHAPE = (3)
# Transformation Matrices: 3x3, numpy shape (3,3)
MSHAPE = (3,3)
# Arrays: arrays of vectors, matrices: array index comes before the others, e.g. vector array of 4: (4,3,1), matrix array of 2: (2,3,3)

# Geometry: directions
ORIGIN = np.array((0.0, 0.0, 0.0)).reshape(VSHAPE)
"""The center of the coordinate system."""
UP = np.array((0.0, 1.0, 0.0)).reshape(VSHAPE)
"""One unit step in the positive Y direction."""
DOWN = np.array((0.0, -1.0, 0.0)).reshape(VSHAPE)
"""One unit step in the negative Y direction."""
RIGHT = np.array((1.0, 0.0, 0.0)).reshape(VSHAPE)
"""One unit step in the positive X direction."""
LEFT = np.array((-1.0, 0.0, 0.0)).reshape(VSHAPE)
"""One unit step in the negative X direction."""
IN = np.array((0.0, 0.0, -1.0)).reshape(VSHAPE)
"""One unit step in the negative Z direction."""
OUT = np.array((0.0, 0.0, 1.0)).reshape(VSHAPE)
"""One unit step in the positive Z direction."""
