import gggears as gg
from build123d import *
from ocp_vscode import *
import numpy as np

n1 = 11
n2 = 33

gamma1 = np.arctan2(n1, n2)
gamma2 = np.pi / 2 - gamma1

gear1 = gg.BevelGear(
    number_of_teeth=n1, cone_angle=gamma1 * 2, helix_angle=np.pi / 6, height=5
)
gear2 = gg.BevelGear(
    number_of_teeth=n2, cone_angle=gamma2 * 2, helix_angle=-np.pi / 6, height=5
)

a_gear1 = Compound(
    children=[gear1.build_part().solid() - gear1.face_location_top * Cylinder(2, 10)],
    label="gear1",
)
a_gear2 = Compound(
    children=[gear2.build_part().solid() - gear2.face_location_top * Cylinder(2, 10)],
    label="gear2",
)

gear1.mesh_to(gear2, target_dir=gg.LEFT)

a_gear1.location = gear1.center_location_bottom
a_gear2.location = gear2.center_location_bottom

gears = Compound(children=[a_gear1, a_gear2], label="gears")

n = 30
duration = 2

time_track = np.linspace(0, duration, n + 1)
gear1_track = np.linspace(0, -gear1.pitch_angle * 180 / np.pi * 2, n + 1)
gear2_track = np.linspace(0, gear2.pitch_angle * 180 / np.pi * 2, n + 1)
# assign the tracks to the gears
animation = Animation(gears)
animation.add_track("/gears/gear1", "rz", time_track, gear1_track)
animation.add_track("/gears/gear2", "rz", time_track, gear2_track)
show(gears)

# Start animation
animation.animate(speed=1)
