# Copyright 2024 Gergely Bencsik
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gggears.gggears_wrapper as gg
import gggears.gearteeth as gt
import gggears.curve as crv
import matplotlib.pyplot as plt
import numpy as np
from gggears.defs import *
import pytest as pytest
from scipy.spatial.transform import Rotation as scp_Rotation
import shapely as shp


def test_rotation():
    """
    Test the rotation matrix of scipy.
    Not really a test but rather learning how to use the rotation matrix
      with np dimensions.
    """
    rot = scp_Rotation.from_euler("y", np.pi / 2)
    assert rot.as_matrix() == pytest.approx(
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    )
    assert rot.as_matrix() @ np.array([1, 0, 0]) == pytest.approx(np.array([0, 0, -1]))
    # multiplying on the right with the transpose of the rotation matrix is the way to go
    assert np.array(
        [RIGHT, UP, LEFT, IN]
    ) @ rot.as_matrix().transpose() == pytest.approx(
        np.array([IN, UP, OUT, LEFT]), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize(
    "num_teeth", [8, 13, 25, 62, 121]
)  # range of teeth from low end to high-ish
@pytest.mark.parametrize("module", [0.5, 2])  # test if module is used correctly
@pytest.mark.parametrize("angle_ref", np.linspace(0, 1, 7))  # angle progression
@pytest.mark.parametrize(
    "root_fillet",
    [-1, 0, 0.1, 0.4],  # negative value for undercut, only for this test though
)
@pytest.mark.parametrize("tip_fillet", [0, 0.1, 0.4])
def test_gear_intersect(
    num_teeth, module, angle_ref, root_fillet, tip_fillet, enable_plotting=False
):
    """
    Test gears by probing intersection of two gears.
    This test creates 2 gears and moves them into meshing position.
    The test expects no intersecting area between them, while rotating one of the gears
    slightly should result in an intersecting area.
    """
    m = module

    if root_fillet < 0:
        undercut = True
        f0 = 0
    else:
        undercut = False
        f0 = root_fillet

    gamma = 0.0

    num_teeth_2 = 52

    n_poly = 300

    gear1 = gg.InvoluteGear(
        number_of_teeth=num_teeth,
        module=m,
        angle=angle_ref * 2 * PI / num_teeth,
        tip_fillet=tip_fillet,
        root_fillet=0,
        helix_angle=0.3,
        cone_angle=gamma * 2,
        profile_shift=0,
        enable_undercut=True,
    )

    gear2 = gg.Gear(
        z_vals=[0, 1],
        module=m,
        tooth_param=gg.GearToothParam(num_teeth_2),
        cone=gg.ConicData(cone_angle=gamma * 2),
        tooth_generator=gt.InvoluteUndercutTooth(),
    )
    gear2 = gg.InvoluteGear(
        number_of_teeth=num_teeth_2,
        module=m,
        dedendum_coefficient=1 + f0,
        tip_fillet=tip_fillet,
        root_fillet=f0,
        helix_angle=-0.3,
        cone_angle=gamma * 2,
        profile_shift=0,
        enable_undercut=undercut,
    )

    gear2.mesh_to(gear1)

    gear_gen1 = gear1.gearcore.curve_gen_at_z(0)
    gear_gen2 = gear2.gearcore.curve_gen_at_z(0)
    outer_curve = crv.TransformedCurve(
        gear1.gearcore.transform,
        gg.generate_boundary(gear_gen1, gear1.gearcore.tooth_param),
    )

    t1 = np.linspace(-2 / num_teeth, 2 / num_teeth, n_poly)
    points = outer_curve(t1)
    points = np.append(points, gear1.gearcore.transform.center[np.newaxis, :], axis=0)

    outer_curve2 = crv.TransformedCurve(
        gear2.gearcore.transform,
        gg.generate_boundary(gear_gen2, gear2.gearcore.tooth_param),
    )
    t2 = np.linspace(-2 / num_teeth_2, 2 / num_teeth_2, n_poly)
    points2 = outer_curve2(t2)
    points2 = np.append(points2, gear2.gearcore.transform.center[np.newaxis, :], axis=0)

    poly1 = shp.geometry.Polygon(points)
    poly2 = shp.geometry.Polygon(points2)

    # 0.3 degrees rotation
    poly3 = shp.affinity.rotate(poly1, 0.3, origin=(0, 0))
    poly4 = shp.affinity.rotate(poly1, -0.3, origin=(0, 0))

    if enable_plotting:
        ax = plt.axes()
        # ax.plot(points[:, 0], points[:, 1])
        # ax.plot(points2[:, 0], points2[:, 1])
        ax.plot(poly1.exterior.xy[0], poly1.exterior.xy[1], marker=".")
        ax.plot(poly2.exterior.xy[0], poly2.exterior.xy[1], marker=".")
        ax.axis("equal")
        plt.show()

    its1 = poly1.intersection(poly2)
    its2 = poly3.intersection(poly2)
    its3 = poly4.intersection(poly2)
    assert its1.area == pytest.approx(0, abs=1e-4)
    assert its2.area != pytest.approx(0, abs=1e-4)
    assert its3.area != pytest.approx(0, abs=1e-4)


if __name__ == "__main__":

    test_gear_intersect(
        num_teeth=8,
        module=2,
        angle_ref=np.float64(0.8333333333333333),
        root_fillet=0.4,
        tip_fillet=0.4,
        enable_plotting=True,
    )
