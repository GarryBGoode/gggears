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
@pytest.mark.parametrize("conic", [False, True])
@pytest.mark.parametrize("cycloid", [False, True])
def test_gear_intersect(
    num_teeth,
    module,
    angle_ref,
    root_fillet,
    tip_fillet,
    conic,
    cycloid,
    enable_plotting=False,
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

    num_teeth_2 = 52

    n_poly = 600

    if conic:
        gamma1 = np.arctan(num_teeth / num_teeth_2)
        gamma2 = PI / 2 - gamma1
    else:
        gamma1 = gamma2 = 0

    if not cycloid:
        gear1 = gg.InvoluteGear(
            number_of_teeth=num_teeth,
            module=m,
            cone_angle=gamma1 * 2,
            angle=angle_ref * 2 * PI / num_teeth,
            tip_fillet=tip_fillet,
            root_fillet=0,
            helix_angle=0.3,
            profile_shift=0,
            enable_undercut=True,
        )

        gear2 = gg.InvoluteGear(
            number_of_teeth=num_teeth_2,
            module=m,
            dedendum_coefficient=1 + f0 + 1e-3,
            tip_fillet=tip_fillet,
            root_fillet=f0,
            helix_angle=-0.3,
            cone_angle=gamma2 * 2,
            profile_shift=0,
            enable_undercut=undercut,
        )
    else:
        # repurpose the undercut bit for the cycloid gear
        if undercut:
            rc = 0.25
        else:
            rc = 0.5
        gear1 = gg.CycloidGear(
            number_of_teeth=num_teeth,
            module=m,
            cone_angle=gamma1 * 2,
            angle=angle_ref * 2 * PI / num_teeth,
            tip_fillet=tip_fillet,
            root_fillet=0,
            helix_angle=0.3,
            inside_cycloid_coefficient=rc,
        )

        gear2 = gg.CycloidGear(
            number_of_teeth=num_teeth_2,
            module=m,
            dedendum_coefficient=1 + f0 + 1e-3,
            tip_fillet=tip_fillet,
            root_fillet=f0,
            helix_angle=-0.3,
            cone_angle=gamma2 * 2,
            inside_cycloid_coefficient=rc,
        )
        gear2.adapt_cycloid_radii(gear1)

    gear2.mesh_to(gear1)

    gear_gen1 = gear1.gearcore.curve_gen_at_z(0)
    gear_gen2 = gear2.gearcore.curve_gen_at_z(0)
    outer_curve = crv.TransformedCurve(
        gear1.gearcore.transform,
        gg.generate_boundary(gear_gen1, gear1.gearcore.tooth_param),
    )

    t_const = 2.5

    t1 = np.linspace(-t_const / num_teeth, t_const / num_teeth, n_poly)
    points = outer_curve(t1)

    polar_tf = gg.GearPolarTransform(
        cone_angle=gamma1 * 2, base_radius=gear1.gearcore.rp
    )
    points1 = polar_tf(points)

    def close_poly(p):
        return np.append(
            p, [p[-1] * 1.01 + LEFT * 30, p[0] * 1.01 + LEFT * 30, p[0]], axis=0
        )

    points1 = close_poly(points1)

    outer_curve2 = crv.TransformedCurve(
        gear2.gearcore.transform,
        gg.generate_boundary(gear_gen2, gear2.gearcore.tooth_param),
    )
    t2 = np.linspace(-t_const / num_teeth_2, t_const / num_teeth_2, n_poly)
    points2 = outer_curve2(t2)
    points2 = polar_tf(points2)
    points2 = np.append(
        points2,
        [points2[-1] * 1.01 + RIGHT * 30, points2[0] * 1.01 + RIGHT * 30, points2[0]],
        axis=0,
    )

    # the y axis is an angle, scaling up by radius to have roughly distance-like values
    points1[:, 1] *= polar_tf.base_radius
    points2[:, 1] *= polar_tf.base_radius

    # the polar transformed points displaced in Y dimension mean rotation
    # slightly rotated shape should intersect
    points3 = points1 + UP * 0.01
    points4 = points1 - UP * 0.01

    poly1 = shp.geometry.Polygon(points1[:, :2])
    poly2 = shp.geometry.Polygon(points2[:, :2])
    poly3 = shp.geometry.Polygon(points3[:, :2])
    poly4 = shp.geometry.Polygon(points4[:, :2])

    if enable_plotting:
        # ax = plt.axes(projection="3d")
        ax = plt.axes()
        ax.plot(poly1.exterior.xy[0], poly1.exterior.xy[1], marker=".")
        ax.plot(poly2.exterior.xy[0], poly2.exterior.xy[1], marker=".")
        ax.plot(poly3.exterior.xy[0], poly3.exterior.xy[1], marker=".")
        ax.plot(poly4.exterior.xy[0], poly4.exterior.xy[1], marker=".")
        # ax.axis("equal")
        plt.show()

    its1 = poly1.intersection(poly2)
    its2 = poly3.intersection(poly2)
    its3 = poly4.intersection(poly2)
    assert its1.area == pytest.approx(0, abs=1e-4)
    assert its2.area != pytest.approx(0, abs=1e-4)
    assert its3.area != pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize("num_teeth", [8, 21, 55, 144])
@pytest.mark.parametrize("module", [0.5, 2])  # test if module is used correctly
@pytest.mark.parametrize("beta", [0, PI / 6])  # spiral angle
# negative value for undercut, only for this test though
@pytest.mark.parametrize("root_fillet", [-1, 0, 0.3])
@pytest.mark.parametrize("height", [0.5, 1.5])
@pytest.mark.parametrize("tip_fillet", [0, 0.25])
@pytest.mark.parametrize("conic", [False, True])
@pytest.mark.parametrize("cycloid", [False, True])
@pytest.mark.parametrize("inside_ring", [False, True])
def test_CAD(
    num_teeth,
    module,
    beta,
    height,
    root_fillet,
    tip_fillet,
    conic,
    cycloid,
    inside_ring,
):
    if conic:
        gamma = PI / 4
    else:
        gamma = 0

    if root_fillet < 0:
        undercut = True
        f0 = 0
    else:
        undercut = False
        f0 = root_fillet

    # ring gear construction can go wrong:
    # there can be a straight radial section of the tooth curve that
    # overlaps the return-line to the ring circle on the closed profile
    # this causes non-planar face construction errors
    if inside_ring and f0 == 0 and not undercut:
        f0 = 0.05

    if not cycloid:
        gear1 = gg.InvoluteGear(
            number_of_teeth=num_teeth,
            module=module,
            cone_angle=gamma * 2,
            helix_angle=beta,
            height=height,
            angle=0,
            tip_fillet=tip_fillet,
            root_fillet=f0,
            profile_shift=0,
            enable_undercut=undercut,
            inside_teeth=inside_ring,
        )
    else:
        if undercut:
            rc = 0.25
        else:
            rc = 0.5

        gear1 = gg.CycloidGear(
            number_of_teeth=num_teeth,
            module=module,
            cone_angle=gamma * 2,
            angle=0,
            tip_fillet=tip_fillet,
            root_fillet=f0,
            helix_angle=beta,
            height=height,
            inside_cycloid_coefficient=rc,
            outside_cycloid_coefficient=rc,
            inside_teeth=inside_ring,
        )

    gearpart = gear1.build_part()
    partvolume = gearpart.volume

    # height is actually the width of the gear surface, so not the z-height of bevels
    r0_add = module * (num_teeth / 2 + gear1.addendum_coefficient)
    r0_ded = module * (num_teeth / 2 - gear1.dedendum_coefficient)
    r0 = (r0_add + r0_ded) / 2
    h = height * np.cos(gamma)
    r1 = r0 - np.sin(gamma) * height
    expected_volume = h * np.pi * (r0**2 + r1**2 + r0 * r1) / 3

    if not inside_ring:
        if conic:
            # conic gear is surprisingly different from a truncated cone in terms of volume
            # so only checking for a very rough approximation
            assert partvolume == pytest.approx(expected_volume * 1.2, rel=0.5, abs=1e-2)
        else:
            assert partvolume == pytest.approx(expected_volume, rel=1e-1, abs=1e-2)
    else:
        # volume approximation doesn't quite work for rings, just check that the volume
        # is not zero or NaN
        assert partvolume > expected_volume * 2 / num_teeth


if __name__ == "__main__":

    # test_gear_intersect(
    #     num_teeth=13,
    #     module=0.5,
    #     angle_ref=np.float64(0.16666666666666666),
    #     root_fillet=0.4,
    #     tip_fillet=0.4,
    #     conic=False,
    #     cycloid=True,
    #     enable_plotting=True,
    # )
    test_CAD(
        num_teeth=55,
        module=2,
        beta=0,
        height=1.5,
        root_fillet=-1,
        tip_fillet=0.25,
        conic=True,
        cycloid=False,
        inside_ring=True,
    )
