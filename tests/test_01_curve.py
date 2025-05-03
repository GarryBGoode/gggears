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

import gggears.curve as crv
import matplotlib.pyplot as plt
import numpy as np
from gggears.defs import *
import pytest as pytest


@pytest.mark.parametrize("radius", [0, 1, 3])
@pytest.mark.parametrize("t", np.linspace(0, 1, 31))
def test_curve_length_param(radius, t):
    """
    Basic test of curve class. Tests the lenght-parametrization of an arc curve.
    """

    # this arc is kind of badly parameterized with t
    def arc(t, radius=radius):
        return np.array([t * radius, radius * np.sqrt(1 - t**2), 0])

    # the curve will update the length-lookup parametrization upon creation
    curve1 = crv.Curve(
        arc,
        active=True,
        t0=-1,
        t1=1,
        params={"radius": radius},
        enable_vectorize=True,
        lenght_approx_ndiv=101,
    )
    assert curve1.length == pytest.approx(np.pi * radius, rel=1e-3, abs=1e-3)
    # redo the length-lookup with a higher resolution cause otherwise the accuracy is abysmal
    curve1.len_approx_N = int(1e4)
    curve1.update_lengths()
    # the result should be close to an angle-parameterized arc
    expected = np.array([np.cos((1 - t) * np.pi), np.sin((1 - t) * np.pi), 0]) * radius

    assert curve1(t) == pytest.approx(expected, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("t", np.linspace(0, 1, 31))
def test_curve_algebra(t):
    """
    Test the algebraic operations of the curve class.
    """

    def arc(t, radius=1):
        return np.array([radius * np.cos(t), radius * np.sin(t), 0])

    curve1 = crv.Curve(
        arc, active=True, t0=0, t1=np.pi, params={"radius": 1}, enable_vectorize=True
    )
    curve2 = crv.Curve(
        arc, active=True, t0=0, t1=np.pi, params={"radius": 1.5}, enable_vectorize=True
    )

    # averaging here should result in an arc with radius 1.5
    curve_mix = (curve1 + curve1 * 2) / 2

    assert curve_mix(t) == pytest.approx(curve2(t), rel=1e-3, abs=1e-3)


def test_curve_chain():
    """
    Test CurveChain basics.
    """

    curve1 = crv.LineCurve(3 * LEFT, RIGHT)
    curve2 = crv.LineCurve(RIGHT, RIGHT + UP)
    curve3 = crv.LineCurve(RIGHT + UP, UP)

    chain = crv.CurveChain(curve1, curve2, curve3)

    assert chain.length == pytest.approx(6, rel=1e-9, abs=1e-9)
    assert not chain.is_closed
    assert chain.is_continuous

    curve4 = crv.LineCurve(UP, 3 * LEFT)
    chain.append(curve4)
    assert chain.is_closed
    assert chain.is_continuous
    assert chain.length == pytest.approx(6 + np.sqrt(3**2 + 1), rel=1e-9, abs=1e-9)

    tvals = np.linspace(0, 1, 31)
    diffs = np.linalg.norm(chain(tvals[0:-1]) - chain(tvals[1:]), axis=1)
    # points evenly spaced in t should have relatively equal distances
    assert np.std(diffs / np.mean(diffs)) < 1e-1


if __name__ == "__main__":
    test_curve_chain()
