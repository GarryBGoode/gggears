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

"""Module for representing curves in 3D space."""

from gggears.defs import *
from gggears.function_generators import *
from scipy.optimize import root, minimize
import copy
from enum import Enum
import functools
import logging


class Curve:
    """
    A class to represent a curve in space.

    Calling a Curve object with a parameter (s) will return the point on the curve at that parameter.
    The parameter (s) is in the range [0, 1] and is lenght-equalized, meaning equally distributed (s)
    values will result in (approximately) equally distanced points in space.
    s=0 is considered the start and s=1 is considered the end of the curve.
    Extrapolation outside this range is possible, but not recommended.

    Attributes
    ----------
    curve_function : callable
        The mathematical function to generate curve points.
        Has to have 1 input argument, but can have multiple keyword arguments.
        Has to return one or more points in 3D space. Points has to correspond to the input argument.
    active : bool
        Used in CurveChains to show if curve is degenerate, such as a 0 length line.
    t_0 : float
        Start of the curve in its natural parameter.
    t_1 : float
        End of the curve in its natural parameter.
    params : dict
        Keyword arguments for the curve_function.
    enable_vectorize : bool
        Vectorize feature is used to iterate over np.array inputs,
        to eg. generate 100 points on the curve with np.linspace.
        If curve_function already handles array inputs (common for numpy functions), this can be disabled.
    """

    def __init__(
        self,
        curve_function: callable,
        active=True,
        t0=0.0,
        t1=1.0,
        params=None,
        enable_vectorize=True,
        lenght_approx_ndiv=21,
    ):
        # active is used to show if curve is degenerate, such as a 0 length line
        if params is None:
            params = {}
        self.active = active
        # start and end of the curve in its natural parameter,
        # eg. an arc that starts and ends at 30 and 60 degrees
        self.t_0 = t0
        self.t_1 = t1
        # mathematical function to generate curve points
        self.function = curve_function
        # parameters of the curve function
        self.params = params

        # length of the curve is stored in the object, calculated by update_lengths()
        self.length = 0
        self.len_approx_N = lenght_approx_ndiv
        # used for length-parametrization conversion
        self.t2s_lookup = {"t": np.array([-1e6, 1e6]), "s": np.array([-1e6, 1e6])}

        # vectorize feature is used to iterate over np.array inputs,
        # to eg. generate 100 points on the curve with np.linspace
        # if curve_function already handles array inputs, this can be disabled
        self.enable_vectorize = enable_vectorize
        if self.active:
            # update_lengths might be CP expensive so only done if active
            self.update_lengths()

    def __call__(self, s):
        t = self.s2t(s)
        if self.enable_vectorize:
            return vectorize(self.function)(t, **self.params)
        else:
            return self.function(t, **self.params)

    def s2t(self, s):
        """Convert length-proportional parameter s to natural parameter t."""
        # numpy interp is faster and more efficient but cannot extrapolate
        retval = np.interp(s, self.t2s_lookup["s"], self.t2s_lookup["t"])
        # custom interpolation function used for extrapolation
        if np.ndim(s) > 0:
            if any(s < 0):
                retval[s < 0] = interpolate(
                    s[s < 0],
                    self.t2s_lookup["s"][0],
                    self.t2s_lookup["s"][1],
                    self.t2s_lookup["t"][0],
                    self.t2s_lookup["t"][1],
                )
            if any(s > 1):
                retval[s > 1] = interpolate(
                    s[s > 1],
                    self.t2s_lookup["s"][-1],
                    self.t2s_lookup["s"][-2],
                    self.t2s_lookup["t"][-1],
                    self.t2s_lookup["t"][-2],
                )
        else:
            if s < 0:
                retval = interpolate(
                    s,
                    self.t2s_lookup["s"][0],
                    self.t2s_lookup["s"][1],
                    self.t2s_lookup["t"][0],
                    self.t2s_lookup["t"][1],
                )
            elif s > 1:
                retval = interpolate(
                    s,
                    self.t2s_lookup["s"][-1],
                    self.t2s_lookup["s"][-2],
                    self.t2s_lookup["t"][-1],
                    self.t2s_lookup["t"][-2],
                )
        return retval

    def t2s(self, t):
        """Convert natural parameter t to length-proportional parameter s."""
        retval = np.interp(t, self.t2s_lookup["t"], self.t2s_lookup["s"])
        if np.ndim(t) > 0:
            if any(t < self.t_0):
                retval[t < self.t_0] = interpolate(
                    t[t < self.t_0],
                    self.t2s_lookup["t"][0],
                    self.t2s_lookup["t"][1],
                    self.t2s_lookup["s"][0],
                    self.t2s_lookup["s"][1],
                )
            if any(t > self.t_1):
                retval[t > self.t_1] = interpolate(
                    t[t > self.t_1],
                    self.t2s_lookup["t"][-1],
                    self.t2s_lookup["t"][-2],
                    self.t2s_lookup["s"][-1],
                    self.t2s_lookup["s"][-2],
                )
        else:
            if t < self.t_0:
                retval = interpolate(
                    t,
                    self.t2s_lookup["t"][0],
                    self.t2s_lookup["t"][1],
                    self.t2s_lookup["s"][0],
                    self.t2s_lookup["s"][1],
                )
            elif t > self.t_1:
                retval = interpolate(
                    t,
                    self.t2s_lookup["t"][-1],
                    self.t2s_lookup["t"][-2],
                    self.t2s_lookup["s"][-1],
                    self.t2s_lookup["s"][-2],
                )
        return retval

    def update_lengths(self):
        """Update the length of the curve and the length-proportional parameter lookup."""
        t_range = np.linspace(self.t_0, self.t_1, self.len_approx_N)
        if self.enable_vectorize:
            value_array = vectorize(self.function)(t_range, **self.params)
        else:
            value_array = self.function(t_range, **self.params)

        curve_len = np.cumsum(
            np.append(
                np.array([0]),
                np.linalg.norm(value_array[1:, :] - value_array[:-1, :], axis=1),
            )
        )
        self.length = curve_len[-1]
        # 0 length curve is degenerate, but division by 0 should still be avoided
        # if curve length is 0 then any point selected at (t) will return the same point
        if self.length == 0:
            self.t2s_lookup["s"] = np.linspace(0, 1, self.len_approx_N)
        else:
            self.t2s_lookup["s"] = curve_len / self.length
        self.t2s_lookup["t"] = t_range

    def reverse(self):
        """Reverse the curve in place."""
        self.t_0, self.t_1 = self.t_1, self.t_0
        self.t2s_lookup["t"] = np.flip(self.t2s_lookup["t"])
        return self

    def derivative(self, t, direction=0, n=1, delta=DELTA):
        """
        Numerically approximate the curve gradient at t.

        Parameters
        ----------
        t: float
            curve parameter where the derivative is evaluated at.
        direction: int
            1: forward, -1: backward, 0: balanced derivative.
        n: int
            derivative order (n=2: second derivative, etc.)
            0 and negative value (integral) does not work.
        delta: float
            small value used for numeric differentiation.
            Hint: consider using larger deltas for higher order derivatives,
            it is easy to run into floating point issues even on double precision.
        """

        def numeric_diff(function, t, direction, delta):
            if direction == 0:
                return (function(t + delta) - function(t - delta)) / 2 / delta
            elif direction > 0:
                return (function(t + delta) - function(t)) / delta
            else:
                return (function(t) - function(t - delta)) / delta

        if n <= 1:
            return numeric_diff(self, t, direction, delta)
        else:
            return numeric_diff(
                self.derivative(t, direction, n - 1, delta), t, direction, delta
            )

    def cut(self, t):
        """Cut the curve at t and return two new curves."""
        curve1 = copy.deepcopy(self)
        curve2 = copy.deepcopy(self)
        curve1.set_end_on(t)
        curve2.set_start_on(t)
        return curve1, curve2

    def copy(self):
        """Deepcopy of the curve"""
        return copy.deepcopy(self)

    def set_start_and_end_on(self, s0, s1):
        """Set the start and end of the curve in length-proportional parameter s."""
        self.t_0 = self.s2t(s0)
        self.t_1 = self.s2t(s1)
        self.update_lengths()

    def set_start_on(self, s0):
        """Set the start of the curve in length-proportional parameter s."""
        self.t_0 = self.s2t(s0)
        self.update_lengths()

    def set_end_on(self, s1):
        """Set the end of the curve in length-proportional parameter s."""
        self.t_1 = self.s2t(s1)
        self.update_lengths()

    def __add__(self, other: "Curve"):
        """Add two curves together. Output vectors are added."""

        def add_func(t, curve=self, other=other):
            return curve(t) + other(t)

        return Curve(
            add_func,
            params={"curve": self, "other": other},
            active=self.active and other.active,
            t0=0,
            t1=1,
            enable_vectorize=False,
        )

    def __sub__(self, other: "Curve"):
        """Subtract two curves. Output vectors are subtracted."""

        def sub_func(t, curve=self, other=other):
            return curve(t) - other(t)

        return Curve(
            sub_func,
            params={"curve": self, "other": other},
            active=self.active and other.active,
            t0=0,
            t1=1,
            enable_vectorize=False,
        )

    def __mul__(self, other):
        """Multiply the curve by a scalar. Output vectors are multiplied."""
        if isinstance(other, (int, float)):

            def mult_func(t, curve=self, val=other):
                return curve(t) * val

            return Curve(
                mult_func,
                active=self.active,
                t0=0,
                t1=1,
                params={"curve": self, "val": other},
                enable_vectorize=self.enable_vectorize,
            )
        else:
            raise TypeError(
                "Multiplication with type {} is not supported".format(type(other))
            )

    def __rmul__(self, other):
        """Multiply the curve by a scalar. Output vectors are multiplied."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide the curve by a scalar. Output vectors are divided."""
        if isinstance(other, (int, float)):
            return Curve(
                lambda t: self(t) / other,
                active=self.active,
                t0=self.t_0,
                t1=self.t_1,
                params={},
                enable_vectorize=self.enable_vectorize,
            )
        else:
            raise TypeError(
                "Division with type {} is not supported".format(type(other))
            )

    @property
    def is_closed(self):
        """Property to check if the curve starts and ends on the same point."""
        return np.linalg.norm(self(0) - self(1)) < DELTA / 100

    def get_curves(self):
        return self


class CurveChain(Curve):
    """A class to represent a kind of polyline made up of Curves. Also can behave like a Curve."""

    def __init__(self, *curves: "Curve", active=True, **kwargs):
        self.curves: list[Curve] = [*curves]
        self._active = active
        self.update_lengths()
        # super().__init__() is actually not needed.
        # the internal data of a curve relates to handling its function(),
        #   a curvechain has no function() of its own
        # all handling (lengths, etc.) is deferred to the contained curves,
        #   curvechain handles navigating the chain

    @property
    def active(self):
        """A CurveChain is active if its own _active attribute is True and has at least 1 subcurve that is active."""
        return self._active and any([curve.active for curve in self.curves])

    @active.setter
    def active(self, value):
        self._active = value

    @property
    def active_list(self):
        """List of active statuses of the curves in the chain."""
        return [curve.active for curve in self.curves]

    def update_lengths(self):
        for curve in self.curves:
            if curve.active:
                curve.update_lengths()

    @property
    def num_curves(self):
        return len(self.curves)

    @property
    def length_array(self):
        """Return the lengths of each curve in the chain in an array."""
        return np.array([curve.length if curve.active else 0 for curve in self.curves])

    @property
    def length(self):
        return np.sum(self.length_array)

    # these might still show up some time
    # needed to override these from inherited functions, but not needed for CurveChain
    def s2t(self, s):
        return s

    def t2s(self, t):
        return t

    @property
    def idx_active_min(self):
        """Find the first active curve in the chain."""
        try:
            return [curve.active for curve in self.curves].index(True)
        # normally this shouldn't be called if the entire curve is inactive but just in case
        except ValueError:
            return len(self.curves)

    @property
    def idx_active_max(self):
        """Find the last active curve in the chain."""
        try:
            return (
                len(self.curves)
                - [curve.active for curve in reversed(self.curves)].index(True)
                - 1
            )
        # normally this shouldn't be called if the entire curve is inactive but just in case
        except ValueError:
            return -1

    def del_inactive_curves(self):
        """Remove inactive curves from the chain."""
        self.curves = [curve for curve in self.curves if curve.active]

    def get_s_index(self, s):
        """Find which curve index s belongs to and how far along it is in the curve.

        Parameters
        ----------
        s : float
            Length-proportional parameter of the chain.

        Returns
        -------
        tuple : (int, float)
            Index of the curve in the chain and the length-proportional parameter of the indexed curve.
        """
        length_portions = self.get_length_portions()
        idx = np.searchsorted(length_portions, s)

        if idx > self.idx_active_max + 1:
            idx = self.idx_active_max + 1
        if idx < self.idx_active_min + 1:
            idx = self.idx_active_min + 1

        if (length_portions[idx] - length_portions[idx - 1]) != 0:
            s_idx = (s - length_portions[idx - 1]) / (
                length_portions[idx] - length_portions[idx - 1]
            )
        else:
            s_idx = 0.5
        return idx - 1, s_idx

    def get_t_for_index(self, idx):
        length_portions = self.get_length_portions()
        return length_portions[idx], length_portions[idx + 1]

    def get_length_portions(self):
        """Return an array of lenght-parameters of the curve starting and ending points."""
        length_sum = np.cumsum(self.length_array)
        length_portions = np.concatenate([[0], length_sum / self.length])
        return length_portions

    def curve_list_eval(self, s):
        """Return the point corresponding to length-proportional parameter s."""
        idx, s2 = self.get_s_index(s)
        point_out = self.curves[idx](s2)
        return point_out

    def __call__(self, p):
        # no need of params of **params for the function
        # the functions technically belong to members,
        #   function of the chain makes little sense
        return vectorize(self.curve_list_eval)(p)

    def get_curves(self, lerp_inactive=False):
        """Return the list of curves in the chain. Flattens the array structure if there are nested chains."""
        curve_list = []
        for curve in self.curves:
            if isinstance(curve, CurveChain):
                curve_list.extend(curve.get_curves())
            else:
                curve_list.append(curve)

        if lerp_inactive:
            for k in range(len(curve_list)):
                ii = k % len(curve_list)
                ii_p1 = (k + 1) % len(curve_list)
                ii_m1 = (k - 1) % len(curve_list)
                if not curve_list[ii].active:
                    curve_list[ii] = LineCurve(
                        curve_list[ii_m1](1), curve_list[ii_p1](0), active=False
                    )
        return curve_list

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, index):
        return self.curves[index]

    def __setitem__(self, index, value):
        self.curves[index] = value

    def __delitem__(self, index):
        self.curves[index] = []

    def __iter__(self):
        return iter(self.curves)

    # TODO: later
    # def __contains__(self, item):
    #     pass
    # def __mul__(self, other):
    #     pass

    # def __add__(self, other):
    #     self.curves = [self.curves, other]
    #     self.update_lengths()

    def append(self, value):
        self.curves.append(value)
        self.update_lengths()

    def extend(self, iterable):
        self.curves = [*self.curves, *iterable]
        self.update_lengths()

    def insert(self, index, value):
        self.curves.insert(index, value)
        self.update_lengths()

    def pop(self, index=-1):
        ret_curve = self.curves.pop(index)
        self.update_lengths()
        return ret_curve

    def clear(self):
        self.curves.clear()

    def reverse(self):
        self.curves.reverse()
        for curve in self.curves:
            curve.reverse()

    def set_start_and_end_on(self, s0, s1, preserve_inactive_curves=False):
        """Set the start and end of the chain in length-proportional parameter s.

        Parameters
        ----------
        s0 : float
            Start of the chain in length-proportional parameter s.
        s1 : float
            End of the chain in length-proportional parameter s.
        preserve_inactive_curves : bool
            If True, curves that are not part of the range will remain in the curves array of
            the chain, with active=False status.
            If False, curves outside of the range get removed.
        """
        idx, s2 = self.get_s_index(s0)
        if preserve_inactive_curves:
            for curve in self.curves[:idx]:
                curve.active = False
            self.curves[idx].set_start_on(s2)
        else:
            self.curves = self.curves[idx:]
            self.curves[0].set_start_on(s2)

        idx, s2 = self.get_s_index(s1)
        if preserve_inactive_curves:
            for curve in self.curves[idx + 1 :]:
                curve.active = False
            self.curves[idx].set_end_on(s2)
        else:
            self.curves = self.curves[: idx + 1]
            self.curves[-1].set_end_on(s2)
        self.update_lengths()

    def set_start_on(self, s0, preserve_inactive_curves=False):
        """Set the start of the chain in length-proportional parameter s.

        Parameters
        ----------
        s0 : float
            Start of the chain in length-proportional parameter s.
        s1 : float
            End of the chain in length-proportional parameter s.
        preserve_inactive_curves : bool
            If True, curves that are not part of the range will remain in the curves array of
            the chain, with active=False status.
            If False, curves outside of the range get removed.
        """
        idx, s2 = self.get_s_index(s0)
        if preserve_inactive_curves:
            for curve in self.curves[:idx]:
                curve.active = False
            self.curves[idx].set_start_on(s2)
        else:
            self.curves = self.curves[idx:]
            self.curves[0].set_start_on(s2)
        self.update_lengths()

    def set_end_on(self, s1, preserve_inactive_curves=False):
        """Set the end of the chain in length-proportional parameter s.

        Parameters
        ----------
        s0 : float
            Start of the chain in length-proportional parameter s.
        s1 : float
            End of the chain in length-proportional parameter s.
        preserve_inactive_curves : bool
            If True, curves that are not part of the range will remain in the curves array of
            the chain, with active=False status.
            If False, curves outside of the range get removed.
        """
        idx, t2 = self.get_s_index(s1)
        if preserve_inactive_curves:
            for curve in self.curves[idx + 1 :]:
                curve.active = False
            self.curves[idx].set_end_on(t2)
        else:
            self.curves = self.curves[: idx + 1]
            self.curves[-1].set_end_on(t2)

        self.update_lengths()

    def cut(self, t, preserve_inactive_curves=False):
        """Cut the chain at t and return two new chains."""
        curve1 = copy.deepcopy(self)
        curve2 = copy.deepcopy(self)

        curve2.set_start_on(t, preserve_inactive_curves)
        curve1.set_end_on(t, preserve_inactive_curves)
        curve1.update_lengths()
        curve2.update_lengths()

        return curve1, curve2

    def fillet_at_locations(self, radius, locations=[0.5, 0.6]):
        """Add a fillet (tangent arc) to the curve chain near the specified locations.

        Suitable points for tangent arc are searched with starting points at the specified locations.
        """
        arc, t1, t2 = fillet_curve(self, radius, start_locations=locations)
        curve1 = copy.deepcopy(self)
        curve2 = copy.deepcopy(self)
        curve1.set_end_on(t1)
        curve2.set_start_on(t2)
        return CurveChain(*curve1.get_curves(), arc, *curve2.get_curves())

    def fillet(self, radius, location=0.5):
        """Add a fillet (tangent arc) to the curve chain at the specified location.

        Suitable points for tangent arc are searched with starting points at the specified single location.
        """
        return self.fillet_at_locations(
            radius, [location + radius / self.length, location - radius / self.length]
        )

    def fillet_at_index(self, radius, index):
        location = self.get_t_for_index(index % self.num_curves)[1]
        return self.fillet(radius, location)

    @property
    def continuity_list(self):
        """A list of booleans showing if the curves in the chain are continuous with each other.

        The last value shows if the curve is closed (continuity between last and first curve).
        Inactive curves count as continuous."""
        out_list = []
        for k in range(self.num_curves):
            diff = np.linalg.norm(
                self.curves[k](1) - self.curves[(k + 1) % self.num_curves](0)
            )
            out_list.append(
                diff < DELTA / 100
                or not self.curves[k].active
                or not self.curves[(k + 1) % self.num_curves].active
            )
        return out_list

    @property
    def is_continuous(self):
        return all(self.continuity_list[:-1])

    @property
    def is_closed(self):
        return np.linalg.norm(self(0) - self(1)) < DELTA / 100 and self.is_continuous


class IntersectMethod(Enum):
    EQUALITY = 1
    MINDISTANCE = 2


def find_curve_intersect(
    curve1: Curve,
    curve2: Curve,
    guess=[0.5, 0.5],
    method: "IntersectMethod" = IntersectMethod.EQUALITY,
):
    """Find the intersection point of two curves."""
    if method == IntersectMethod.EQUALITY:
        res = root(
            lambda t: curve1(t[0]) - curve2(t[1]), np.array([guess[0], guess[1], 0])
        )
    elif method == IntersectMethod.MINDISTANCE:

        def minfunc(t):
            diff = (curve1(t[0]) - curve2(t[1])) / DELTA
            return np.dot(diff, diff)

        res = minimize(minfunc, np.array([guess[0], guess[1]]))
    return res


def find_curve_line_intersect(curve, offset=ORIGIN, line_direction=RIGHT, guess=0):
    """Find the intersection point of a curve and a line."""
    res = root(
        lambda t: np.linalg.norm(np.cross((curve(t) - offset), line_direction)), guess
    )
    return res


def find_curve_plane_intersect(
    curve, plane_normal=OUT, offset=ORIGIN, guess=0, method=IntersectMethod.EQUALITY
):
    """Find the intersection point of a curve and a plane."""

    def target_func(t):
        val = np.dot((curve(t[0]) - offset), plane_normal)
        return val

    if method == IntersectMethod.EQUALITY:
        res = root(target_func, guess)
    else:
        res = minimize(lambda t: target_func(t) ** 2, guess)
    return res


def find_curve_nearest_point(curve: Curve, point, guesses=[0.5]):
    """Find the curve parameter where the curve is closest to a point."""
    results = []
    for guess in guesses:
        results.append(
            minimize(
                lambda t: np.dot((curve(t[0]) - point), (curve(t[0]) - point)), guess
            )
        )
    pass

    return min(results, key=lambda res: res.fun).x[0]


def fit_bezier_hermite_cubic(target_curve: Curve):
    """Fit a cubic bezier curve to a curve using Hermite interpolation."""
    points = np.zeros((4, 3))
    points[0] = target_curve(0)
    points[3] = target_curve(1)
    points[1] = points[0] + target_curve.derivative(0, 1, delta=1e-4) / 3
    points[2] = points[3] - target_curve.derivative(1, -1, delta=1e-4) / 3
    return points


def fit_bezier_hermite_quadratic(target_curve: Curve):
    """Fit a quadratic bezier curve to a curve using Hermite interpolation."""
    points = np.zeros((3, 3))
    points[0] = target_curve(0)
    points[2] = target_curve(1)
    d1 = target_curve.derivative(0, 1, delta=1e-4) / 3
    d2 = target_curve.derivative(1, -1, delta=1e-4) / 3
    # rooting is probably overkill but I don't want to think harder
    sol = root(
        lambda x: points[0] + x[0] * d1 - points[2] - x[1] * d2,
        0.5 * np.ones(points[0].shape),
    )
    points[1] = points[0] + sol.x[0] * d1
    return points


def fit_nurb_points(
    target_points: np.ndarray, n_points=4, force_2D=False, initpoints=None
):
    N_target = target_points.shape[0]

    N_Dim = 2 if force_2D else 3
    scaler = 1

    def point_allocator(x):
        points = np.zeros((n_points, N_Dim))
        points[0] = target_points[0, :N_Dim]
        points[-1] = target_points[-1, :N_Dim]
        weights = np.ones((n_points))
        for k in range(1, n_points - 1):
            ii = N_Dim * (k - 1)
            points[k] = np.array([x[ii + j] for j in range(N_Dim)])
            weights[k] = x[N_Dim * (n_points - 2) + k - 1]
        t = np.zeros((N_target))
        t[1:-1] = x[
            (N_Dim + 1) * (n_points - 2) : (N_Dim + 1) * (n_points - 2) + N_target - 2
        ]
        t[-1] = 1
        return points, weights, t

    def inverse_allocator(points, weights, t):
        x = np.zeros((N_Dim + 1) * (n_points - 2) + N_target - 2)

        for k in range(1, n_points - 1):
            ii = N_Dim * (k - 1)
            for j in range(N_Dim):
                x[ii + j] = points[k, j]
            x[N_Dim * (n_points - 2) + k - 1] = weights[k]

        x[
            (N_Dim + 1) * (n_points - 2) : (N_Dim + 1) * (n_points - 2) + N_target - 2
        ] = t[1:-1]
        return x

    if initpoints is None:
        initpoints = bezierdc(t=np.linspace(0, 1, n_points), points=target_points)

    initguess_x = inverse_allocator(
        points=initpoints,
        weights=np.ones(n_points),
        t=np.linspace(0, 1, N_target),
    )

    def cost_fun_combined(x):
        points, weights, t = point_allocator(x)
        diff = (target_points[:, :N_Dim] - nurbezier(t, points, weights)) * scaler
        deriv_dpoints = -nurbezier_diff_points(t, points.shape[0], weights)
        deriv_dweights = -nurbezier_diff_weights(t, points, weights)
        deriv_dt = -nurbezier_diff_t(t, points, weights)
        dp = deriv_dpoints.T @ diff
        dt = np.diag(diff @ deriv_dt.T)
        dw = np.einsum("ij,ikj->k", diff, deriv_dweights)
        jac = inverse_allocator(dp, dw, dt)
        costval = np.sum(diff**2) / 2
        return costval, jac

    sol = minimize(
        cost_fun_combined,
        initguess_x,
        # method="TNC",
        method="BFGS",
        jac=True,
        # tol=1e-8,
        # options={"eps": DELTA / 100},
    )

    logging.debug(
        f"Nurbs point fit stats: n_iter: {sol.nit}, nfev: {sol.nfev}, status: {sol.status}"
    )
    logging.debug(f"Final cost: {sol.fun}")
    logging.debug(f"message: {sol.message}")

    points, weights, t = point_allocator(sol.x)

    return sol, points, weights


def fit_nurb_optim(
    target_curve: Curve, n_points=4, force_2D=False, samp_ratio=1.5, initguess=None
):
    N_Dim = 2 if force_2D else 3

    scaler = 1
    # each bezier point brings 4 DoF unknown (xyz + w)
    # each eval point uses 2 DoF known (xyz - t)
    # on average at least 2 eval points are needed per bezier point

    n_fit_points = int(np.ceil((n_points - 2) * 2 * samp_ratio)) + 2
    # the 2 edge points are enforced
    tvals = np.linspace(0, 1, n_fit_points)
    target_points = (target_curve(tvals) * scaler)[:, :N_Dim]

    if initguess is None:
        if n_points == 3:
            initpoints = fit_bezier_hermite_quadratic(target_curve)
        elif n_points == 4:
            initpoints = fit_bezier_hermite_cubic(target_curve)
        else:
            hermite_points = fit_bezier_hermite_cubic(target_curve)
            initpoints = np.zeros((n_points, N_Dim))
            initpoints[0] = hermite_points[0]
            initpoints[-1] = hermite_points[-1]
            initpoints[1] = hermite_points[1]
            initpoints[-2] = hermite_points[2]
            initpoints[2:-2] = np.linspace(
                hermite_points[1], hermite_points[-1], n_points - 2
            )[1:-1]
    else:
        initpoints = initguess
    sol, points, weights = fit_nurb_points(
        target_points, n_points, force_2D=force_2D, initpoints=initpoints
    )
    if force_2D:
        points = np.pad(
            points, [(0, 0), (0, 1)], constant_values=np.mean(target_curve(tvals)[:, 2])
        )

    return sol, points, weights


def convert_curve_nurbezier(input_curve: Curve, skip_inactive=True, **kwargs):
    """Convert a curve to a NURBS curve.

    If the input is a single Curve, it is converted to a NURB curve. If the input is a CurveChain,
    it is converted to NURBS.
    If the input is an arc (ArcCurve) shorter tha 180deg, it is converted to a 3-point NURB with exact solution.
    If the input is a LineCurve, it is converted to a 2-point NURB.
    Otherwise conversion is done via approximation using optimization."""
    if hasattr(input_curve, "__iter__"):
        out_curve_list = []
        for curve in input_curve:
            if curve.active or not skip_inactive:
                out_curve_list.append(convert_curve_nurbezier(curve, **kwargs))
        return NURBSCurve(*out_curve_list)

    else:
        if isinstance(input_curve, NurbCurve):
            return input_curve
        elif isinstance(input_curve, TransformedCurve):
            transform = input_curve.transform_method
            convert = convert_curve_nurbezier(input_curve.target_curve, **kwargs)
            convert.apply_transform(transform)
            return convert
        elif isinstance(input_curve, LineCurve):
            bz_points = np.array([input_curve(0), input_curve(1)])
            bz_weights = np.ones((2))
        elif isinstance(input_curve, ArcCurve):
            if abs(input_curve.angle) < PI:
                bz_points, bz_weights = calc_nurbezier_arc(
                    input_curve(0), input_curve(1), input_curve.center
                )
            else:
                sol, bz_points, bz_weights = fit_nurb_optim(input_curve, **kwargs)
        else:
            sol, bz_points, bz_weights = fit_nurb_optim(input_curve, **kwargs)
        # out_curve = Curve(nurbezier,params={'points':bz_points,'weights':bz_weights})
        out_curve = NurbCurve(bz_points, bz_weights, active=input_curve.active)

    return out_curve


def calc_tangent_arc(
    curve1: Curve,
    curve2: Curve,
    radius: float,
    start_locations=[1, 0],
    method=IntersectMethod.EQUALITY,
):
    def calc_centers(t1, t2):
        p1 = curve1(t1)
        p2 = curve2(t2)
        tan1 = normalize_vector(curve1.derivative(t1))
        tan2 = normalize_vector(curve2.derivative(t2))

        arc_axis = np.cross(tan1, tan2)
        angle = np.linalg.norm(arc_axis)
        arc_axis = arc_axis / angle

        normal1 = np.cross(tan1, arc_axis)
        normal2 = np.cross(tan2, arc_axis)

        center1 = p1 - normal1 * radius
        center2 = p2 - normal2 * radius
        return center1, center2

    def cost_fun(x):
        t1 = start_locations[0] - x[0]
        t2 = start_locations[1] + x[1]
        center1, center2 = calc_centers(t1, t2)
        return center1 - center2

    def cost_fun2(x):
        t1 = start_locations[0] - x[0]
        t2 = start_locations[1] + x[1]
        center1, center2 = calc_centers(t1, t2)
        return np.dot(center1 - center2, center1 - center2)

    if method == IntersectMethod.EQUALITY:
        sol1 = root(cost_fun, np.array([0, 0, 0]))
    elif method == IntersectMethod.MINDISTANCE:
        sol1 = minimize(cost_fun2, np.array([0, 0]))

    t1 = start_locations[0] - sol1.x[0]
    t2 = start_locations[1] + sol1.x[1]
    center1, center2 = calc_centers(t1, t2)
    center = (center1 + center2) / 2

    arc = ArcCurve.from_2_point_center(p0=curve1(t1), p1=curve2(t2), center=center)
    return arc, t1, t2, sol1


def fillet_curve(input_curves: CurveChain, radius: float, start_locations=[0.5, 0.5]):

    def calc_centers(t1, t2):
        p1 = input_curves(t1)
        p2 = input_curves(t2)
        tan1 = normalize_vector(input_curves.derivative(t1))
        tan2 = normalize_vector(input_curves.derivative(t2))

        arc_axis = np.cross(tan1, tan2)
        angle = np.linalg.norm(arc_axis)
        arc_axis = arc_axis / angle

        normal1 = np.cross(tan1, arc_axis)
        normal2 = np.cross(tan2, arc_axis)

        center1 = p1 - normal1 * radius
        center2 = p2 - normal2 * radius
        return center1, center2

    def cost_fun(x):
        t1 = start_locations[0] - x[0]
        t2 = start_locations[1] + x[1]
        center1, center2 = calc_centers(t1, t2)
        # return [np.linalg.norm(center1-center2),np.linalg.norm(center1-center2)]
        return center1 - center2

    sol1 = root(cost_fun, np.array([0, 0, 0]))
    if not sol1.success:
        # try again with different first guess
        guess2 = radius / input_curves.length
        sol1 = root(cost_fun, np.array([guess2, guess2, 0]))
    if not sol1.success:
        # try again with different first guess
        guess2 = radius / input_curves.length
        sol1 = root(cost_fun, np.array([DELTA, DELTA, 0]))
    t1 = start_locations[0] - sol1.x[0]
    t2 = start_locations[1] + sol1.x[1]

    center1, center2 = calc_centers(t1, t2)
    center = (center1 + center2) / 2

    arc = ArcCurve.from_2_point_center(
        p0=input_curves(t1), p1=input_curves(t2), center=center
    )

    return arc, t1, t2


def curve_index(curve: Curve, center: np.ndarray = ORIGIN, n_points=1000):
    """Calculate the index (winding number) of a curve around a point or points."""
    # assumed curve is closed
    points = curve(np.linspace(0, 1, n_points + 1))[:-1, :]
    cpoints = points[:, 0].astype(complex) + points[:, 1].astype(complex) * 1j
    diffs = (np.roll(cpoints, -1, axis=0) - np.roll(cpoints, 1, axis=0)) / 2
    if center.ndim == 1:
        ccenter = center[0] + center[1] * 1j
        index = np.abs(np.sum(diffs / (cpoints - ccenter))) * 1 / (2 * PI)
        return index
    else:
        ccenter = center[:, 0] + center[:, 1] * 1j
        index = (
            np.abs(
                np.sum(
                    diffs[np.newaxis, :]
                    / (cpoints[np.newaxis, :] - ccenter[:, np.newaxis]),
                    axis=1,
                )
            )
            * 1
            / (2 * PI)
        )
        return index


class LineCurve(Curve):
    """Class to represent a line as a Curve."""

    def __init__(self, p0=ORIGIN, p1=ORIGIN, active=True, enable_vectorize=False):
        self.p0 = p0
        self.p1 = p1
        super().__init__(
            self.line_func,
            active,
            t0=0,
            t1=1,
            params={},
            enable_vectorize=enable_vectorize,
        )

    def line_func(self, t):
        if isinstance(t, np.ndarray):
            return (1 - t)[:, np.newaxis] * self.p0[np.newaxis, :] + t[
                :, np.newaxis
            ] * self.p1[np.newaxis, :]
        else:
            return self.p0 * (1 - t) + self.p1 * t

    def update_lengths(self):
        self.length = np.linalg.norm(self.p1 - self.p0) * np.abs(self.t_1 - self.t_0)
        self.t2s_lookup["s"] = np.array([0, 1])
        self.t2s_lookup["t"] = np.array([self.t_0, self.t_1])

    def transform(self, transform: callable) -> "LineCurve":
        p0 = transform(self.p0)
        p1 = transform(self.p1)
        return LineCurve(p0, p1, active=self.active)


class ArcCurve(Curve):
    """Class to represent an arc as a Curve."""

    def __init__(
        self,
        radius=1.0,
        angle=PI / 2,
        center=ORIGIN,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        active=True,
    ):
        self._radius = radius
        self._angle = angle
        self._center = center
        self._yaw = yaw
        self._pitch = pitch
        self._roll = roll
        self._rotmat = self.gen_rotmat()
        super().__init__(self.arcfunc, active=active, enable_vectorize=False)

    def gen_rotmat(self):
        return scp_Rotation.from_euler(
            "zyx", [self._yaw, self._pitch, self._roll]
        ).as_matrix()

    def arcfunc(self, t):
        rot_arc = scp_Rotation.from_euler("z", self._angle * t).as_matrix()
        points = self._rotmat @ rot_arc @ (RIGHT * self._radius) + self._center
        return points

    def update_lengths(self):
        self.length = self.radius * self.angle
        self.t2s_lookup["s"] = np.array([0, 1])
        self.t2s_lookup["t"] = np.array([self.t_0, self.t_1])

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @property
    def r(self):
        return self._radius

    @property
    def angle(self):
        return self._angle

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = value

    @property
    def p0(self):
        return self(0)

    @property
    def p1(self):
        return self(1)

    @property
    def curvature(self):
        return 1 / self._radius

    @property
    def revolutions(self):
        return self._angle // (PI * 2)

    @property
    def axis(self):
        return self._rotmat @ OUT

    @property
    def roll(self):
        return self._roll

    @property
    def pitch(self):
        return self._pitch

    @property
    def yaw(self):
        return self._yaw

    @property
    def rotmat(self):
        return self._rotmat

    @rotmat.setter
    def rotmat(self, R):
        yaw, pitch, roll = scp_Rotation.from_matrix(R).as_euler("zyx")
        self._rotmat = R
        self._yaw = yaw
        self._pitch = pitch
        self._roll = roll

    def gen_rotation_angles(self):
        return scp_Rotation.from_matrix(self._rotmat).as_euler("zyx")

    @classmethod
    def from_2_point_center(
        cls, p0=RIGHT, p1=UP, center=ORIGIN, revolutions=0, active=True
    ):
        r = np.linalg.norm(p0 - center)
        x = normalize_vector(p0 - center)
        z = normalize_vector(np.cross(p0 - center, p1 - center))
        y = np.cross(z, x)
        R = np.transpose(np.array([x, y, z]))
        yaw, pitch, roll = scp_Rotation.from_matrix(R).as_euler("zyx")
        return cls(
            radius=r,
            angle=angle_between_vectors(p0 - center, p1 - center)
            + revolutions * PI * 2,
            center=center,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            active=active,
        )

    @classmethod
    def from_2_point_curvature(
        cls, p0=RIGHT, p1=UP, curvature=1, axis=OUT, revolutions=0, active=True
    ):
        r = 1 / curvature
        if any((p1 - p0) != 0):
            dp = normalize_vector(p1 - p0)
        else:
            # this is baaad but can't deal with it rn
            dp = UP

        axis = normalize_vector(axis - np.dot(axis, dp) * dp)

        if abs(r) < np.linalg.norm(p1 - p0) / 2:
            r = np.linalg.norm(p1 - p0) / 2 * np.sign(r)

        h = np.sqrt(r**2 - (np.linalg.norm(p1 - p0) / 2) ** 2) * np.sign(r)

        center = (p0 + p1) / 2 - np.cross(dp, axis) * h
        return cls.from_2_point_center(
            p0=p0, p1=p1, center=center, revolutions=revolutions, active=active
        )

    @classmethod
    def from_point_center_angle(
        cls, p0=RIGHT, center=ORIGIN, angle=PI / 2, axis=OUT, active=True
    ):
        r = np.linalg.norm(p0 - center)
        x = normalize_vector(p0 - center)
        y = normalize_vector(np.cross(axis, x))
        z = np.cross(x, y)
        R = np.transpose(np.array([x, y, z]))
        yaw, pitch, roll = scp_Rotation.from_matrix(R).as_euler("zyx")
        return cls(
            radius=r,
            angle=angle,
            center=center,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            active=active,
        )

    @classmethod
    def from_radius_center_angle(
        cls,
        radius=1,
        center=ORIGIN,
        angle_start=0.0,
        angle_end=PI / 2,
        axis=OUT,
        active=True,
    ):
        z = axis
        x = np.cross(UP, z)
        if np.linalg.norm(x) == 0:
            x = OUT
        else:
            x = normalize_vector(x)
        y = np.cross(z, x)
        R = np.transpose(np.array([x, y, z]))
        # in theory this should not return any yaw value
        yaw, pitch, roll = scp_Rotation.from_matrix(R).as_euler("zyx")
        return cls(
            radius=radius,
            angle=angle_end - angle_start,
            center=center,
            yaw=angle_start,
            pitch=pitch,
            roll=roll,
            active=active,
        )

    def transform(self, transform: callable) -> "ArcCurve":
        p0 = transform(self(0))
        center0 = transform(self._center)
        center2 = transform(self.center + self.axis)
        axis2 = normalize_vector(center2 - center0)
        return ArcCurve.from_point_center_angle(
            p0, center0, self.angle, axis2, active=self.active
        )


class InvoluteCurve(Curve):
    """Class to represent an involute curve as a Curve."""

    def __init__(
        self,
        r=1,
        t0=0,
        t1=1,
        angle=0,
        v_offs=ORIGIN,
        z_offs=0,
        active=True,
        enable_vectorize=True,
    ):
        self.r = r
        self.angle = angle
        self.v_offs = v_offs
        self.z_offs = z_offs
        super().__init__(
            lambda t: involute_circle(
                t, r=self.r, angle=self.angle, v_offs=self.v_offs, z_offs=self.z_offs
            ),
            active=active,
            t0=t0,
            t1=t1,
            params={},
            enable_vectorize=enable_vectorize,
        )

    @property
    def base_radius(self):
        return self.r


class SphericalInvoluteCurve(Curve):
    """Class to represent a spherical involute curve as a Curve."""

    def __init__(
        self,
        r=1,
        t0=0,
        t1=1,
        angle=0,
        c_sphere=1,
        v_offs=ORIGIN,
        z_offs=0,
        active=True,
        enable_vectorize=True,
    ):
        self.r = r
        self.angle = angle
        self.c_sphere = c_sphere
        self.v_offs = v_offs
        self.z_offs = z_offs
        super().__init__(
            lambda t: involute_sphere(
                t,
                r=self.r,
                C=self.c_sphere,
                angle=self.angle,
                v_offs=self.v_offs,
                z_offs=self.z_offs,
            ),
            active,
            t0,
            t1,
            params={},
            enable_vectorize=enable_vectorize,
        )

    @property
    def center(self):
        return (
            np.sqrt(self.R**2 - self.r**2) * np.sign(self.c_sphere) + self.z_offs
        ) * OUT

    @property
    def center_sphere(self):
        return self.center

    @property
    def R(self):
        return 1 / self.c_sphere

    @property
    def base_radius(self):
        return self.r


class OctoidCurve(Curve):

    def __init__(
        self,
        r=1.0,
        t0=0.0,
        t1=1.0,
        angle=0.0,
        c_sphere=1.0,
        alpha=20 * PI / 180,
        v_offs=ORIGIN,
        z_offs=0.0,
        active=True,
        enable_vectorize=True,
    ):
        self.r = r
        self.angle = angle
        self.c_sphere = c_sphere
        self.v_offs = v_offs
        self.z_offs = z_offs
        self.alpha = alpha
        super().__init__(
            lambda t: octoid(
                t,
                base_rad=self.r,
                sphere_rad=1 / self.c_sphere,
                alpha=self.alpha,
                angle=self.angle,
                v_offs=self.v_offs,
                z_offs=self.z_offs,
            ),
            active,
            t0,
            t1,
            enable_vectorize=enable_vectorize,
        )

    @property
    def center(self):
        return (
            np.sqrt(self.R**2 - self.r**2) * np.sign(self.c_sphere) + self.z_offs
        ) * OUT

    @property
    def center_sphere(self):
        return self.center

    @property
    def R(self):
        return 1 / self.c_sphere

    @property
    def base_radius(self):
        return self.r


class TransformedCurve(Curve):
    """Class for applying simple transformations to a Curve and using the result as a
    Curve."""

    def __init__(
        self,
        transform: callable,
        curve: Curve,
        params=None,
        enable_vectorize=False,
        active=True,
    ):
        self.target_curve = curve
        self.transform_method = transform

        super().__init__(
            lambda t: self.apply_transform(self.target_curve(t)),
            active=self.target_curve.active,
            t0=0,
            t1=1,
            params=params,
            enable_vectorize=enable_vectorize,
        )

    def apply_transform(self, point):
        return self.transform_method(point, **self.params)


class MirroredCurve(TransformedCurve):
    """Class for mirroring a Curve across a plane."""

    def __init__(self, curve: Curve, plane_normal=RIGHT, center=ORIGIN):
        self.plane_normal = normalize_vector(plane_normal)
        self.center = center

        def mirror_func(p):
            p2 = p - self.center
            h = np.dot(p2, self.plane_normal)
            if hasattr(h, "__iter__"):
                return (
                    p2
                    - 2 * h[:, np.newaxis] * self.plane_normal[np.newaxis, :]
                    + self.center
                )
            else:
                return p2 - 2 * h * self.plane_normal + self.center

        super().__init__(mirror_func, curve)


class RotatedCurve(TransformedCurve):
    """Class for rotating a Curve around an axis."""

    def __init__(self, curve: Curve, angle=0.0, axis=OUT, center=ORIGIN):
        self.axis = normalize_vector(axis)
        self.angle = angle
        self.center = center

        def rotate_func(p):
            p2 = p - self.center
            return (
                scp_Rotation.from_rotvec(self.angle * self.axis).apply(p2) + self.center
            )

        super().__init__(rotate_func, curve)


class NurbCurve(Curve):
    """Class to represent a NURB (a single piece of non-universial rational bezier)
    curve as a Curve."""

    def __init__(self, points, weights, active=True):
        self.points = points
        self.weights = weights
        super().__init__(
            self.function, active, t0=0, t1=1, params={}, enable_vectorize=False
        )

    def function(self, t):
        return nurbezier(t, self.points, self.weights)

    def reverse(self):
        self.points = np.flip(self.points, axis=0)
        self.weights = np.flip(self.weights, axis=0)
        return self

    @property
    def n_points(self):
        return self.points.shape[0]

    def apply_transform(self, transform):
        self.points = transform(self.points)
        return self


class NURBSCurve(CurveChain):
    """Class to represent a NURBS curve chain as a CurveChain of NURB curves."""

    def __init__(self, *curves: "NurbCurve", active=True, **kwargs):
        self.curves: NurbCurve = [*curves]
        self._active = active
        self.update_lengths()

    @property
    def n_points(self):
        return sum([curve.n_points for curve in self.curves])

    @property
    def points(self):
        out_arr = np.concatenate([curve.points[:-1] for curve in self.curves])
        out_arr = np.append(out_arr, self.curves[-1].points[-1, np.newaxis], axis=0)
        return out_arr

    @points.setter
    def points(self, newpoints):
        if newpoints.shape != self.points.shape:
            raise ValueError("New points shape must match the old one")
        for k, curve in enumerate(self.curves):
            # knots start with 0 which is the first point
            idx = self.knots[k]
            curve.points = newpoints[idx : idx + curve.n_points]

    @property
    def weights(self):
        out_arr = np.concatenate([curve.weights[:-1] for curve in self.curves])
        out_arr = np.append(out_arr, self.curves[-1].weights[-1, np.newaxis], axis=0)
        return out_arr

    @property
    def knots(self):
        out_arr = np.array([0])
        for curve in self.curves:
            out_arr = np.append(out_arr, curve.n_points - 1 + out_arr[-1])
        return out_arr

    def enforce_continuity(self):
        for curve1, curve2 in zip(self.curves[:-1], self.curves[1:]):
            midpoints = (curve1.points[-1] + curve2.points[0]) / 2
            midweights = (curve1.weights[-1] + curve2.weights[0]) / 2
            curve1.points[-1] = midpoints
            curve2.points[0] = midpoints
            curve1.weights[-1] = midweights
            curve2.weights[0] = midweights
        if self.is_closed:
            midpoints = (self.curves[-1].points[-1] + self.curves[0].points[0]) / 2
            midweights = (self.curves[-1].weights[-1] + self.curves[0].weights[0]) / 2
            self.curves[-1].points[-1] = midpoints
            self.curves[0].points[0] = midpoints
            self.curves[-1].weights[-1] = midweights
            self.curves[0].weights[0] = midweights

    def update_inactive_continuity(self):
        for k in range(len(self.curves)):
            if not self.curves[k].active:
                p = self(self.get_t_for_index(k)[0])
                for j in range(self.curves[k].points.shape[0]):
                    self.curves[k].points[j] = p

    @classmethod
    def from_points(cls, points, knots, weights=None, active=True):
        if weights is None:
            weights = np.ones((points.shape[0]))
        curves = []
        for k in range(len(knots) - 1):
            curves.append(
                NurbCurve(
                    points[knots[k] : knots[k + 1] + 1],
                    weights[knots[k] : knots[k + 1] + 1],
                    active=active,
                )
            )
        return cls(*curves, active=active)

    def apply_transform(self, transform):
        for curve in self.curves:
            curve.apply_transform(transform)
        return self

    @classmethod
    def from_curve_chain(cls, curve_chain: CurveChain):
        check = all(
            [isinstance(curve, NurbCurve) for curve in curve_chain.get_curves()]
        )
        if not check:
            raise ValueError("All curves of chain must be NurbCurves")
        else:
            return cls(*curve_chain.get_curves(), active=curve_chain.active)


class CycloidCurve(Curve):
    """Class to represent a cycloid curve as a Curve."""

    def __init__(
        self,
        rb: float = 1,
        rc: float = 1,
        angle: float = 0,
        v_offs=ORIGIN,
        z_offs: float = 0,
        t0=0,
        t1=1,
        active=True,
        enable_vectorize=True,
    ):
        self.rb = rb
        self.rc = rc
        self.angle = angle
        self.v_offs = v_offs
        self.z_offs = z_offs

        super().__init__(
            self.cycloid_func,
            active=active,
            t0=t0,
            t1=t1,
            enable_vectorize=enable_vectorize,
        )

    def cycloid_func(self, t):
        return cycloid_circle(
            t,
            rb=self.rb,
            rc=self.rc,
            angle=self.angle,
            v_offs=self.v_offs,
            z_offs=self.z_offs,
        )


class CycloidConicCurve(Curve):
    """Class to represent a cycloid curve as a Curve."""

    def __init__(
        self,
        rb: float = 1,
        rc: float = 1,
        angle: float = 0,
        C: float = 0.5,
        v_offs=ORIGIN,
        z_offs: float = 0,
        t0=0,
        t1=1,
        active=True,
        enable_vectorize=True,
    ):
        self.rb = rb
        self.rc = rc
        self.angle = angle
        self.v_offs = v_offs
        self.z_offs = z_offs
        self.C = C

        super().__init__(
            self.cycloid_func,
            active=active,
            t0=t0,
            t1=t1,
            enable_vectorize=enable_vectorize,
        )

    @property
    def R(self):
        return 1 / self.C

    def cycloid_func(self, t):
        return cycloid_cone(
            t,
            rb=self.rb,
            rc=self.rc,
            C=self.C,
            angle=self.angle,
            v_offs=self.v_offs,
            z_offs=self.z_offs,
        )
