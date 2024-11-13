"""
Copyright 2024 Gergely Bencsik
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import gggears.gggears_core as gg
import numpy as np
import gggears.curve as crv
from scipy.optimize import minimize
from gggears.defs import *
from gggears.function_generators import bezierdc
import dataclasses
from typing import Union, List


class GearToNurbs:
    def __init__(
        self,
        gear: gg.InvoluteGear,
        n_points_hz=4,
        n_points_vert=4,
        oversampling_ratio=2.5,
        convertmethod="fast",
    ):
        # self.params = gear.params
        self.gear = gear
        self.z_vals = gear.z_vals
        self.n_points_hz = n_points_hz
        self.n_points_vert = n_points_vert
        self.oversamp_ratio = oversampling_ratio
        self.n_z_tweens = int(
            np.ceil((self.n_points_vert - 2) * self.oversamp_ratio) + 2
        )
        self.gear_stacks: List[gg.GearRefProfile] = self.generate_gear_stacks()
        self.gear_generator_ref = self.gear_stacks[0][0]
        self.nurb_profile_stacks = self.generate_nurbs()
        self.side_surf_data = self.generate_surface_points_sides(method=convertmethod)

    def generate_nurbs(self):
        nurb_profile_stacks = []
        for gear_stack_loc in self.gear_stacks:
            nurb_stack = []
            for gearprofile in gear_stack_loc:
                closed_profile_curve = gg.generate_profile_closed(
                    gearprofile, self.gear.shape_param(0).cone
                )
                rd_nurb = crv.convert_curve_nurbezier(gearprofile.rd_curve)
                tooth_nurb = crv.convert_curve_nurbezier(
                    gearprofile.tooth_curve,
                    n_points=self.n_points_hz,
                    samp_ratio=self.oversamp_ratio,
                )
                ra_nurb = crv.convert_curve_nurbezier(gearprofile.ra_curve)
                ro_nurb = crv.convert_curve_nurbezier(gearprofile.ro_curve)
                ro_nurb.reverse()
                ro_connector_1_nurb = crv.convert_curve_nurbezier(
                    closed_profile_curve[-3]
                )
                ro_connector_0_nurb = crv.convert_curve_nurbezier(
                    closed_profile_curve[-1]
                )

                tooth_mirror_nurb = tooth_nurb.copy()
                tooth_mirror_nurb.points = tooth_mirror_nurb.points * np.array(
                    [1, -1, 1]
                )
                tooth_mirror_nurb.reverse()
                # tooth_mirror_nurb.update_lengths()
                curve_list = [
                    rd_nurb,
                    *tooth_nurb.get_curves(),
                    ra_nurb,
                    *tooth_mirror_nurb.get_curves(),
                    ro_connector_1_nurb,
                    ro_nurb,
                    ro_connector_0_nurb,
                ]

                NurbsConv = crv.NURBSCurve(
                    *[curve for curve in curve_list if curve.active]
                )

                NurbsConv.enforce_continuity()
                for nurb in NurbsConv:
                    nurb.points = gearprofile.transform(nurb.points)
                nurb_stack.append(NurbsConv)
            nurb_profile_stacks.append(nurb_stack)
        return nurb_profile_stacks

    def generate_gear_stacks(self):
        gear_stacks = []
        for ii in range(len(self.z_vals) - 1):
            # need more gear slices than nurb points to produce 'best' fit without overfitting
            # oversamp ratio controls how many more
            # the 2 end points will be locked down, the middle points are approximated by fitting
            z_tweens = np.linspace(
                self.z_vals[ii], self.z_vals[ii + 1], self.n_z_tweens
            )
            gear_stack_loc = [self.gear.curve_gen_at_z(z) for z in z_tweens]
            gear_stacks.append(gear_stack_loc)
        return gear_stacks

    def generate_surface_points_sides(self, method="fast"):
        surface_data = []
        for ii in range(len(self.z_vals) - 1):
            # axis 0: vertical, axis 1: horizontal, axis 2: x-y-z-w
            stack = self.nurb_profile_stacks[ii]
            points_asd = np.stack([nurbs.points for nurbs in stack], axis=0)
            weights_asd = np.stack([nurbs.weights for nurbs in stack], axis=0)
            points_combined = np.concatenate(
                [points_asd, weights_asd[:, :, np.newaxis]], axis=2
            )
            if method == "fast":
                sol2, points2, weights2 = self.solve_surface_fast(
                    points_combined, n_points_vert=self.n_points_vert
                )
            else:
                sol2, points2, weights2, t = self.solve_surface(
                    points_combined, n_points_vert=self.n_points_vert, t_weight=0.01
                )

            surface_data.append(
                NurbSurfaceData(
                    points=points2, weights=weights2, knots=stack[0].knots[:]
                )
            )

        return surface_data

    def solve_surface(self, target_points, n_points_vert=4, t_weight=0.01):

        n = target_points.shape[1]
        m = n_points_vert
        o = target_points.shape[0]

        def point_allocator(x):
            points = np.zeros((m, n, 4))
            points[0, :, :] = target_points[0, :, :]
            points[-1, :, :] = target_points[-1, :, :]

            points[1:-1, :, :] = x[: (m - 2) * n * 4].reshape(((m - 2), n, 4))
            t = x[(m - 2) * n * 4 : (m - 2) * n * 4 + o - 2]
            return points, t

        def inverse_allocator(points, t):
            x = np.zeros(((m - 2) * n * 4 + o))
            x[: (m - 2) * n * 4] = points[1:-1, :, :].reshape((m - 2) * n * 4)
            x[(m - 2) * n * 4 : (m - 2) * n * 4 + o - 2] = t
            return x

        def cost_fun(x):
            points, t = point_allocator(x)
            tref = np.linspace(0, 1, o)[1:-1]
            diff = bezierdc(t, points) - target_points[1:-1, :, :]
            t_diff = (t - tref) * diff.size / t.size * t_weight

            return np.sum(diff * diff) + np.dot(t_diff, t_diff)

        init_t = np.linspace(0, 1, o)[1:-1]
        init_points = bezierdc(np.linspace(0, 1, m), target_points)
        init_guess_x = inverse_allocator(init_points, init_t)

        sol = minimize(cost_fun, init_guess_x)
        points_sol, t = point_allocator(sol.x)
        points_out = points_sol[:, :, :3]
        weights_out = points_sol[:, :, 3]
        return sol, points_out, weights_out, t

    def solve_surface_fast(self, target_points, n_points_vert=4):

        n = target_points.shape[1]
        m = n_points_vert
        o = target_points.shape[0]

        def point_allocator(x):
            points = np.zeros((m, n, 4))
            points[0, :, :] = target_points[0, :, :]
            points[-1, :, :] = target_points[-1, :, :]
            points[1:-1, :, :] = x[: (m - 2) * n * 4].reshape(((m - 2), n, 4))
            return points

        def inverse_allocator(points, t):
            x = np.zeros(((m - 2) * n * 4 + o))
            x[: (m - 2) * n * 4] = points[1:-1, :, :].reshape((m - 2) * n * 4)
            return x

        def cost_fun(x):
            points = point_allocator(x)
            tref = np.linspace(0, 1, o)[1:-1]
            return np.sum((bezierdc(tref, points) - target_points[1:-1, :, :]) ** 2)

        init_t = np.linspace(0, 1, o)[1:-1]
        init_points = bezierdc(np.linspace(0, 1, m), target_points)
        init_guess_x = inverse_allocator(init_points, init_t)

        sol = minimize(cost_fun, init_guess_x)
        points_sol = point_allocator(sol.x)
        points_out = points_sol[:, :, :3]
        weights_out = points_sol[:, :, 3]
        return sol, points_out, weights_out


@dataclasses.dataclass
class NurbSurfaceData:
    """
    Dataclass for storing surface data of b-spline strips.
    """

    points: np.ndarray
    weights: np.ndarray
    knots: np.ndarray
    n_points_vert: int = 4

    def get_patches(self):
        for ui in range(len(self.knots) - 1):

            u0 = self.knots[ui]
            u1 = self.knots[ui + 1]
            points = self.points[:, u0 : u1 + 1, :]
            weights = self.weights[:, u0 : u1 + 1]
            yield {"points": points, "weights": weights}


@dataclasses.dataclass
class NurbSurfaceData:
    """
    Dataclass for storing surface data of b-spline strips.
    """

    points: np.ndarray
    weights: np.ndarray
    knots: np.ndarray
    n_points_vert: int = 4

    def get_patches(self):
        for ui in range(len(self.knots) - 1):

            u0 = self.knots[ui]
            u1 = self.knots[ui + 1]
            points = self.points[:, u0 : u1 + 1, :]
            weights = self.weights[:, u0 : u1 + 1]
            yield {"points": points, "weights": weights}
