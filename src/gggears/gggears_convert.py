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

import gggears.gggears_core as gg
import numpy as np
import gggears.curve as crv
from scipy.optimize import minimize
from gggears.defs import *
from gggears.function_generators import *
import dataclasses
from typing import Union, List
import time
import logging


class GearToNurbs:
    """A class to manage the conversion of gear profile curves into NURBS surfaces.

    Parameters
    ----------
    gear : gg.Gear
        The gear object to build.
    n_points_hz : int, optional
        Number of points used for spline approximation for each segment of the 2D gear
        profile that is not a line or an arc. Lines and arcs use exact NURB
        representation with 2 and 3 points, respectively. The default is 4.
    n_points_vert : int, optional
        Number of 2D profile slices used for generating 3D surfaces. The default is 4.
    oversampling_ratio : float, optional
        Ratio of the number of evaluations of analytical functions to the number of
        unknown points in spline approximation. Affects both horizontal points and
        vertical slices. For spline approximation, the endpoints are fixed, so the
        unkown points are the mid-points. Minimum value is 2, the default is 3. When
        fractional, the number of evaluations is rounded up.
        Example: for a 3-point spline and oversampling of 3, the unkown point is the
        middle one, the number of evaluations are the 2 end points + 3 in the middle,
        so 5 in total.
    """

    def __init__(
        self,
        gear: gg.Gear,
        n_points_hz=4,
        n_points_vert=4,
        oversampling_ratio=3,
    ):
        self.gear = gear
        self.z_vals = gear.z_vals
        self.n_points_hz = n_points_hz
        self.n_points_vert = n_points_vert
        self.oversamp_ratio = oversampling_ratio
        self.n_z_tweens = int(
            np.ceil((self.n_points_vert - 2) * self.oversamp_ratio) + 2
        )
        start = time.time()
        self.gear_stacks: List[gg.GearRefProfile] = self.generate_gear_stacks()
        logging.info(f"Gears generated in {time.time()-start:.5f} seconds")
        self.gear_generator_ref = self.gear_stacks[0][0]
        start = time.time()
        self.nurb_profile_stacks = self.generate_nurbs()
        logging.info(f"Nurbs generated in {time.time()-start:.5f} seconds")
        start = time.time()
        self.side_surf_data = self.generate_surface_points_sides()
        logging.info(f"Surfaces generated in {time.time()-start:.5f} seconds")

    def generate_nurbs(self):
        nurb_profile_stacks = []
        for gear_stack_loc in self.gear_stacks:
            nurb_stack = []
            for k in range(len(gear_stack_loc)):
                gearprofile = gear_stack_loc[k]

                NURB_profile = gearprofile_to_nurb(
                    gearprofile, n_points=self.n_points_hz
                )

                nurb_stack.append(NURB_profile)
            nurb_profile_stacks.append(nurb_stack)
        return nurb_profile_stacks

    def generate_gear_stacks(self) -> List[List[gg.GearRefProfileExtended]]:
        gear_stacks = []
        for ii in range(len(self.z_vals) - 1):
            # need more gear slices than nurb points to produce 'best' fit without overfitting
            # oversamp ratio controls how many more
            # the 2 end points will be locked down, the middle points are approximated by fitting
            z_tweens = np.linspace(
                self.z_vals[ii], self.z_vals[ii + 1], self.n_z_tweens
            )
            gear_stack_loc = [
                gg.GearRefProfileExtended.from_refprofile(
                    self.gear.curve_gen_at_z(z), self.gear.cone
                )
                for z in z_tweens
            ]
            gear_stacks.append(gear_stack_loc)
        return gear_stacks

    def generate_surface_points_sides(self):
        surface_data = []
        for ii in range(len(self.z_vals) - 1):
            nurb_profile_stack = self.nurb_profile_stacks[ii]
            stack = []
            for k in range(len(nurb_profile_stack)):
                nurb = crv.NURBSCurve(
                    *[
                        curve
                        for curve in nurb_profile_stack[k]
                        .profile_closed.copy()
                        .get_curves()
                        if curve.active
                    ]
                )
                stack.append(nurb)

            # axis 0: vertical, axis 1: horizontal, axis 2: x-y-z-w

            points_asd = np.stack([nurbs.points for nurbs in stack], axis=0)
            weights_asd = np.stack([nurbs.weights for nurbs in stack], axis=0)
            points_combined = np.concatenate(
                [points_asd, weights_asd[:, :, np.newaxis]], axis=2
            )
            if self.n_z_tweens == 2:
                # if only 2 layers exist, no need to solve for the side-surface
                surface_data.append(
                    NurbSurfaceData(
                        points=points_asd,
                        weights=weights_asd,
                        knots=stack[0].knots[:],
                    )
                )
            else:

                # fast solver assumes t values are evenly distributed
                sol2, points_init, weights_init = self.solve_surface_fast(
                    points_combined, n_points_vert=self.n_points_vert
                )
                init_data = np.concatenate(
                    [points_init, weights_init[:, :, np.newaxis]], axis=2
                )
                # regular solver treats t values as unknowns
                sol2, points2, weights2, t = self.solve_surface(
                    points_combined,
                    n_points_vert=self.n_points_vert,
                    t_weight=0.01,
                    init_points=init_data,
                )

                surface_data.append(
                    NurbSurfaceData(
                        points=points2, weights=weights2, knots=stack[0].knots[:]
                    )
                )

        return surface_data

    def solve_surface(
        self, target_points, n_points_vert=4, t_weight=0.01, init_points=None
    ):

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
            target_mids = target_points[1:-1, :, :]
            target_mids = target_mids.reshape(target_mids.shape[0], -1)
            points_flat = points.reshape(points.shape[0], -1)
            diff = target_mids - bezierdc(t, points_flat)
            deriv_dpoints = -bezier_diff_p(t, points_flat.shape[0])
            deriv_dt = -bezier_diff_t(t, points_flat)
            dp = deriv_dpoints.T @ diff
            dt = np.diag(diff @ deriv_dt.T)
            dp = dp.reshape(points.shape)
            return np.sum(diff * diff), inverse_allocator(dp, dt)

        init_t = np.linspace(0, 1, o)[1:-1]
        if init_points is None:
            init_points = bezierdc(np.linspace(0, 1, m), target_points)

        init_guess_x = inverse_allocator(init_points, init_t)

        sol = minimize(cost_fun, init_guess_x, jac=True)
        points_sol, t = point_allocator(sol.x)
        points_out = points_sol[:, :, :3]
        weights_out = points_sol[:, :, 3]
        logging.debug(
            f"Nurbs point fit stats: n_iter: {sol.nit}, nfev: {sol.nfev}, status: {sol.status}"
        )
        logging.debug(f"Final cost: {sol.fun}")
        logging.debug(f"message: {sol.message}")
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

        def inverse_allocator(points):
            x = np.zeros(((m - 2) * n * 4 + o))
            x[: (m - 2) * n * 4] = points[1:-1, :, :].reshape((m - 2) * n * 4)
            return x

        def cost_fun(x):
            points = point_allocator(x)
            t = np.linspace(0, 1, o)[1:-1]
            target_mids = target_points[1:-1, :, :]
            target_mids = target_mids.reshape(target_mids.shape[0], -1)
            points_flat = points.reshape(points.shape[0], -1)
            diff = target_mids - bezierdc(t, points_flat)
            deriv_dpoints = -bezier_diff_p(t, points_flat.shape[0])
            dp = deriv_dpoints.T @ diff
            dp = dp.reshape(points.shape)
            return np.sum(diff * diff), inverse_allocator(dp)

        init_points = bezierdc(np.linspace(0, 1, m), target_points)
        init_guess_x = inverse_allocator(init_points)

        sol = minimize(cost_fun, init_guess_x, jac=True)
        points_sol = point_allocator(sol.x)
        points_out = points_sol[:, :, :3]
        weights_out = points_sol[:, :, 3]
        logging.debug(
            f"Nurbs point fit stats: n_iter: {sol.nit}, nfev: {sol.nfev}, status: {sol.status}"
        )
        logging.debug(f"Final cost: {sol.fun}")
        logging.debug(f"message: {sol.message}")
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


def gearprofile_to_nurb(gearprofile: gg.GearRefProfile, n_points=4, oversamp_ratio=2):
    tooth_nurb = crv.convert_curve_nurbezier(
        gearprofile.tooth_curve.get_curves(),
        n_points=n_points,
        samp_ratio=oversamp_ratio,
    )

    tooth_mirror_nurb = tooth_nurb.copy()
    tooth_mirror_nurb.points = tooth_mirror_nurb.points * np.array([1, -1, 1])
    tooth_mirror_nurb.reverse()

    if isinstance(gearprofile, gg.GearRefProfileExtended):
        NURB_profile = gg.GearRefProfileExtended(
            ra_curve=crv.convert_curve_nurbezier(gearprofile.ra_curve).apply_transform(
                gearprofile.transform
            ),
            rd_curve=crv.convert_curve_nurbezier(gearprofile.rd_curve).apply_transform(
                gearprofile.transform
            ),
            ro_curve=crv.convert_curve_nurbezier(gearprofile.ro_curve).apply_transform(
                gearprofile.transform
            ),
            tooth_curve=tooth_nurb.apply_transform(gearprofile.transform),
            tooth_curve_mirror=tooth_mirror_nurb.apply_transform(gearprofile.transform),
            pitch_angle=gearprofile.pitch_angle,
            # unity transform since the transform is directly applied to the curves
            transform=gg.GearTransform(),
            ro_connector_0=crv.convert_curve_nurbezier(
                gearprofile.ro_connector_0
            ).apply_transform(gearprofile.transform),
            ro_connector_1=crv.convert_curve_nurbezier(
                gearprofile.ro_connector_1
            ).apply_transform(gearprofile.transform),
            ro_connector_2=crv.convert_curve_nurbezier(
                gearprofile.ro_connector_2
            ).apply_transform(gearprofile.transform),
            rd_connector=crv.convert_curve_nurbezier(
                gearprofile.rd_connector
            ).apply_transform(gearprofile.transform),
            ra_connector=crv.convert_curve_nurbezier(
                gearprofile.ra_connector
            ).apply_transform(gearprofile.transform),
            ro_curve_tooth=crv.convert_curve_nurbezier(
                gearprofile.ro_curve_tooth
            ).apply_transform(gearprofile.transform),
            ro_curve_dedendum=crv.convert_curve_nurbezier(
                gearprofile.ro_curve_dedendum
            ).apply_transform(gearprofile.transform),
            tooth_centerline=crv.convert_curve_nurbezier(
                gearprofile.tooth_centerline
            ).apply_transform(gearprofile.transform),
        )
    elif isinstance(gearprofile, gg.GearRefProfile):
        NURB_profile = gg.GearRefProfile(
            ra_curve=crv.convert_curve_nurbezier(gearprofile.ra_curve).apply_transform(
                gearprofile.transform
            ),
            rd_curve=crv.convert_curve_nurbezier(gearprofile.rd_curve).apply_transform(
                gearprofile.transform
            ),
            ro_curve=crv.convert_curve_nurbezier(gearprofile.ro_curve).apply_transform(
                gearprofile.transform
            ),
            tooth_curve=tooth_nurb.apply_transform(gearprofile.transform),
            tooth_curve_mirror=tooth_mirror_nurb.apply_transform(gearprofile.transform),
            pitch_angle=gearprofile.pitch_angle,
            # unity transform since the transform is directly applied to the curves
            transform=gg.GearTransform(),
        )
    # NURB_profile.transform = gg.GearTransform()

    return NURB_profile
