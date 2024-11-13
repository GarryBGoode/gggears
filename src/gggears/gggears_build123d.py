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

from gggears.gggears_core import *
from gggears.function_generators import *
from gggears.curve import *
from build123d import *
from gggears.gggears_convert import *
from scipy.spatial.transform import Rotation as scp_Rotation
import numpy as np
import time


def nppoint2Vector(p: np.ndarray):
    if p.size == 3:
        return Vector((p[0], p[1], p[2]))
    else:
        return [Vector((p[k, 0], p[k, 1], p[k, 2])) for k in range(p.shape[0])]


class GearBuilder(GearToNurbs):
    def __init__(
        self,
        gear: gg.InvoluteGear,
        n_points_hz=4,
        n_points_vert=4,
        oversampling_ratio=2.5,
        add_plug=False,
        method="fast",
    ):
        super().__init__(
            gear=gear,
            n_points_hz=n_points_hz,
            n_points_vert=n_points_vert,
            oversampling_ratio=oversampling_ratio,
            convertmethod=method,
        )
        surfaces = []
        ro_surfaces = []
        for k in range(len(self.gear.z_vals) - 1):
            surfdata_z = self.side_surf_data[k]

            for patch in surfdata_z.get_patches():
                points = patch["points"]
                weights = patch["weights"]
                vpoints = [nppoint2Vector(points[k]) for k in range(points.shape[0])]
                surfaces.append(Face.make_bezier_surface(vpoints, weights.tolist()))
            ro_surfaces.append(surfaces[-2])
        self.surfaces = surfaces
        top_points, top_weights = (
            self.side_surf_data[-1].points[-1, :, :],
            self.side_surf_data[-1].weights[-1, :],
        )
        top_curve = crv.NURBSCurve.from_points(
            top_points, knots=self.side_surf_data[-1].knots, weights=top_weights
        )
        splines = [self.gen_splines(curve) for curve in top_curve.get_curves()]
        top_surface = Face.make_surface(Wire(splines))

        bot_points, bot_weights = (
            self.side_surf_data[0].points[0, :, :],
            self.side_surf_data[0].weights[0, :],
        )
        bot_curve = crv.NURBSCurve.from_points(
            bot_points, knots=self.side_surf_data[0].knots, weights=bot_weights
        )
        splines = [self.gen_splines(curve) for curve in bot_curve.get_curves()]
        bot_surface = Face.make_surface(Wire(splines))

        if len(ro_surfaces) > 1:
            ro_surface = Face.fuse(*ro_surfaces)
        else:
            ro_surface = ro_surfaces[0]
        ro_spline_top = self.gen_splines(top_curve.get_curves()[-2])
        ro_spline_bot = self.gen_splines(bot_curve.get_curves()[-2])
        surfaces.insert(0, bot_surface)
        surfaces.append(top_surface)
        shell = Shell(surfaces)
        solid1 = Solid(shell)

        if not solid1.is_valid():
            Warning("Tooth profile solid is not valid")

        self.profile_solid = solid1

        n_teeth = self.gear.tooth_param.num_teeth_act
        bin_n_teeth = bin(n_teeth)[2:]
        shape_dict = []
        solid2_to_fuse = []
        angle_construct = 0.0
        angle_idx = 0
        tol = 1e-6
        # axis1 = Axis(
        #     (0, 0, 0),
        #     (
        #         self.gear.transform.z_axis[0],
        #         self.gear.transform.z_axis[1],
        #         self.gear.transform.z_axis[2],
        #     ),
        # )
        axis1 = Axis.Z
        start = time.time()
        print("starting fusion")
        for k in range(len(bin_n_teeth)):
            print(f"fusion step: {k}")
            if k == 0:
                shape_dict.append(solid1)
                angle = 0
            else:
                angle = self.gear.tooth_param.pitch_angle * RAD2DEG * (2 ** (k - 1))
                rotshape = (
                    shape_dict[k - 1]
                    # .translate(nppoint2Vector(-self.gear.transform.center))
                    .rotate(axis1, angle)
                    # .translate(nppoint2Vector(self.gear.transform.center))
                )
                shape_dict.append(shape_dict[k - 1].fuse(rotshape, glue=False, tol=tol))

            if bin_n_teeth[-(k + 1)] == "1":

                angle_construct = (
                    angle_idx * self.gear.tooth_param.pitch_angle * RAD2DEG
                )
                rotshape = (
                    shape_dict[k]
                    # .translate(nppoint2Vector(-self.gear.transform.center))
                    .rotate(axis1, angle_construct)
                    # .translate(nppoint2Vector(self.gear.transform.center))
                )

                solid2_to_fuse.append(rotshape)
                angle_idx = angle_idx + 2**k

        if len(solid2_to_fuse) > 1:
            self.solid = Solid.fuse(*solid2_to_fuse, glue=False, tol=tol).clean()
        else:
            self.solid = solid2_to_fuse[0].clean()

        plug_surfaces = []
        plug_splines_top = []
        plug_splines_bot = []
        if add_plug:
            for k in range(n_teeth):
                plug_surfaces.append(
                    ro_surface
                    # .translate(nppoint2Vector(-self.gear.transform.center))
                    .rotate(axis1, self.gear.tooth_param.pitch_angle * RAD2DEG * k)
                    # .translate(nppoint2Vector(self.gear.transform.center))
                )
                plug_splines_bot.append(
                    ro_spline_bot
                    # .translate(nppoint2Vector(-self.gear.transform.center))
                    .rotate(axis1, self.gear.tooth_param.pitch_angle * RAD2DEG * k)
                    # .translate(nppoint2Vector(self.gear.transform.center))
                )
                plug_splines_top.append(
                    ro_spline_top
                    # .translate(nppoint2Vector(-self.gear.transform.center))
                    .rotate(axis1, self.gear.tooth_param.pitch_angle * RAD2DEG * k)
                    # .translate(nppoint2Vector(self.gear.transform.center))
                )
            plug_top = Face.make_surface(Wire(plug_splines_top))
            plug_bot = Face.make_surface(Wire(plug_splines_bot))
            plug_surfaces.insert(0, plug_bot)
            plug_surfaces.append(plug_top)
            plug = Solid(Shell(plug_surfaces))
            self.solid = self.solid.fuse(plug).clean()

        print(f"fuse time: {time.time()-start}")
        self.solid = Part(self.solid)
        # rot1 = scp_Rotation.from_matrix(self.gear.transform.orientation)

        # degrees = rot1.as_euler("xyz", degrees=True)
        # self.solid = self.solid.scale(self.gear.transform.scale)
        # self.solid = (
        #     Rotation(degrees[0], degrees[1], degrees[2])
        #     * Rotation(0, 0, self.gear.transform.angle * 180 / PI)
        #     * self.solid
        # )
        # self.solid = self.solid.translate(self.gear.transform.center)

    def gen_splines(self, curve_bezier: Curve):
        vectors = nppoint2Vector(curve_bezier.points)
        weights = curve_bezier.weights.tolist()
        return Edge.make_bezier(*vectors, weights=weights)


def apply_transform_part(part: Part, transform: GearBaseTransform):
    rot1 = scp_Rotation.from_matrix(transform.orientation)
    degrees = rot1.as_euler("zyx", degrees=True)
    part = part.scale(transform.scale)
    part = Rotation(0, 0, transform.angle * 180 / PI) * part
    part = (
        Rotation(Z=degrees[2], Y=degrees[1], X=degrees[0], ordering=Extrinsic.ZYX)
        * part
    )

    part = part.translate(transform.center)
    return part
