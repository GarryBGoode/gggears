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
import build123d as bd
from gggears.gggears_convert import *
from gggears.gggears_base_classes import *
from scipy.spatial.transform import Rotation as scp_Rotation
import numpy as np
import time
import logging
import warnings
import copy


class GearBuilder(GearToNurbs):
    """A class for building build123d Part objects from gear profiles.

    The class inherits from GearToNurbs, which is responsible for generating the NURBS
    surface points and weights, this class is responsible for converting to build123d.
    Conversion happens in a reference space, with scaling of 1 (module of 1) and on the
    XY plane, default orientation. A transformation is applied after conversion to
    represent the final part.

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
        n_points_hz: int = 4,
        n_points_vert: int = 4,
        oversampling_ratio: float = 3,
        side_surface_extension_ratio: float = 0.01,
    ):
        if gear.cone.cone_angle == 0:
            # gear construction by creating all outside surfaces
            # and then using them to define a Solid
            super().__init__(
                gear=gear,
                n_points_hz=n_points_hz,
                n_points_vert=n_points_vert,
                oversampling_ratio=oversampling_ratio,
            )
            bot_cover = self.generate_cover(
                self.nurb_profile_stacks[0][0], self.gear_stacks[0][0]
            )
            top_cover = self.generate_cover(
                self.nurb_profile_stacks[-1][-1], self.gear_stacks[-1][-1]
            )
            surfaces = self.gen_side_surfaces()
            if gear.tooth_param.inside_teeth:
                surfaces.append(self.gen_outside_ring())
            surfaces.append(bot_cover)
            surfaces.append(top_cover)

            self.solid = bd.Solid(bd.Shell(surfaces))
        else:
            # gear construction by defining a reference solid (a blank)
            # and cutting it with the side surfaces
            z_vals_save = copy.deepcopy(gear.z_vals)
            zdiff = gear.z_vals[-1] - gear.z_vals[0]
            # extend z_vals by 10% to ensure cutting intersection for cutting (split op.)
            gear.z_vals[-1] += side_surface_extension_ratio * zdiff
            gear.z_vals[0] -= side_surface_extension_ratio * zdiff
            super().__init__(
                gear=gear,
                n_points_hz=n_points_hz,
                n_points_vert=n_points_vert,
                oversampling_ratio=oversampling_ratio,
            )
            # restore original z_vals
            self.gear.z_vals = z_vals_save
            side_surfaces = self.gen_side_surfaces()

            # parameters of ref_solid depend on accurate (original) z_vals
            ref_solid = self.gen_ref_solid()
            # cut ref solid by side_surfaces
            if bd.__version__ > "0.8.0":
                split_result = ref_solid.split(
                    tool=bd.Shell(side_surfaces), keep=bd.Keep.ALL
                ).solids()
            else:
                split_result = ref_solid.split(
                    tool=bd.Shell(side_surfaces), keep=bd.Keep.BOTH
                ).solids()

            if len(split_result) < 2:
                raise RuntimeError(
                    "Split operation of blank solid via gear surfaces failed."
                )
            # the valid result is the one with the smaller volume out of the 2

            split_result.sort(key=lambda x: x.bounding_box().center().Z)
            if self.gear.tooth_param.inside_teeth:
                self.solid = split_result[1]
            else:
                self.solid = split_result[0]

        self.part = bd.Part() + self.solid
        self.part_transformed = bd.BasePartObject(
            apply_transform_part(self.solid, self.gear.transform)
        )

    def gen_ref_solid(self):
        profile0 = self.gear.curve_gen_at_z(self.gear.z_vals[0])
        profile1 = self.gear.curve_gen_at_z(self.gear.z_vals[-1])

        gearcopy = copy.deepcopy(self.gear)
        gearcopy.transform = GearTransform()

        center0, R0 = gearcopy.sphere_data_at_z(self.gear.z_vals[0])
        center1, R1 = gearcopy.sphere_data_at_z(self.gear.z_vals[-1])
        R0 = np.abs(R0)
        R1 = np.abs(R1)

        bottom_angle = (
            180 / PI * gearcopy.shape_recipe(gearcopy.z_vals[0]).transform.angle
        )
        top_angle = (
            180 / PI * gearcopy.shape_recipe(gearcopy.z_vals[-1]).transform.angle
        )
        sph1 = (
            bd.Solid.make_sphere(radius=R0, angle1=-90, angle2=90, angle3=360)
            .rotate(bd.Axis.Z, bottom_angle)
            .translate(center0)
        )
        sph2 = (
            bd.Solid.make_sphere(radius=R1, angle1=-90, angle2=90, angle3=360)
            .rotate(bd.Axis.Z, top_angle)
            .translate(center1)
        )
        ref_solid = (sph1 + sph2) - sph1.intersect(sph2)
        ref_solid = ref_solid

        c_o_0 = profile0.transform(profile0.ro_curve.center)
        c_o_1 = profile1.transform(profile1.ro_curve.center)
        h_o = c_o_1[2] - c_o_0[2]
        r_o_0 = profile0.ro_curve.radius * profile0.transform.scale
        r_o_1 = profile1.ro_curve.radius * profile1.transform.scale
        r_o_cone = bd.Solid.make_cone(
            r_o_0, r_o_1, h_o, plane=(bd.Plane.XY).offset(c_o_0[2])
        )

        if isinstance(ref_solid, bd.ShapeList):
            ref_solid = ref_solid.sort_by_distance(np2v((c_o_0 + c_o_1) / 2))[0]

        if gearcopy.tooth_param.inside_teeth:
            r_o_face = r_o_cone.faces().sort_by(bd.Axis.Z)[1]
            if bd.__version__ > "0.8.0":
                split_result = ref_solid.split(r_o_face, keep=bd.Keep.ALL).solids()
            else:
                split_result = ref_solid.split(r_o_face, keep=bd.Keep.BOTH).solids()
            if bd.__version__ > "0.8.0":
                split_result = ref_solid.split(r_o_face, keep=bd.Keep.ALL).solids()
            else:
                split_result = ref_solid.split(r_o_face, keep=bd.Keep.BOTH).solids()
            split_result.sort(key=lambda x: x.volume)
            ref_solid = split_result[0]

        else:
            ref_solid = ref_solid.fuse(r_o_cone)
            ref_solid = ref_solid.clean()
            ref_solid = ref_solid.split(bd.Plane.XY.offset(c_o_0[2]), keep=bd.Keep.TOP)

        return ref_solid

    def gen_side_surfaces(self):
        n_teeth = self.gear.tooth_param.num_teeth_act
        surfaces = []

        for j in range(n_teeth):
            for k in range(len(self.gear.z_vals) - 1):
                surfdata_z = self.side_surf_data[k]
                patches = [*surfdata_z.get_patches()]
                for patch in patches[:-3]:
                    # shape: vert x horiz x xyz
                    points = patch["points"]
                    weights = patch["weights"]
                    vpoints = [
                        nppoint2Vector(points[k]) for k in range(points.shape[0])
                    ]
                    face = bd.Face.make_bezier_surface(
                        vpoints, weights.tolist()
                    ).rotate(
                        bd.Axis.Z,
                        angle=self.gear.tooth_param.pitch_angle * j * 180 / PI,
                    )
                    surfaces.append(face)

        return surfaces

    def gen_outside_ring(self):
        r_o = -self.gear.shape_recipe.limits.h_o + self.gear.tooth_param.num_teeth / 2
        ring_base = bd.Edge.make_circle(radius=r_o, plane=bd.Plane.XY)

        edge_ring = bd.Line(
            [
                bd.Vector((r_o, 0, self.gear.z_vals[0])),
                bd.Vector((r_o, 0, self.gear.z_vals[-1])),
            ]
        )
        ring_surf = bd.Face.sweep(profile=edge_ring, path=ring_base)
        return ring_surf

    def generate_cover(
        self, nurb_stack: GearRefProfileExtended, gear_stack: GearRefProfileExtended
    ):

        if self.gear.cone.cone_angle != 0:

            if not self.gear.tooth_param.inside_teeth:
                curve = crv.NURBSCurve.from_curve_chain(nurb_stack.tooth_profile_closed)
                splines = gen_splines(curve)

                face_tooth = bd.Face.make_surface(bd.Wire(splines))

                cover_edge_curve = crv.NURBSCurve(
                    nurb_stack.rd_curve, nurb_stack.rd_connector
                )
                cover_edge = bd.Edge() + gen_splines(cover_edge_curve)

                num_teeth = self.gear.tooth_param.num_teeth_act
                cover_edge = cover_edge + [
                    cover_edge.rotate(
                        axis=bd.Axis.Z,
                        angle=(j + 1) * nurb_stack.pitch_angle * 180 / PI,
                    )
                    for j in range(num_teeth - 1)
                ]
                cover_face = bd.Face(bd.Wire(cover_edge))

                out_face = cover_face + [
                    face_tooth.rotate(
                        axis=bd.Axis.Z, angle=j * nurb_stack.pitch_angle * 180 / PI
                    )
                    for j in range(num_teeth)
                ]
                return out_face
            else:
                curve = crv.NURBSCurve.from_curve_chain(nurb_stack.profile_closed)
                splines = gen_splines(curve)
                face_tooth = bd.Face.make_surface(bd.Wire(splines))
                num_teeth = self.gear.tooth_param.num_teeth_act
                out_face = bd.Face.fuse(
                    *[
                        face_tooth.rotate(
                            axis=bd.Axis.Z, angle=j * nurb_stack.pitch_angle * 180 / PI
                        )
                        for j in range(num_teeth)
                    ]
                )
                return out_face
        else:

            num_teeth = self.gear.tooth_param.num_teeth_act
            curve = crv.NURBSCurve.from_curve_chain(nurb_stack.profile)
            curve.del_inactive_curves()
            curve.enforce_continuity()
            splines = gen_splines(curve)
            profile_edge = bd.Wire(splines)
            splines = bd.Edge() + [
                profile_edge.rotate(
                    axis=bd.Axis.Z,
                    angle=nurb_stack.pitch_angle * 180 / PI * j,
                )
                for j in range(num_teeth)
            ]

            if self.gear.tooth_param.inside_teeth:
                r_o = (
                    -self.gear.shape_recipe.limits.h_o
                    + self.gear.tooth_param.num_teeth / 2
                )
                z_val = gear_stack.transform.center[2]
                ring = bd.Edge.make_circle(radius=r_o, plane=bd.Plane.XY.offset(z_val))
                return bd.Face(bd.Wire(ring), inner_wires=[bd.Wire(splines)])
            else:
                return bd.Face(bd.Wire(splines))


class GearBuilder_old(GearToNurbs):
    """A class for building Part objects from gear profiles."""

    def __init__(
        self,
        gear: gg.Gear,
        n_points_hz=4,
        n_points_vert=4,
        oversampling_ratio=2.5,
        add_plug=False,
    ):
        super().__init__(
            gear=gear,
            n_points_hz=n_points_hz,
            n_points_vert=n_points_vert,
            oversampling_ratio=oversampling_ratio,
        )
        surfaces = []
        ro_surfaces = []

        start = time.time()
        for k in range(len(self.gear.z_vals) - 1):
            surfdata_z = self.side_surf_data[k]

            for patch in surfdata_z.get_patches():
                points = patch["points"]
                weights = patch["weights"]
                vpoints = [nppoint2Vector(points[k]) for k in range(points.shape[0])]
                surfaces.append(bd.Face.make_bezier_surface(vpoints, weights.tolist()))
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
        top_surface = bd.Face.make_surface(bd.Wire(splines))

        bot_points, bot_weights = (
            self.side_surf_data[0].points[0, :, :],
            self.side_surf_data[0].weights[0, :],
        )
        bot_curve = crv.NURBSCurve.from_points(
            bot_points, knots=self.side_surf_data[0].knots, weights=bot_weights
        )
        splines = [self.gen_splines(curve) for curve in bot_curve.get_curves()]
        bot_surface = bd.Face.make_surface(bd.Wire(splines))

        if len(ro_surfaces) > 1:
            ro_surface = bd.Face.fuse(*ro_surfaces)
        else:
            ro_surface = ro_surfaces[0]
        ro_spline_top = self.gen_splines(top_curve.get_curves()[-2])
        ro_spline_bot = self.gen_splines(bot_curve.get_curves()[-2])
        surfaces.insert(0, bot_surface)
        surfaces.append(top_surface)
        shell = bd.Shell(surfaces)
        solid1 = bd.Solid(shell)
        solid1 = fix_attempt(solid1)

        logging.log(
            logging.INFO, f"Gear 1-tooth solid build time: {time.time()-start:.5f}"
        )
        start = time.time()

        self.profile_solid = solid1

        n_teeth = self.gear.tooth_param.num_teeth_act
        bin_n_teeth = bin(n_teeth)[2:]
        shape_dict = []
        solid2_to_fuse = []
        angle_construct = 0.0
        angle_idx = 0
        tol = 1e-4

        axis1 = bd.Axis.Z

        for k in range(len(bin_n_teeth)):

            if k == 0:
                shape_dict.append(solid1)
                angle = 0
            else:
                angle = self.gear.tooth_param.pitch_angle * RAD2DEG * (2 ** (k - 1))
                rotshape = shape_dict[k - 1].rotate(axis1, angle)
                fuse_shape = (
                    shape_dict[k - 1].fuse(rotshape, glue=False, tol=tol).clean()
                )
                fuse_shape = fix_attempt(fuse_shape)
                shape_dict.append(fuse_shape)

            if bin_n_teeth[-(k + 1)] == "1":

                angle_construct = (
                    angle_idx * self.gear.tooth_param.pitch_angle * RAD2DEG
                )
                rotshape = shape_dict[k].rotate(axis1, angle_construct)

                solid2_to_fuse.append(rotshape)
                angle_idx = angle_idx + 2**k

        if len(solid2_to_fuse) > 1:
            self.solid = bd.Solid.fuse(*solid2_to_fuse, glue=False, tol=tol).clean()
        else:
            self.solid = solid2_to_fuse[0].clean()

        self.solid = fix_attempt(self.solid)

        plug_surfaces = []
        plug_splines_top = []
        plug_splines_bot = []
        if add_plug:
            for k in range(n_teeth):
                plug_surfaces.append(
                    ro_surface.rotate(
                        axis1, self.gear.tooth_param.pitch_angle * RAD2DEG * k
                    )
                )
                plug_splines_bot.append(
                    ro_spline_bot.rotate(
                        axis1, self.gear.tooth_param.pitch_angle * RAD2DEG * k
                    )
                )
                plug_splines_top.append(
                    ro_spline_top.rotate(
                        axis1, self.gear.tooth_param.pitch_angle * RAD2DEG * k
                    )
                )
            plug_top = bd.Face.make_surface(bd.Wire(plug_splines_top))
            plug_bot = bd.Face.make_surface(bd.Wire(plug_splines_bot))
            plug_surfaces.insert(0, plug_bot)
            plug_surfaces.append(plug_top)
            plug = bd.Solid(bd.Shell(plug_surfaces))
            plug = fix_attempt(plug)
            self.solid = self.solid.fuse(plug).clean()
            self.solid = fix_attempt(self.solid)

        logging.log(
            logging.INFO, f"Gear solid fuse time: {time.time()-start:.5f} seconds"
        )
        self.solid = bd.BasePartObject(self.solid).fix()
        self.solid_transformed = apply_transform_part(self.solid, self.gear.transform)
        self.part_transformed = self.solid_transformed

    def gen_splines(self, curve_bezier: Curve):
        vectors = nppoint2Vector(curve_bezier.points)
        weights = curve_bezier.weights.tolist()
        return bd.Edge.make_bezier(*vectors, weights=weights)


def apply_transform_part(part: bd.Part, transform: GearTransform):
    rot1 = scp_Rotation.from_matrix(transform.orientation)
    # rot1.as_angle_axis()
    degrees = rot1.as_euler("zyx", degrees=True)
    part = part.scale(transform.scale)
    part = bd.Rotation(0, 0, transform.angle * 180 / PI) * part
    part = (
        bd.Rotation(Z=degrees[2], Y=degrees[1], X=degrees[0], ordering=bd.Extrinsic.ZYX)
        * part
    )

    part = part.translate(transform.center)
    return part


def fix_attempt(solid):
    if not solid.is_valid():
        warnings.warn("Invalid solid found", RuntimeWarning, stacklevel=2)
        solid = solid.fix()
    return solid


def nppoint2Vector(p: np.ndarray):
    if p.size == 3:
        return bd.Vector((p[0], p[1], p[2]))
    else:
        return [bd.Vector((p[k, 0], p[k, 1], p[k, 2])) for k in range(p.shape[0])]


def np2v(p: np.ndarray):
    # shorthand for npppoint2Vector
    return nppoint2Vector(p)


def gen_splines(curve_bezier: Curve):
    if isinstance(curve_bezier, NURBSCurve) or isinstance(curve_bezier, CurveChain):
        splines = []
        for curve in curve_bezier.get_curves():
            if curve.active:
                vectors = nppoint2Vector(curve.points)
                weights = curve.weights.tolist()
                splines.append(bd.Edge.make_bezier(*vectors, weights=weights))
        return splines
    else:
        vectors = nppoint2Vector(curve_bezier.points)
        weights = curve_bezier.weights.tolist()
        return bd.Edge.make_bezier(*vectors, weights=weights)


def transform2Location(transform: GearTransform):
    rot1 = scp_Rotation.from_matrix(transform.orientation)
    degrees = rot1.as_euler("zyx", degrees=True)
    loc = bd.Location(
        transform.center,
        [degrees[2] + transform.angle * 180 / PI, degrees[1], degrees[0]],
        bd.Extrinsic.ZYX,
    )

    return loc


def generate_boundary_edges(
    nurbprofile: GearRefProfile,
    transform: GearTransform = None,
    angle_range: float = 2 * PI,
):
    if transform is None:
        # identity transform by default
        transform = GearTransform()
    nurb_profile = gearprofile_to_nurb(nurbprofile)
    # don't want to get more inputs about num of teeth, but without rounding this can
    # lose 1 tooth
    N = int(np.round(angle_range / nurbprofile.pitch_angle))

    curves = []
    for i in range(N):
        # angle = i * profile.pitch_angle
        curves.extend(
            [
                nurb.apply_transform(transform)
                for nurb in nurb_profile.profile.copy().get_curves()
            ]
        )
        transform.angle += nurbprofile.pitch_angle

    nurbs_curve = crv.NURBSCurve(*curves)
    nurbs_curve.enforce_continuity()

    return gen_splines(nurbs_curve)


def arc_to_b123d(arc: crv.ArcCurve) -> bd.Edge:
    """Converts a gggears ArcCurve to a build123d Edge object."""
    if (arc.t_0, arc.t_1) != (0, 1):
        # arc can be extended, better make a new one
        arc2 = crv.ArcCurve.from_2_point_center(arc.p0, arc.p1, arc.center)
    else:
        arc2 = arc

    loc = bd.Location(
        arc2.center,
        [arc2.roll * 180 / PI, arc2.pitch * 180 / PI, arc2.yaw * 180 / PI],
        bd.Intrinsic.XYZ,
    )

    if arc2.angle < 0:
        start = arc2.angle * 180 / PI
        end = 0
    else:
        start = 0
        end = arc2.angle * 180 / PI

    bd_arc = bd.Edge.make_circle(
        radius=arc2.radius,
        plane=bd.Plane(loc),
        start_angle=start,
        end_angle=end,
    )
    return bd_arc


def line_to_b123d(line: crv.LineCurve) -> bd.Edge:
    """Converts a gggears LineCurve to a build123d Edge object."""
    return bd.Edge.make_line(np2v(line.p0), np2v(line.p1))


def curve_to_edges(curve: crv.Curve):
    if isinstance(curve, crv.CurveChain):
        return [curve_to_edges(curve) for curve in curve.get_curves()]
    elif isinstance(curve, crv.NURBSCurve) | isinstance(curve, crv.NurbCurve):
        return gen_splines(curve)
    elif isinstance(curve, crv.ArcCurve):
        return [arc_to_b123d(curve)]
    elif isinstance(curve, crv.LineCurve):
        return [line_to_b123d(curve)]
    elif isinstance(curve, crv.TransformedCurve):
        nurb = crv.convert_curve_nurbezier(curve.target_curve)
        nurb.apply_transform(curve.transform_method)
        return gen_splines(nurb)
    else:
        nurb = crv.convert_curve_nurbezier(curve)
        return gen_splines(nurb)
