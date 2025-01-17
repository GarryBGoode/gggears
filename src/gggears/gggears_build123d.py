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
from gggears.gggears_base_classes import *
from scipy.spatial.transform import Rotation as scp_Rotation
import numpy as np
import time
import logging
import warnings


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
    method : str, optional
        Selector between "fast" and "slow" conversion method for NURBS surfaces. The
        fast method uses evenly distributed t-values in the vertical direction, while
        the slow method considers t values unknowns and solves for them.
        The default is slow.
    projection : bool, optional
        Used in the construction of top and bottom face of bevel (conic) gears. These
        faces are in theory spherical. If True, the faces are projected onto a sphere
        during construction. If False, the faces are constructed via build123d default
        surface generator from boundary splines. Projection can sometimes fail, while
        the default constructor can sometimes result in wavy surface artifacts.
        The default is True.
    """

    def __init__(
        self,
        gear: gg.Gear,
        n_points_hz: int = 4,
        n_points_vert: int = 4,
        oversampling_ratio: float = 3,
        method: str = "slow",
    ):
        if gear.cone.cone_angle == 0:
            super().__init__(
                gear=gear,
                n_points_hz=n_points_hz,
                n_points_vert=n_points_vert,
                oversampling_ratio=oversampling_ratio,
                convertmethod=method,
            )
            bot_cover = self.generate_cover(
                self.nurb_profile_stacks[0][0], self.gear_stacks[0][0]
            )
            top_cover = self.generate_cover(
                self.nurb_profile_stacks[-1][-1], self.gear_stacks[-1][-1]
            )
            # surfaces = self.gen_side_surfaces()
            surfaces = self.gen_side_surfaces_basic()
            if gear.tooth_param.inside_teeth:
                surfaces.append(self.gen_outside_ring())
            surfaces.append(bot_cover)
            surfaces.append(top_cover)

            self.solid = Solid(Shell(surfaces))
        else:
            z_vals_save = copy.deepcopy(gear.z_vals)
            zmid = (gear.z_vals[-1] + gear.z_vals[0]) / 2
            zdiff = gear.z_vals[-1] - gear.z_vals[0]
            # extend z_vals by 10% to ensure cutting intersection for split operation
            gear.z_vals[-1], gear.z_vals[0] = (
                zmid + zdiff * 1.1 / 2,
                zmid - zdiff * 1.1 / 2,
            )
            super().__init__(
                gear=gear,
                n_points_hz=n_points_hz,
                n_points_vert=n_points_vert,
                oversampling_ratio=oversampling_ratio,
                convertmethod=method,
            )
            # restore original z_vals
            self.gear.z_vals = z_vals_save

            side_surfaces = self.gen_side_surfaces_basic()

            ref_solid = self.gen_ref_solid()
            tool = Face.fuse(*side_surfaces)
            split_result = ref_solid.split(tool=Shell(tool), keep=Keep.ALL).solids()
            split_result.sort(key=lambda x: x.volume)
            self.solid = split_result[0]

        self.part = Part() + self.solid
        self.part_transformed = BasePartObject(
            apply_transform_part(self.solid, self.gear.transform)
        )

    def gen_ref_solid(self):
        profile0 = self.gear.curve_gen_at_z(self.gear.z_vals[0])
        profile1 = self.gear.curve_gen_at_z(self.gear.z_vals[-1])

        gamma = self.gear.cone.cone_angle / 2
        R0 = self.gear.tooth_param.num_teeth / 2 / np.sin(gamma)
        h0 = R0 * np.cos(gamma)
        R1 = R0 * self.gear.shape_recipe(self.gear.z_vals[-1]).transform.scale
        center = Vector(0, 0, h0)
        bottom_angle = (
            180 / PI * self.gear.shape_recipe(self.gear.z_vals[0]).transform.angle
        )
        top_angle = (
            180 / PI * self.gear.shape_recipe(self.gear.z_vals[-1]).transform.angle
        )
        ref_solid = Solid.make_sphere(
            radius=R0, angle1=-90, angle2=90, angle3=360
        ).rotate(Axis.Z, bottom_angle) - Solid.make_sphere(
            radius=R1, angle1=-90, angle2=90, angle3=360
        ).rotate(
            Axis.Z, top_angle
        )
        ref_solid = ref_solid.translate(center)

        c_o_0 = profile0.transform(profile0.ro_curve.center)
        c_o_1 = profile1.transform(profile1.ro_curve.center)
        h_o = c_o_1[2] - c_o_0[2]
        r_o_0 = profile0.ro_curve.radius * profile0.transform.scale
        r_o_1 = profile1.ro_curve.radius * profile1.transform.scale
        r_o_cone = Solid.make_cone(r_o_0, r_o_1, h_o, plane=(Plane.XY).offset(c_o_0[2]))

        if self.gear.tooth_param.inside_teeth:
            r_o_face = r_o_cone.faces().sort_by(Axis.Z)[1]
            split_result = ref_solid.split(r_o_face, keep=Keep.ALL).solids()
            split_result.sort(key=lambda x: x.volume)
            ref_solid = split_result[0]

        else:
            ref_solid = ref_solid.fuse(r_o_cone)
            ref_solid = ref_solid.clean()
            ref_solid = ref_solid.split(Plane.XY.offset(c_o_0[2]), keep=Keep.TOP)

        return ref_solid

    def gen_side_surfaces_basic(self):
        n_teeth = self.gear.tooth_param.num_teeth_act
        surfaces = []

        for j in range(n_teeth):
            for k in range(len(self.gear.z_vals) - 1):
                surfdata_z = self.side_surf_data[k]
                patches = [*surfdata_z.get_patches()]
                for patch in patches[:-3]:
                    # for patch in patches:
                    # shape: vert x horiz x xyz
                    points = patch["points"]
                    weights = patch["weights"]
                    vpoints = [
                        nppoint2Vector(points[k]) for k in range(points.shape[0])
                    ]
                    face = Face.make_bezier_surface(vpoints, weights.tolist()).rotate(
                        Axis.Z,
                        angle=self.gear.tooth_param.pitch_angle * j * 180 / PI,
                    )
                    surfaces.append(face)

        return surfaces

    def gen_outside_ring(self):
        r_o = -self.gear.shape_recipe.limits.h_o + self.gear.tooth_param.num_teeth / 2
        ring_base = Edge.make_circle(radius=r_o, plane=Plane.XY)

        edge_ring = Line(
            [
                Vector((r_o, 0, self.gear.z_vals[0])),
                Vector((r_o, 0, self.gear.z_vals[-1])),
            ]
        )
        ring_surf = Face.sweep(profile=edge_ring, path=ring_base)
        return ring_surf

    def generate_cover(
        self, nurb_stack: GearRefProfileExtended, gear_stack: GearRefProfileExtended
    ):

        if self.gear.cone.cone_angle != 0:

            if not self.gear.tooth_param.inside_teeth:
                curve = crv.NURBSCurve.from_curve_chain(nurb_stack.tooth_profile_closed)
                splines = gen_splines(curve)

                if self.projection:
                    # center_sph = self.gear.center_sphere / self.gear.module
                    # R0 = self.gear.cone.R
                    # R1 = R0 * nurb_stack.transform.scale
                    center_sph = self.gear.shape_recipe(0).cone.center
                    R0 = self.gear.shape_recipe(0).cone.R
                    R1 = R0 * nurb_stack.transform.scale

                    sphere = Sphere(
                        radius=R1,
                        arc_size1=-90,
                        arc_size2=0,
                        arc_size3=360,
                        align=(Align.CENTER, Align.CENTER, Align.MAX),
                        rotation=Rotation(0, 0, 90),
                    ).translate(Vector((0, 0, center_sph[2])))

                    # this projetion operation is not robust,
                    # it failed for some gear shapes on github runner while I was unable
                    # to reproduce the error locally
                    try:
                        wire_proj = Wire.project_to_shape(
                            Wire(splines), sphere, center=nppoint2Vector(center_sph)
                        )
                        face_tooth = Face.make_surface(wire_proj[0])
                    except RuntimeError:
                        # if projection fails, use the original face
                        # sometimes this results in weird wavy shapes
                        face_tooth = Face.make_surface(Wire(splines))
                        warnings.warn(
                            "Spherical projection failed, attempting default surface "
                            "generation",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                else:
                    face_tooth = Face.make_surface(Wire(splines))

                cover_edge_curve = crv.NURBSCurve(
                    nurb_stack.rd_curve, nurb_stack.rd_connector
                )
                cover_edge = Edge() + gen_splines(cover_edge_curve)

                num_teeth = self.gear.tooth_param.num_teeth_act
                cover_edge = cover_edge + [
                    cover_edge.rotate(
                        axis=Axis.Z, angle=(j + 1) * nurb_stack.pitch_angle * 180 / PI
                    )
                    for j in range(num_teeth - 1)
                ]
                cover_face = Face(Wire(cover_edge))

                out_face = cover_face + [
                    face_tooth.rotate(
                        axis=Axis.Z, angle=j * nurb_stack.pitch_angle * 180 / PI
                    )
                    for j in range(num_teeth)
                ]
                return out_face
            else:
                curve = crv.NURBSCurve.from_curve_chain(nurb_stack.profile_closed)
                splines = gen_splines(curve)
                face_tooth = Face.make_surface(Wire(splines))
                num_teeth = self.gear.tooth_param.num_teeth_act
                out_face = Face.fuse(
                    *[
                        face_tooth.rotate(
                            axis=Axis.Z, angle=j * nurb_stack.pitch_angle * 180 / PI
                        )
                        for j in range(num_teeth)
                    ]
                )
                return out_face
        else:

            num_teeth = self.gear.tooth_param.num_teeth_act
            curve = crv.NURBSCurve.from_curve_chain(nurb_stack.profile)
            curve.del_inactive_curves()
            profile_edge = Edge() + gen_splines(curve)
            splines = Edge() + [
                profile_edge.rotate(
                    axis=Axis.Z,
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
                ring = Edge.make_circle(radius=r_o, plane=Plane.XY.offset(z_val))
                return Face(Wire(ring), inner_wires=[Wire(splines)])
            else:
                return Face(Wire(splines))


class GearBuilder_old(GearToNurbs):
    """A class for building Part objects from gear profiles."""

    def __init__(
        self,
        gear: gg.Gear,
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

        start = time.time()
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

        axis1 = Axis.Z

        for k in range(len(bin_n_teeth)):

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
                fuse_shape = (
                    shape_dict[k - 1].fuse(rotshape, glue=False, tol=tol).clean()
                )
                fuse_shape = fix_attempt(fuse_shape)
                shape_dict.append(fuse_shape)

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

        self.solid = fix_attempt(self.solid)

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
            plug = fix_attempt(plug)
            self.solid = self.solid.fuse(plug).clean()
            self.solid = fix_attempt(self.solid)

        logging.log(
            logging.INFO, f"Gear solid fuse time: {time.time()-start:.5f} seconds"
        )
        self.solid = Part(self.solid).fix()
        self.solid_transformed = apply_transform_part(self.solid, self.gear.transform)
        self.part_transformed = self.solid_transformed

    def gen_splines(self, curve_bezier: Curve):
        vectors = nppoint2Vector(curve_bezier.points)
        weights = curve_bezier.weights.tolist()
        return Edge.make_bezier(*vectors, weights=weights)


def apply_transform_part(part: Part, transform: GearTransform):
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


def fix_attempt(solid):
    if not solid.is_valid():
        warnings.warn("Invalid solid found", RuntimeWarning, stacklevel=2)
        solid = solid.fix()
    return solid


def nppoint2Vector(p: np.ndarray):
    if p.size == 3:
        return Vector((p[0], p[1], p[2]))
    else:
        return [Vector((p[k, 0], p[k, 1], p[k, 2])) for k in range(p.shape[0])]


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
                splines.append(Edge.make_bezier(*vectors, weights=weights))
        return splines
    else:
        vectors = nppoint2Vector(curve_bezier.points)
        weights = curve_bezier.weights.tolist()
        return Edge.make_bezier(*vectors, weights=weights)


def transform2Location(transform: GearTransform):
    rot1 = scp_Rotation.from_matrix(transform.orientation)
    degrees = rot1.as_euler("zyx", degrees=True)
    loc = Location(
        transform.center,
        [degrees[2] + transform.angle * 180 / PI, degrees[1], degrees[0]],
        Extrinsic.ZYX,
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


def arc_to_b123d(arc: crv.ArcCurve) -> Edge:
    """Converts a gggears ArcCurve to a build123d Edge object."""
    loc = Location(
        arc.center,
        [arc.roll * 180 / PI, arc.pitch * 180 / PI, arc.yaw * 180 / PI],
        Intrinsic.XYZ,
    )

    return Edge.make_circle(
        radius=arc.radius,
        plane=Plane(loc),
        start_angle=0,
        end_angle=arc.angle * 180 / PI,
    )


def line_to_b123d(line: crv.LineCurve) -> Edge:
    """Converts a gggears LineCurve to a build123d Edge object."""
    return Edge.make_line(np2v(line.p0), np2v(line.p1))


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
