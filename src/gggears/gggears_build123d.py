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
        projection: bool = True,
    ):
        super().__init__(
            gear=gear,
            n_points_hz=n_points_hz,
            n_points_vert=n_points_vert,
            oversampling_ratio=oversampling_ratio,
            convertmethod=method,
        )
        self.projection = projection
        bot_cover = self.generate_cover(self.nurb_profile_stacks[0][0])
        top_cover = self.generate_cover(self.nurb_profile_stacks[-1][-1])
        side_surfaces = self.gen_side_surfaces()
        full_surfaces = side_surfaces
        full_surfaces.append(top_cover)
        full_surfaces.append(bot_cover)

        self.solid = Solid(Shell(full_surfaces))
        self.part = Part() + self.solid
        self.part_transformed = BasePartObject(
            apply_transform_part(self.solid, self.gear.transform)
        )

    def gen_side_surfaces(self):
        n_teeth = self.gear.tooth_param.num_teeth_act
        surfaces = []

        tooth_surfaces_nz = []
        for k in range(len(self.gear.z_vals) - 1):
            surfdata_z = self.side_surf_data[k]
            patches = [*surfdata_z.get_patches()]
            surfaces_z = []
            for patch in patches:
                # for patch in patches:
                # shape: vert x horiz x xyz
                points = patch["points"]
                weights = patch["weights"]
                vpoints = [nppoint2Vector(points[k]) for k in range(points.shape[0])]
                face = Face.make_bezier_surface(vpoints, weights.tolist())

                surfaces_z.append(face)
            tooth_surfaces_nz.append(surfaces_z)

        tooth_surfaces = []
        for j in range(len(patches)):
            loc_face = Face()
            for k in range(len(self.gear.z_vals) - 1):
                loc_face = loc_face + tooth_surfaces_nz[k][j]
            tooth_surfaces.append(loc_face)

        if not self.gear.tooth_param.inside_teeth:
            # fuse tooth surfaces into 1 object
            # last 3 surface elements are closing the tooth which is not needed here
            tooth_surface = Face.fuse(*tooth_surfaces[:-3])
            for j in range(n_teeth):
                tooth_surface_rot = tooth_surface.rotate(
                    Axis.Z,
                    angle=self.gear.tooth_param.pitch_angle * j * 180 / PI,
                )
                surfaces.append(tooth_surface_rot)
            return surfaces

        else:
            # inside ring gears
            if self.gear.cone.cone_angle == 0:
                # spur inside ring gear

                # fuse tooth surfaces into 1 object
                tooth_surface = Face.fuse(*tooth_surfaces[:-3])
                for j in range(n_teeth):
                    tooth_surface_rot = tooth_surface.rotate(
                        Axis.Z,
                        angle=self.gear.tooth_param.pitch_angle * j * 180 / PI,
                    )
                    surfaces.append(tooth_surface_rot)

                r_o = (
                    -self.gear.shape_recipe.limits.h_o
                    + self.gear.tooth_param.num_teeth / 2
                )
                ring_base = Circle(radius=r_o).edge()

                edge_ring = Line(
                    [Vector((r_o, 0, 0)), Vector((r_o, 0, self.gear.z_vals[-1]))]
                )
                ring_surf = Face.sweep(profile=edge_ring, path=ring_base)

                surfaces.append(ring_surf)
                return surfaces
            else:
                # conic inside ring gear
                tooth_surface = Face.fuse(*tooth_surfaces[:-1])
                for j in range(n_teeth):
                    tooth_surface_rot = tooth_surface.rotate(
                        Axis.Z,
                        angle=self.gear.tooth_param.pitch_angle * j * 180 / PI,
                    )
                    surfaces.append(tooth_surface_rot)
                return surfaces

    def generate_cover(self, nurb_stack: GearRefProfileExtended):

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
                ring = (
                    Circle(radius=r_o)
                    .translate(Vector(0, 0, nurb_stack.transform.center[2]))
                    .edge()
                )
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
