'''
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
'''

from gggears.gggears_core import *
from gggears.function_generators import *
from gggears.curve import *
from build123d import *
from gggears.gggears_convert import *
import numpy as np
import time
from scipy.optimize import root
from ocp_vscode import show, set_port
set_port(3939)


def nppoint2Vector(p: np.ndarray):
    if p.size==3:
        return Vector((p[0],p[1],p[2]))
    else:
        return [Vector((p[k,0],p[k,1],p[k,2])) for k in range(p.shape[0])]


class GearBuilder(GearToNurbs):
    def __init__(self,
                 gear:gg.InvoluteGear,
                 n_points_hz=4,
                 n_points_vert=4,
                 oversampling_ratio=2.5,
                 add_plug=False,
                 method='fast'):
        super().__init__(gear=gear,
                         n_points_hz=n_points_hz,
                         n_points_vert=n_points_vert,
                         oversampling_ratio=oversampling_ratio,
                         convertmethod=method)
        surfaces = []
        ro_surfaces = []
        for k in range(len(self.params.z_vals)-1):
            surfdata_z = self.side_surf_data[k]

            for patch in surfdata_z.get_patches():
                points = patch['points']
                weights = patch['weights']
                vpoints = [nppoint2Vector(points[k]) for k in range(points.shape[0])]
                surfaces.append(Face.make_bezier_surface(vpoints,weights.tolist()))
            ro_surfaces.append(surfaces[-2])
        self.surfaces = surfaces
        top_points,  top_weights = self.side_surf_data[-1].points[-1,:,:], self.side_surf_data[-1].weights[-1,:]
        top_curve = crv.NURBSCurve.from_points(top_points,knots=self.side_surf_data[-1].knots,weights=top_weights)
        splines = [self.gen_splines(curve) for curve in top_curve.get_curves()]
        top_surface = Face.make_surface(Wire(splines))

        bot_points,  bot_weights = self.side_surf_data[0].points[0,:,:], self.side_surf_data[0].weights[0,:]
        bot_curve = crv.NURBSCurve.from_points(bot_points,knots=self.side_surf_data[0].knots,weights=bot_weights)
        splines = [self.gen_splines(curve) for curve in bot_curve.get_curves()]
        bot_surface = Face.make_surface(Wire(splines))

        if len(ro_surfaces)>1:
            ro_surface = Face.fuse(*ro_surfaces)
        else:
            ro_surface = ro_surfaces[0]
        ro_spline_top = self.gen_splines(top_curve.get_curves()[-2])
        ro_spline_bot = self.gen_splines(bot_curve.get_curves()[-2])
        surfaces.insert(0,bot_surface)
        surfaces.append(top_surface)
        shell = Shell(surfaces)
        solid1 = Solid(shell)

        if not solid1.is_valid():
            Warning("Tooth profile solid is not valid")

        self.profile_solid = solid1

        n_teeth = int(np.floor(self.gear.params.n_teeth-self.gear.params.n_cutout_teeth))
        bin_n_teeth = bin(n_teeth)[2:]
        shape_dict = []
        solid2_to_fuse = []
        angle_construct = 0.0
        angle_idx = 0
        tol = 1E-6
        axis1 = Axis((0,0,0),(self.params.axis[0],self.params.axis[1],self.params.axis[2]))
        start=time.time()
        print("starting fusion")
        for k in range(len(bin_n_teeth)):
            print(f"fusion step: {k}")
            if k==0:
                shape_dict.append(solid1)
                angle=0
            else:
                angle = self.gear.params.pitch_angle*RAD2DEG*(2**(k-1))
                rotshape = shape_dict[k-1].translate(nppoint2Vector(-self.params.center(0))
                                                     ).rotate(axis1,angle).translate(
                                                         nppoint2Vector(self.params.center(0)))
                shape_dict.append(shape_dict[k-1].fuse(rotshape,glue=False,tol=tol))

            if bin_n_teeth[-(k+1)]=='1':


                angle_construct = angle_idx * self.gear.params.pitch_angle*RAD2DEG
                rotshape = shape_dict[k].translate(nppoint2Vector(-self.params.center(0))
                                                     ).rotate(axis1,angle_construct).translate(
                                                         nppoint2Vector(self.params.center(0)))

                solid2_to_fuse.append(rotshape)
                angle_idx = angle_idx+2**k

        if len(solid2_to_fuse)>1:
            self.solid = Solid.fuse(*solid2_to_fuse, glue=False,tol=tol).clean()
        else:
            self.solid = solid2_to_fuse[0].clean()

        plug_surfaces = []
        plug_splines_top = []
        plug_splines_bot = []
        if add_plug:
            for k in range(n_teeth):
                plug_surfaces.append(ro_surface.rotate(axis1,self.params.pitch_angle*RAD2DEG*k).translate(nppoint2Vector(self.params.center(0))))
                plug_splines_bot.append(ro_spline_bot.rotate(axis1,self.params.pitch_angle*RAD2DEG*k))
                plug_splines_top.append(ro_spline_top.rotate(axis1,self.params.pitch_angle*RAD2DEG*k))
            plug_top = Face.make_surface(Wire(plug_splines_top))
            plug_bot = Face.make_surface(Wire(plug_splines_bot))
            plug_surfaces.insert(0,plug_bot)
            plug_surfaces.append(plug_top)
            plug = Solid(Shell(plug_surfaces))
            self.solid = self.solid.fuse(plug).clean()

        print(f"fuse time: {time.time()-start}")

    def gen_splines(self,curve_bezier:Curve):
        vectors = nppoint2Vector(curve_bezier.points)
        weights = curve_bezier.weights.tolist()
        return Edge.make_bezier(*vectors,weights=weights)




if __name__ == "__main__":
    start = time.time()

    num_teeth = 20
    #module
    m=4
    # half cone angle
    gamma=0.5*PI/3
    sol =  root(lambda x: np.sin(x)/np.sin(PI/2-x)-0.5,PI/4)
    gamma = sol.x[0]

    axis=OUT
    axis2=RIGHT
    beta=0.0
    param = InvoluteGearParamManager(z_vals=[0,4],
                                     n_teeth=num_teeth,
                                     module=lambda t: m* (1-t*np.sin(gamma)/num_teeth*2),
                                     center=lambda z: m*z*axis*np.cos(gamma),
                                     cone_angle=gamma*2,
                                     angle=lambda z: z*beta+0.2,
                                     h_d=1.2,
                                     h_a=1.0,
                                     h_o=2.5,
                                     root_fillet=0.3,
                                     tip_fillet=0.0,
                                     tip_reduction=0.0,
                                     profile_reduction=0,
                                     profile_shift=0.0,
                                     enable_undercut=False,
                                     inside_teeth=False)
    
    param2 = copy.deepcopy(param)
    param2.cone_angle=PI-gamma*2
    param2.n_teeth = np.round(num_teeth*np.sin(param2.cone_angle/2)/np.sin(param.cone_angle/2))
    param2.module = lambda t: m* (1-t*np.sin(PI/2-gamma)/ param2.n_teeth *2)
    param2.n_cutout_teeth = 0
    param2.angle=lambda z: - z*beta / param2.n_teeth * num_teeth
    param2.h_a=1.0
    param2.h_d=1.2
    param2.profile_shift=0.0
    param2.tip_reduction=0.1
    param2.inside_teeth=False
    param2.enable_undercut=True
    param2.root_fillet=0
    gear_ref = InvoluteGear(param)
    gear_ref_2 = InvoluteGear(param2)
    gear_ref_2.mesh_to(gear_ref,target_dir=rotate_vector(RIGHT, 2*PI/4 * 0.0))

    gear2 = GearBuilder(gear=gear_ref_2,
                        n_points_vert=3,
                        n_points_hz=3,
                        add_plug=False,
                        method='slow')

    gear1 = GearBuilder(gear=gear_ref,
                        n_points_vert=3,
                        n_points_hz=3,
                        add_plug=False,
                        method='slow')
    

    print(f"gear build time: {time.time()-start}")
    
    show(gear1.solid,gear2.solid)
    # solid1 = gear1.solid.translate(nppoint2Vector(-gear1.gear_generator_ref.center_sphere))
    # solid2 = solid1.rotate(Axis.Y,90).mirror(Plane.XY).rotate(Axis.X,gear1.gear_generator_ref.pitch_angle*RAD2DEG*0.5)
    # show(solid1,solid2)

