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

from gggears import *
from function_generators import *
from curve import *
from build123d import *
import numpy as np
import time
from ocp_vscode import show, set_port
set_port(3939)


def nppoint2Vector(p: np.ndarray):
    if p.size==3:
        return Vector((p[0],p[1],p[2]))
    else:
        return [Vector((p[k,0],p[k,1],p[k,2])) for k in range(p.shape[0])]


class GearBuilder():
    def __init__(self,
                 params: InvoluteGearParamManager,
                 add_plug = True,
                 n_points_hz = 4,
                 n_points_vert=4,
                 **kwargs):
        self.params = params
        self.gear = InvoluteGear(params=self.params)
        self.n_points_hz = n_points_hz
        self.n_points_vert = n_points_vert
        self.vert_oversamp_ratio = 2.5
        self.force_2D = False

        self.gear_stacks = []
        self.nurb_profile_stacks = []
        profile_surfaces = []



        start_gear = time.time()
        for ii in range(len(self.params.z_vals)-1):
            # need more gear slices than nurb points to produce 'best' fit without overfitting
            # oversamp ratio controls how many more
            # the 2 end points will be locked down, the middle points are approximated by fitting
            z_tweens = np.linspace(self.params.z_vals[ii],
                                   self.params.z_vals[ii+1],
                                   int(np.ceil((n_points_vert-2)*self.vert_oversamp_ratio)) + 2)
            gear_stack_loc = [self.gear.setup_generator(self.params(z)) for z in z_tweens]
            self.gear_stacks.append(gear_stack_loc)

        self.gear_generator_ref = self.gear_stacks[0][0]
        print(f"gear generation time: {time.time()-start_gear}")

        start_nurb = time.time()
        for gear_stack_loc in self.gear_stacks:

            nurb_stack=[]
            for gearprofile in gear_stack_loc:
                profile1 = gearprofile.generate_profile_closed(rd_coeff_right=0.5,rd_coeff_left=0.5)
                ref_profile = CurveChain(*profile1.get_curves())
                NurbsConv = convert_curve_nurbezier(ref_profile)
                NurbsConv.enforce_continuity()
                for nurb in NurbsConv:
                    nurb.points = gearprofile.base_transform(nurb.points)
                nurb_stack.append(NurbsConv)
            self.nurb_profile_stacks.append(nurb_stack)

        print(f"nurb generation time: {time.time()-start_nurb}")
        for nurb_stack in self.nurb_profile_stacks:
            self.nurb_profile_stack = nurb_stack
            profile_surfaces.extend(self.generate_side_surfaces())

        profile_surfaces.insert(0,self.generate_cover_surface(self.nurb_profile_stacks[0][0]))
        profile_surfaces.append(self.generate_cover_surface(self.nurb_profile_stacks[-1][-1]))

        # visual debug is harder if these surfaces go into self right from the start
        self.profile_surfaces = profile_surfaces

        shell1 = Shell(self.profile_surfaces)
        solid1 = Solid(shell1)

        if not solid1.is_valid():
            # solid1 = Solid(shell1.fix())
            Warning("Tooth profile solid is not valid")


        self.profile_solid = solid1

        self.params_stack = [self.params(z) for z in self.params.z_vals]
        # flatten the gear stacks
        self.gear_stack = [ gear  for gear_stack in self.gear_stacks for gear in gear_stack]

        ra_min = np.min([gear.ra for gear in self.gear_stack])
        ra_max = np.max([gear.ra for gear in self.gear_stack])
        rd_min = np.min([gear.rd for gear in self.gear_stack])
        rd_max = np.max([gear.rd for gear in self.gear_stack])
        ro_min = np.min([gear.ro for gear in self.gear_stack])
        ro_max = np.max([gear.ro for gear in self.gear_stack])

        ro_0 = self.gear_stack[0].ro
        ro_1 = self.gear_stack[-1].ro
        ra_0 = self.gear_stack[0].ra
        ra_1 = self.gear_stack[-1].ra

        zmin = np.min([gear.base_transform(gear.ro_curve(0.5))[2] for gear in self.gear_stack])
        zmax = np.max([gear.base_transform(gear.ro_curve(0.5))[2] for gear in self.gear_stack])

        axis1 = Axis((0,0,0),(self.params.axis[0],self.params.axis[1],self.params.axis[2]))

        if any([param.cone_angle!=0 for param in self.params_stack]):
            if self.params.inside_teeth:
                z_height = zmax-zmin
                align1 = (Align.CENTER, Align.CENTER, Align.MIN)
                plug = Cylinder(ro_0,z_height,align=align1) - \
                      Cylinder(ro_1,z_height,align=align1) - \
                      Cone(ro_0,ro_1,zmax-zmin,align=align1)
                plug = plug.translate(nppoint2Vector(OUT*(zmin)))
            else:
                plug = Cone(ro_0,ro_1,zmax-zmin,align=(Align.CENTER, Align.CENTER, Align.MIN))
                plug = plug.translate(nppoint2Vector(OUT*(zmin)))

        else:
            if self.params.inside_teeth:
                plug = Cylinder(ro_max,zmax-zmin) - \
                        Cylinder(ra_max+DELTA,zmax-zmin)
                plug = plug.translate(nppoint2Vector(OUT*(zmax+zmin)/2))
            else:
                plug = Cylinder(ro_min+DELTA,zmax-zmin)
                plug = plug.translate(nppoint2Vector(OUT*(zmax+zmin)/2))




        start = time.time()
        n_teeth = int(np.floor(self.gear.params.n_teeth-self.gear.params.n_cutout_teeth))

        bin_n_teeth = bin(n_teeth)[2:]
        shape_dict = []
        solid2_to_fuse = []
        angle_construct = 0.0
        angle_idx = 0
        tol = 1E-4
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

        if add_plug:
            self.solid = self.solid.fuse(plug).clean()

        print(f"fuse time: {time.time()-start}")

    def fix_spline_connection(self):
        self.point_diffs=[]
        for profile in self.nurb_profile_stack:
            n=len(profile)
            for k in range(len(profile)):
                ii = k%n
                jj = (k-1)%n
                p_mean = (profile[ii].params['points'][0]+profile[jj].params['points'][-1])/2.0
                self.point_diffs.append(np.linalg.norm(profile[ii].params['points'][0]-profile[jj].params['points'][-1]))
                profile[ii].params['points'][0] = profile[jj].params['points'][-1] = p_mean

    def gen_splines(self,curve_bezier:Curve):
        vectors = nppoint2Vector(curve_bezier.points)
        weights = curve_bezier.weights.tolist()
        return Edge.make_bezier(*vectors,weights=weights)

    def generate_side_surfaces(self):
        top_splines = []
        bot_splines = []
        profile_surfaces=[]
        point_collector2 = []

        # k goes along the profile length
        for k in range(len(self.nurb_profile_stack[0])):
            # vertical stack is built for 1 2d profile element (eg. all involute curves stacked up)
            curve_stack_vert = [ curve[k] for curve in self.nurb_profile_stack]
            # point_stack = np.stack([curve.params['points'] for curve in curve_stack_vert])
            # weight_stack = np.stack([curve.params['weights'] for curve in curve_stack_vert])
            point_stack = np.stack([curve.points for curve in curve_stack_vert])
            weight_stack = np.stack([curve.weights for curve in curve_stack_vert])
            # example size numbers: points for one curve: 4x3, weights for one curve: 4, point stack: 6x4x3, weight stack: 6x4
            # adding weights as 4th coordinate
            # axis 0: vertical, axis 1: horizontal, axis 2: x-y-z-w
            target_points = np.concatenate([point_stack,weight_stack[:,:,np.newaxis]],axis=2)
            point_collector2.append(target_points[:,:-1,:])


        target_point_2 = np.concatenate(point_collector2,axis=1)
        start_surf = time.time()
        sol2,points2,weights2 = self.solve_surface(target_point_2, n_points_vert=self.n_points_vert)
        print(f"surface solve time: {time.time()-start_surf}")
        points2 = np.concatenate([points2,weights2[:,:,np.newaxis]],axis=2)

        # idx = np.arange(0,points2.shape[1],self.n_points_hz-1)
        idx = self.nurb_profile_stack[0].knots[:-1]
        points3=np.roll(np.insert(points2,idx,points2[:,idx,:],axis=1),-1,axis=1)


        idx_loc=0
        start_surf_gen = time.time()
        for k in range(len(self.nurb_profile_stack[0])):
            n_points = self.nurb_profile_stack[0][k].n_points

            points_w = points3[:,idx_loc:idx_loc+n_points,:]
            idx_loc = idx_loc + n_points
            points = points_w[:,:,:3]
            weights = points_w[:,:,3]
            vpoints = [nppoint2Vector(points[k]) for k in range(points.shape[0])]

            surface_1 = Face.make_bezier_surface(vpoints,weights.tolist())
            if surface_1.area>DELTA:
                profile_surfaces.append(surface_1)
        print(f"surface gen time: {time.time()-start_surf_gen}")
        return profile_surfaces


    def generate_cover_surface(self, nurb_chain: CurveChain):
        # resisting 1-lining this for improved debugging
        splines = [self.gen_splines(curve) for curve in nurb_chain if curve.length>DELTA and curve.active]
        return Face.make_surface(Wire(splines)).clean()



    def solve_surface(self,target_points, n_points_vert = 4):

        n = target_points.shape[1]
        m = n_points_vert
        o = target_points.shape[0]

        def point_allocator(x):
            points = np.zeros((m,n,4))
            points[0,:,:] = target_points[0,:,:]
            points[-1,:,:] = target_points[-1,:,:]

            points[1:-1,:,:] = x[:(m-2)*n*4].reshape(((m-2),n,4))
            t = x[(m-2)*n*4:(m-2)*n*4+o-2]
            return points,t

        def inverse_allocator(points,t):
            x = np.zeros(((m-2)*n*4+o))
            x[:(m-2)*n*4] = points[1:-1,:,:].reshape((m-2)*n*4)
            x[(m-2)*n*4:(m-2)*n*4+o-2] = t
            return x

        def cost_fun(x):
            points,t = point_allocator(x)
            tref = np.linspace(0,1,o)[1:-1]
            return np.sum((bezierdc(t,points)-target_points[1:-1,:,:])**2) + np.sum((t-tref)**2)*10

        init_t = np.linspace(0,1,o)[1:-1]
        init_points = bezierdc(np.linspace(0,1,m),target_points)
        init_guess_x = inverse_allocator(init_points,init_t)

        sol = minimize(cost_fun,init_guess_x)
        points_sol, t = point_allocator(sol.x)
        points_out = points_sol[:,:,:3]
        weights_out = points_sol[:,:,3]
        return sol, points_out, weights_out

    def solve_surface2(self,target_points, n_points_vert = 4):

        n = target_points.shape[1]
        m = n_points_vert
        o = target_points.shape[0]

        def point_allocator(x):
            points = np.zeros((m,n,4))
            points[0,:,:] = target_points[0,:,:]
            points[-1,:,:] = target_points[-1,:,:]

            points[1:-1,:,:] = x[:(m-2)*n*4].reshape(((m-2),n,4))
            return points

        def inverse_allocator(points,t):
            x = np.zeros(((m-2)*n*4+o))
            x[:(m-2)*n*4] = points[1:-1,:,:].reshape((m-2)*n*4)
            return x

        def cost_fun(x):
            points = point_allocator(x)
            tref = np.linspace(0,1,o)[1:-1]
            return np.sum((bezierdc(tref,points)-target_points[1:-1,:,:])**2)

        init_t = np.linspace(0,1,o)[1:-1]
        init_points = bezierdc(np.linspace(0,1,m),target_points)
        init_guess_x = inverse_allocator(init_points,init_t)

        sol = minimize(cost_fun,init_guess_x)
        points_sol = point_allocator(sol.x)
        points_out = points_sol[:,:,:3]
        weights_out = points_sol[:,:,3]
        return sol, points_out, weights_out



start = time.time()

n_z = 9

gamma=PI/2 * 0.5

axis=OUT
axis2=RIGHT
param = InvoluteGearParamManager(z_vals=[0,2],
                                 n_teeth=n_z,
                                 module=lambda z: 1-np.tan(gamma)*z/n_z*2,
                                 center=lambda z: z*axis,
                                 cone_angle=gamma*2,
                                 angle=lambda z: 0.2*z,
                                 axis=axis,
                                 h_d=1.2,
                                 h_o=2.5,
                                 root_fillet=0.0,
                                 tip_fillet=0.0,
                                 tip_reduction=0.1,
                                 profile_reduction=0,
                                 profile_shift=0.0,
                                 enable_undercut=True,
                                 inside_teeth=False)
gear1 = GearBuilder(params=param,
                    n_points_vert=3,
                    n_points_hz=4,
                    add_plug=False)

param2= InvoluteGearParamManager(z_vals=[0,2],
                                 n_teeth=n_z,
                                 module=lambda z: 1-np.tan(gamma)*z/n_z*2,
                                 center=lambda z: z*axis2+gear1.gear_generator_ref.rp*LEFT+gear1.gear_generator_ref.center_sphere,
                                 cone_angle=gamma*2,
                                 angle=lambda z: -0.2*z-PI/n_z*0,
                                 axis=axis2,
                                 h_d=1.2,
                                 h_o=2.5,
                                 root_fillet=0.0,
                                 tip_fillet=0.0,
                                 tip_reduction=0.1,
                                 profile_reduction=0,
                                 profile_shift=0.0,
                                 enable_undercut=True,
                                 inside_teeth=False)

gear2 = GearBuilder(params=param2,
                    n_points_vert=3,
                    n_points_hz=4,
                    add_plug=False)

print(f"total time: {time.time()-start}")


solid3 = gear2.solid.rotate(Axis((0,0,0),(0,0,1)),120)
solid4 = gear2.solid.rotate(Axis((0,0,0),(0,0,1)),-120)
solid5 = gear1.solid.rotate(Axis((0,0,gear1.gear_generator_ref.center_sphere[2]),
                                        (0,1,0)),
                                        angle=180)
# mirror(Plane(origin=Vector(0,0,gear1.gear_generator_ref.center_sphere[2]),normal=Vector(0,0,1)))
# solid5 = solid5.rotate(Axis.Z,360/n_z/2)
show(gear1.solid,gear2.solid,solid3,solid4,solid5)
