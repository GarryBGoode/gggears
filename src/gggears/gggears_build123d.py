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
from typing import Type
from build123d import *
import numpy as np
import numpy.typing as npt
import time
from ocp_vscode import show, show_object, reset_show, set_port, set_defaults, get_defaults
set_port(3939)


def nppoint2Vector(p: np.ndarray):
    if p.size==3:
        return Vector((p[0],p[1],p[2]))
    else:
        return [Vector((p[k,0],p[k,1],p[k,2])) for k in range(p.shape[0])]


class GearBuilder():
    def __init__(self,
                 gear: GearCylindric,
                 n_points_hz = 4,
                 n_points_vert=4,
                 **kwargs):
        self.gear = gear
        self.n_points_hz = n_points_hz
        self.n_points_vert = n_points_vert
        self.vert_oversamp_ratio = 2

        self.gear_stack = []
        self.gear_stacks = []
        self.nurb_profile_stacks = []
        solid_levels = []
        profile_surfaces = []

        for ii in range(len(self.gear.z_vals)-1):
            # need more gear slices than nurb points to produce 'best' fit without overfitting
            # oversamp ratio controls how many more
            # the 2 end points will be locked down, the middle points are approximated by fitting
            z_tweens = np.linspace(self.gear.z_vals[ii],
                                   self.gear.z_vals[ii+1],
                                   int(np.ceil((n_points_vert-2)*self.vert_oversamp_ratio)) + 2)
            gear_stack_loc = [self.gear.generate_gear_slice(z) for z in z_tweens]
            self.gear_stacks.append(gear_stack_loc)
            self.nurb_profile_stacks.append([convert_curve_nurbezier(gearprofile.profile_closed.get_curves(lerp_inactive=True),
                                                               n_points=self.n_points_hz, 
                                                               force_2D=True) 
                                       for gearprofile in gear_stack_loc])

        for nurb_stack in self.nurb_profile_stacks:
            self.nurb_profile_stack = nurb_stack
            profile_surfaces.extend(self.generate_side_surfaces())

        profile_surfaces.insert(0,self.generate_cover_surface(self.nurb_profile_stacks[0][0]))
        profile_surfaces.append(self.generate_cover_surface(self.nurb_profile_stacks[-1][-1]))


        
        self.profile_surfaces = profile_surfaces
        shell1 = Shell(self.profile_surfaces)
        solid1 = Solid(shell1)

        if not solid1.is_valid():
            # solid1 = Solid(shell1.fix())
            Warning("Tooth profile solid is not valid")
            

        self.profile_solid = solid1
        # flatten the gear stacks
        self.gear_stack = [ gear  for gear_stack in self.gear_stacks for gear in gear_stack]
        
        ra_min = np.min([gear.ra for gear in self.gear_stack])
        ra_max = np.max([gear.ra for gear in self.gear_stack])
        rd_min = np.min([gear.rd for gear in self.gear_stack])
        rd_max = np.max([gear.rd for gear in self.gear_stack])
        ro_min = np.min([gear.ro for gear in self.gear_stack])
        ro_max = np.max([gear.ro for gear in self.gear_stack])

        zmin = np.min([gear.center[2] for gear in self.gear_stack])
        zmax = np.max([gear.center[2] for gear in self.gear_stack])

        if self.gear.params.inside_teeth:
            plug = Cylinder(ro_max,zmax-zmin) - \
                    Cylinder(ra_max+DELTA,zmax-zmin)
            plug = plug.translate(nppoint2Vector(OUT*(zmax+zmin)/2))
        else:
            plug = Cylinder(rd_min+DELTA,zmax-zmin)
            plug = plug.translate(nppoint2Vector(OUT*(zmax+zmin)/2))



        start = time.time()
        n_teeth = int(np.floor(self.gear.params.num_of_teeth-self.gear.params.cutout_teeth_num))

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
                angle = self.gear.params.profile_param.pitch_angle*RAD2DEG*(2**(k-1))
                shape_dict.append(shape_dict[k-1].fuse(shape_dict[k-1].rotate(Axis((0,0,0),(0,0,1)),angle),glue=False,tol=tol))

            if bin_n_teeth[-(k+1)]=='1':

                
                angle_construct = angle_idx * self.gear.params.profile_param.pitch_angle*RAD2DEG
                
                solid2_to_fuse.append(shape_dict[k].rotate(Axis((0,0,0),(0,0,1)),angle_construct))
                angle_idx = angle_idx+2**k
                
        self.solid = Solid.fuse(plug,*solid2_to_fuse, glue=False,tol=tol).clean()

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
        vectors = nppoint2Vector(curve_bezier.params['points'])
        weights = curve_bezier.params['weights'].tolist()
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
            point_stack = np.stack([curve.params['points'] for curve in curve_stack_vert])
            weight_stack = np.stack([curve.params['weights'] for curve in curve_stack_vert])

            # example size numbers: points for one curve: 4x3, weights for one curve: 4, point stack: 6x4x3, weight stack: 6x4
            # adding weights as 4th coordinate
            # axis 0: vertical, axis 1: horizontal, axis 2: x-y-z-w
            target_points = np.concatenate([point_stack,weight_stack[:,:,np.newaxis]],axis=2)
            point_collector2.append(target_points[:,:-1,:])


        target_point_2 = np.concatenate(point_collector2,axis=1)
        sol2,points2,weights2 = self.solve_surface(target_point_2, n_points_vert=self.n_points_vert)
        points2 = np.concatenate([points2,weights2[:,:,np.newaxis]],axis=2)

        idx = np.arange(0,points2.shape[1],self.n_points_hz-1)
        points3=np.roll(np.insert(points2,idx,points2[:,idx,:],axis=1),-1,axis=1)



        for k in range(len(self.nurb_profile_stack[0])):

            points_w = points3[:,k*self.n_points_hz:(k+1)*self.n_points_hz,:]
            points = points_w[:,:,:3]
            weights = points_w[:,:,3]
            vpoints = [nppoint2Vector(points[k]) for k in range(points.shape[0])]

            surface_1 = Face.make_bezier_surface(vpoints,weights.tolist())
            if surface_1.area>DELTA:
                profile_surfaces.append(surface_1)
        
        return profile_surfaces
        
    
    def generate_cover_surface(self, nurb_chain: CurveChain):
        # resisting 1-lining this for improved debugging
        splines = [self.gen_splines(curve) for curve in nurb_chain if curve.length>DELTA]
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

start = time.time()

n_z = 17

# gear1 = GearProfile2D(pitch_angle=np.pi*2/8, h_d=1.25,h_a=1,profile_shift=0.25)
gear2 = GearBuilder(GearCylindric(z_vals=np.array([0,1,2]),n_tweens=6,
                                  params=GearParamHandler(num_of_teeth=n_z,
                                                          angle=lambda z: 0.05*(abs(z-1))**1*PI*2,
                                                          center=lambda z: z*3*OUT,
                                                          inside_teeth=False,
                                                          profile_overlap=0,
                                                          cutout_teeth_num=0,
                                                          profile_param= InvoluteProfileHandler(
                                                              profile_shift=lambda z: 0.2-0.0*(z-1)**2,
                                                              root_fillet=0.2,
                                                              h_a= lambda z: 1,
                                                              enable_undercut=True))),
                                                              n_points_vert=4,
                                                              n_points_hz=5)


show(gear2.solid,gear2.profile_solid)