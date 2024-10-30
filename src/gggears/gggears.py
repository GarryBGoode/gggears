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

from function_generators import *
# from curve import *
from defs import *
from scipy.optimize import root
from scipy.optimize import minimize
import dataclasses
import curve as crv
from typing import Callable

@dataclasses.dataclass
class ZFunctionMixin():
    '''
    This class is used to add capability to gear parameter classes to define changing parameters along Z axis of 3D gear as functions of Z value.
    Simple example would be changing the angle to create helical gears.
    '''
    def __call__(self,z):
        dict_loc = copy.deepcopy(self.__dict__)
        class_loc = copy.deepcopy(self)
        for key,value in dict_loc.items():
            if callable(value):
                # dict_loc[key] = value(z)
                class_loc = dataclasses.replace(class_loc,**{key:value(z)})
        return class_loc

@dataclasses.dataclass
class InvoluteProfileParam():
    pitch_angle: float = 2 * PI / 16
    alpha: float = 20.0 * PI / 180
    h_a: float = 1.0
    h_d: float = 1.2
    profile_shift: float = 0.0
    profile_reduction: float = 0.0
    tip_fillet: float = 0.0
    root_fillet: float = 0.0
    tip_reduction: float = 0.0
    enable_undercut: bool = True

@dataclasses.dataclass
class InvoluteParamMin():
    pitch_angle: float = 2 * PI / 16
    cone_angle: float = 0.0
    alpha: float = 20.0 * PI / 180
    profile_shift: float = 0.0
    profile_reduction: float = 0.0
    h_d: float = 1.2
    enable_undercut: bool = True

    @property
    def rp(self):
        return PI/self.pitch_angle
    
    @property
    def gamma(self):
        return self.cone_angle/2

@dataclasses.dataclass
class GearProfileGenParam():
    pitch_angle: float = 2 * PI / 16
    h_a: float = 1.0
    h_d: float = 1.2
    tip_fillet: float = 0.0
    root_fillet: float = 0.0
    tip_reduction: float = 0.0

    @property
    def rp(self):
        # non-intuitively 1 PI and not 2 due to module convention
        # m*z = D, pitch angle = 2pi/z, D = m*2pi/pitch_angle, R = D/2 = m*pi=/pitch_angle
        return PI/self.pitch_angle


class GearCurveGenerator():
    def __init__(self,
                 n_teeth: float = 16,
                 n_cutout_teeth: int = 0,
                 reference_tooth_curve: crv.Curve = None,
                 module: float = 1,
                 cone_angle: float = 0,
                 axis_offset: float = 0,
                 center: np.ndarray = ORIGIN,
                 angle: float = 0,
                 axis: np.ndarray = OUT,
                 h_a: float = 1,
                 h_d:float = 1.2,
                 h_o: float = 2,
                 tip_fillet: float = 0.0,
                 root_fillet: float = 0.0,
                 tip_reduction: float = 0.0,
                 inside_teeth: bool = False,
                 ) -> None:
        self.n_teeth = n_teeth
        self.n_cutout_teeth = n_cutout_teeth
        self.module = module
        self.cone_angle = cone_angle
        self.axis_offset = axis_offset
        self.center = center
        self.angle = angle
        self.axis = axis
        self.h_a = h_a
        self.h_d = h_d
        self.h_o = h_o
        self.tip_fillet = tip_fillet
        self.root_fillet = root_fillet
        self.tip_reduction = tip_reduction
        self.inside_teeth = inside_teeth
        self.tooth_curve = reference_tooth_curve

        self.generate_ref_base_circles()
        self.update_tip()
        self.update_tip_fillet()
        self.update_root_fillet()
        self.generate_profile()
    
    @property
    def rp_ref(self):
        return self.n_teeth/2

    @property
    def gamma(self):
        return self.cone_angle/2
    
    @property
    def R_ref(self):
        return self.rp_ref/np.sin(self.gamma)
    
    @property
    def center_sphere_ref(self):
        return self.R_ref*np.cos(self.gamma)*self.axis

    @property 
    def rp(self):
        return self.rp_ref*self.module

    @property
    def R(self):
        return self.R_ref*self.module
    
    @property
    def center_sphere(self):
        return self.base_transform(self.center_sphere_ref)
    
    @property
    def pitch_angle(self):
        return 2*PI/self.n_teeth

    def base_transform(self,point):
        rot_axis = np.cross(OUT,self.axis)
        if np.linalg.norm(rot_axis)<1E-12:
            # this can be if axis is OUT or -OUT
            if all(abs(self.axis-OUT)<1E-12):
                rot_angle = 0
                rot_axis = RIGHT
            else:
                rot_angle = PI
                rot_axis = RIGHT
        else:
            rot_angle = angle_between_vectors(OUT,self.axis)
            rot_axis = normalize_vector(rot_axis)

        return scp_Rotation.from_rotvec(rot_angle*rot_axis).apply(point)*self.module + self.center

    def polar_transform(self,point):
        if self.cone_angle==0:
            return xyz_to_cylindrical(point)
        else:
            point = xyz_to_spherical(point,center=self.center_sphere_ref)
            # R theta phi in spherical
            # quasi r = (PI/2-phi) * self.R
            # theta = theta
            # z = self.R-R 
            if point.ndim==1:
                return np.array([(PI/2-point[2])*self.R_ref,
                                 point[1],
                                 (self.R_ref-point[0])])
            else:
                return np.array([(PI/2-point[:,2])*self.R_ref,
                                point[:,1],
                                (self.R_ref-point[:,0])]).transpose()
        
    def inverse_polar_transform(self,point):
        if self.cone_angle==0:
            return cylindrical_to_xyz(point)
        else:
            if point.ndim==1:
                point2 = np.array([self.R_ref-point[2],
                                point[1],
                                PI/2-point[0]/self.R_ref])
                return spherical_to_xyz(point2, 
                                        center=self.center_sphere_ref)
            else:
                point2 = np.array([self.R_ref-point[:,2],
                                   point[:,1],
                                   PI/2-point[:,0]/self.R_ref]).transpose()
                return spherical_to_xyz(point2, 
                                        center=self.center_sphere_ref)

    def r_height_func(self,point):
        if self.cone_angle==0:
            return self.r_height_func_cylindrical(point)
        else:
            return self.r_height_func_spherical(point)
    
    def r_height_func_cylindrical(self,point):
        return np.linalg.norm((point-self.center)[:2])
    def r_height_func_spherical(self,point):
        return self.R*(PI-xyz_to_spherical(point,center=self.center)[2])

    def generate_ref_base_circles(self):

        p0 = RIGHT*self.rp_ref
        pa = self.inverse_polar_transform(self.polar_transform(p0) + np.array([self.h_a,0,0]))
        pd = self.inverse_polar_transform(self.polar_transform(p0) - np.array([self.h_d,0,0]))
        po = self.inverse_polar_transform(self.polar_transform(p0) - np.array([self.h_o,0,0]))

        self.rp_circle = crv.ArcCurve.from_point_center_angle(p0=p0,center=OUT*p0[2],angle=2*PI)
        self.ra_circle = crv.ArcCurve.from_point_center_angle(p0=pa,center=OUT*pa[2],angle=2*PI)
        self.rd_circle = crv.ArcCurve.from_point_center_angle(p0=pd,center=OUT*pd[2],angle=2*PI)
        self.ro_circle = crv.ArcCurve.from_point_center_angle(p0=po,center=OUT*po[2],angle=2*PI)
    
    def update_tip(self):
        sols = []
        rdh = self.r_height_func(self.rd_circle(0))
        rah = self.r_height_func(self.ra_circle(0))
        for guess in np.linspace(0.1,0.9,4):
            sol1 = crv.find_curve_plane_intersect(self.tooth_curve,plane_normal=UP,guess=guess)
            r_sol = self.r_height_func(self.tooth_curve(sol1.x[0]))
            if sol1.success and r_sol>rdh:
                sols.append(sol1)

        if len(sols) > 0:
            sol = sols[np.argmin([sol.x[0] for sol in sols])]
            r_sol = self.r_height_func(self.tooth_curve(sol.x[0]))

            if r_sol-self.tip_reduction<rah:
                if self.tip_reduction>0:
                    sol2 = root(lambda t: self.r_height_func(self.tooth_curve(t[0]))-r_sol+self.tip_reduction,[sol.x[0]])
                    self.tooth_curve.set_end_on(sol2.x[0])
                else:
                    self.tooth_curve.set_end_on(sol.x[0])
                self.ra_circle = crv.ArcCurve.from_point_center_angle(p0=self.tooth_curve(1),
                                                                        center=self.tooth_curve(1)*np.array([0,0,1]),
                                                                        angle=2*PI)
    
    def generate_profile(self):

        # if tip fillet is used, tooth curve tip is already settled
        # in fact this solver tends to fail due to tangential nature of fillet
        if not self.tip_fillet>0:
            sol_tip_2 = crv.find_curve_intersect(self.tooth_curve,self.ra_circle,guess=[1,0])
            solcheck = np.linalg.norm(self.tooth_curve(sol_tip_2.x[0])-self.ra_circle(sol_tip_2.x[1]))
            if sol_tip_2.success or solcheck<1E-12:
                self.tooth_curve.set_end_on(sol_tip_2.x[0])
            else:
                sol_mid = crv.find_curve_plane_intersect(self.tooth_curve,plane_normal=UP,guess=1)
                self.tooth_curve.set_end_on(sol_mid.x[0])

        if not self.root_fillet>0:
            sol_root_1 = crv.find_curve_intersect(self.tooth_curve,self.rd_circle,guess=[0,0], method=crv.IntersectMethod.MINDISTANCE)  
            solcheck = np.linalg.norm(self.tooth_curve(sol_root_1.x[0])-self.rd_circle(sol_root_1.x[1]))
            if sol_root_1.success or solcheck<1E-12:
                self.tooth_curve.set_start_on(sol_root_1.x[0])
            else:
                sol_mid2 = crv.find_curve_plane_intersect(self.tooth_curve,plane_normal=rotate_vector(UP,-self.pitch_angle/2),guess=0)
                self.tooth_curve.set_start_on(sol_mid2.x[0])
        
        self.tooth_mirror = crv.MirroredCurve(self.tooth_curve,plane_normal=UP)
        self.tooth_mirror.reverse()
        tooth_rotate = crv.RotatedCurve(self.tooth_mirror,angle=-self.pitch_angle,axis=OUT)

        pa1 = self.tooth_curve(1)
        pa2 = self.tooth_mirror(0)
        center_a = ((pa1+pa2)/2*np.array([0,0,1]))*OUT
        self.ra_curve = crv.ArcCurve.from_2_point_center(p0=pa1,p1=pa2,center=center_a)

        pd1 = self.tooth_curve(0)
        pd2 = tooth_rotate(1)
        center_d = ((pd1+pd2)/2*np.array([0,0,1]))*OUT
        self.rd_curve = crv.ArcCurve.from_2_point_center(p0=pd2,p1=pd1,center=center_d)

        self.profile = crv.CurveChain(self.rd_curve,self.tooth_curve,self.ra_curve,self.tooth_mirror)
        return self.profile
    
    def update_tip_fillet(self):
        if self.tip_fillet>0:
            sol1 = crv.find_curve_intersect(self.tooth_curve,self.ra_circle,guess=[0.9,0], method=crv.IntersectMethod.MINDISTANCE)
            # if sol is found and the intersection is below the x line
            if sol1.success and self.ra_circle(sol1.x[1])[1]<0:
                sharp_tip = False
                guesses = np.asarray([0.5,1,1.5])*self.tip_fillet
                for guess in guesses:
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.tooth_curve,
                                                          self.ra_circle,
                                                          self.tip_fillet,
                                                          start_locations=[sol1.x[0]-guess/self.tooth_curve.length,
                                                                           sol1.x[1]+guess/self.ra_circle.length],
                                                          method=crv.IntersectMethod.MINDISTANCE)
                    if sol.success:
                        break
                    
                if arc(1)[1]<0:
                    self.tooth_curve.set_end_on(t1)
                    self.tooth_curve.append(arc)
                else:
                    sharp_tip = True
            else:
                sharp_tip = True
            
            if sharp_tip:
                    mirror_curve = crv.MirroredCurve(self.tooth_curve,plane_normal=UP)
                    mirror_curve.reverse()
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.tooth_curve,
                                                          mirror_curve,
                                                          self.tip_fillet,
                                                          start_locations=[0+self.tip_fillet/self.ra_circle.length,
                                                                           1-self.tip_fillet/self.ra_circle.length],
                                                          method=crv.IntersectMethod.MINDISTANCE)
                    if sol.success:
                        # this is almost guaranteed to succeed, the middle of this arc should be on the x axis
                        # the length-proportion based curve parameterization might make it off by a tiny bit so solver is used instead
                        sol2 = crv.find_curve_plane_intersect(arc,plane_normal=UP,guess=0.5)
                        arc.set_end_on(sol2.x[0])
                        self.tooth_curve.set_end_on(t1)
                        self.tooth_curve.append(arc)


    def update_root_fillet(self):

        def angle_check(p):
            return angle_between_vector_and_plane(p,UP) < self.pitch_angle/2
        
        if self.root_fillet>0:
            sol1 = crv.find_curve_intersect(self.tooth_curve,self.rd_circle,guess=[0,0])
            if sol1.success and angle_check(self.rd_circle(sol1.x[1])):
                sharp_root = False
                guesses = np.asarray([0.5,1,1.5])*self.root_fillet
                for guess in guesses:
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.rd_circle,
                                                        self.tooth_curve,
                                                        self.root_fillet,
                                                        start_locations=[sol1.x[1]-guess/self.rd_circle.length,
                                                                         sol1.x[0]+guess/self.tooth_curve.length])
                    if sol.success:
                        break
                if angle_check(arc(0)):
                    self.tooth_curve.set_start_on(t2)
                    self.tooth_curve.insert(0,arc)
                else:
                    sharp_root = True
            else:
                sharp_root = True

            if sharp_root:
                mirror_curve = crv.MirroredCurve(self.tooth_curve,plane_normal=rotate_vector(UP,-self.pitch_angle/2))
                mirror_curve.reverse()
                arc, t1,t2,sol = crv.calc_tangent_arc(mirror_curve,
                                                      self.tooth_curve,
                                                      self.root_fillet,
                                                      start_locations=[1-self.root_fillet/self.tooth_curve.length,
                                                                       0+self.root_fillet/self.tooth_curve.length])
                if sol.success:
                    sol2 = crv.find_curve_plane_intersect(arc,plane_normal=rotate_vector(UP,-self.pitch_angle/2),guess=0.5)
                    arc.set_start_on(sol2.x[0])
                    self.tooth_curve.set_start_on(t2)
                    self.tooth_curve.insert(0,arc)
            
    def generate_profile_closed(self,rd_coeff_right=1.0,rd_coeff_left=0.0):
        v0 = np.cross(self.profile(0),OUT)
        v1 = np.cross(self.profile(1),OUT)
        sol0 = crv.find_curve_plane_intersect(self.ro_circle,v0,guess=0)
        sol1 = crv.find_curve_plane_intersect(self.ro_circle,v1,guess=0)
        
        p0 = self.ro_circle(sol0.x[0])
        p1 = self.ro_circle(sol1.x[0])

        if self.cone_angle==0:
            connector_1 = crv.LineCurve(self.profile(1),p1)
            connector_0 = crv.LineCurve(p0,self.profile(0))
        else:
            connector_1 = crv.ArcCurve.from_2_point_center(p0=self.profile(1),p1=p1,center=self.center_sphere_ref)
            connector_0 = crv.ArcCurve.from_2_point_center(p0=p0,p1=self.profile(0),center=self.center_sphere_ref)
        ro_curve = crv.ArcCurve.from_2_point_center(p0=p1,p1=p0,center=self.ro_circle.center)

        self.profile_closed = crv.CurveChain(self.profile,connector_1,ro_curve,connector_0)

        return self.profile_closed
    
    def generate_gear_pattern(self,profile:crv.Curve):
        def func(t):
            t2,k = self.tooth_moduler(t)
            p = profile(t2)
            return self.base_transform(scp_Rotation.from_euler('z',k*self.pitch_angle).apply(p))

        return crv.Curve(func,0,1)

    def tooth_moduler(self,t):
        t2 = ((np.floor(self.n_teeth)-self.n_cutout_teeth)*t)
        return t2%1, t2//1

class GearSegmentGenerator(GearProfileGenParam):
    '''curve constructor to generate arc segments around the gear tooth flank, creates a unit segment to be repeated for the gear profile'''
    def __init__(self,tooth_curve_generator: Callable, **kwargs):
        super().__init__(**kwargs)
        self.rp_circle, self.rd_circle, self.ra_circle = self.generate_base_circles()
        self.tooth_curve_generator: Callable[[], crv.Curve] = tooth_curve_generator
        self.tooth_curve : crv.Curve = self.tooth_curve_generator()
        self.profile:crv.CurveChain

        self.update_tip()
        self.update_tip_fillet()
        self.update_root_fillet()
        self.generate_profile()
        
        

    def generate_base_circles(self):
        rp_circe = crv.ArcCurve(radius=self.rp,angle=2*PI)
        ra_circle = crv.ArcCurve(radius=self.rp+self.h_a,angle=2*PI)
        rd_circle = crv.ArcCurve(radius=self.rp-self.h_d,angle=2*PI)
        return rp_circe, rd_circle, ra_circle
    
    
    @property
    def ra(self):
        return self.ra_circle.radius
    
    @property
    def rd(self):
        return self.rd_circle.radius
    
    def r_height_func(self,point):
        return np.linalg.norm(point)

    def update_tip(self):
        sols = []
        rdh = self.r_height_func(self.rd_circle(0))
        rah = self.r_height_func(self.ra_circle(0))
        for guess in np.linspace(0.1,0.9,4):
            sol1 = crv.find_curve_plane_intersect(self.tooth_curve,plane_normal=UP,guess=guess)
            r_sol = self.r_height_func(self.tooth_curve(sol1.x[0]))
            if sol1.success and r_sol>rdh:
                sols.append(sol1)

        if len(sols) > 0:
            sol = sols[np.argmin([sol.x[0] for sol in sols])]
            r_sol = self.r_height_func(self.tooth_curve(sol.x[0]))

            if r_sol-self.tip_reduction<rah:
                if self.tip_reduction>0:
                    sol2 = root(lambda t: self.r_height_func(self.tooth_curve(t[0]))-r_sol+self.tip_reduction,[sol.x[0]])
                    self.tooth_curve.set_end_on(sol2.x[0])
                else:
                    self.tooth_curve.set_end_on(sol.x[0])
                self.ra_circle = crv.ArcCurve.from_point_center_angle(p0=self.tooth_curve(1),
                                                                        center=self.tooth_curve(1)*np.array([0,0,1]),
                                                                        angle=2*PI)
    


    def generate_profile(self):

        # if tip fillet is used, tooth curve tip is already settled
        # in fact this solver tends to fail due to tangential nature of fillet
        if not self.tip_fillet>0:
            sol_tip_2 = crv.find_curve_intersect(self.tooth_curve,self.ra_circle,guess=[1,0])
            if sol_tip_2.success:
                self.tooth_curve.set_end_on(sol_tip_2.x[0])
            else:
                sol_mid = crv.find_curve_plane_intersect(self.tooth_curve,plane_normal=UP,guess=1)
                self.tooth_curve.set_end_on(sol_mid.x[0])

        if not self.root_fillet>0:
            sol_root_1 = crv.find_curve_intersect(self.tooth_curve,self.rd_circle,guess=[0,0])
            if sol_root_1.success:
                self.tooth_curve.set_start_on(sol_root_1.x[0])
            else:
                sol_mid2 = crv.find_curve_plane_intersect(self.tooth_curve,plane_normal=rotate_vector(UP,-self.pitch_angle/2),guess=0)
                self.tooth_curve.set_start_on(sol_mid2.x[0])
        
        self.tooth_mirror = crv.MirroredCurve(self.tooth_curve,plane_normal=UP)
        self.tooth_mirror.reverse()
        tooth_rotate = crv.RotatedCurve(self.tooth_mirror,angle=-self.pitch_angle,axis=OUT)

        pa1 = self.tooth_curve(1)
        pa2 = self.tooth_mirror(0)
        center_a = ((pa1+pa2)/2*np.array([0,0,1]))*OUT
        self.ra_curve = crv.ArcCurve.from_2_point_center(p0=pa1,p1=pa2,center=center_a)

        pd1 = self.tooth_curve(0)
        pd2 = tooth_rotate(1)
        center_d = ((pd1+pd2)/2*np.array([0,0,1]))*OUT
        self.rd_curve = crv.ArcCurve.from_2_point_center(p0=pd2,p1=pd1,center=center_d)

        self.profile = crv.CurveChain(self.rd_curve,self.tooth_curve,self.ra_curve,self.tooth_mirror)
        return self.profile
    
    def update_tip_fillet(self):
        if self.tip_fillet>0:
            sol1 = crv.find_curve_intersect(self.tooth_curve,self.ra_circle,guess=[0.9,0], method=crv.IntersectMethod.MINDISTANCE)
            # if sol is found and the intersection is below the x line
            if sol1.success and self.ra_circle(sol1.x[1])[1]<0:
                sharp_tip = False
                guesses = np.asarray([0.5,1,1.5])*self.tip_fillet
                for guess in guesses:
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.tooth_curve,
                                                          self.ra_circle,
                                                          self.tip_fillet,
                                                          start_locations=[sol1.x[0]-guess/self.tooth_curve.length,
                                                                           sol1.x[1]+guess/self.ra_circle.length],
                                                          method=crv.IntersectMethod.MINDISTANCE)
                    if sol.success:
                        break
                    
                if arc(1)[1]<0:
                    self.tooth_curve.set_end_on(t1)
                    self.tooth_curve.append(arc)
                else:
                    sharp_tip = True
            else:
                sharp_tip = True
            
            if sharp_tip:
                    mirror_curve = crv.MirroredCurve(self.tooth_curve,plane_normal=UP)
                    mirror_curve.reverse()
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.tooth_curve,
                                                          mirror_curve,
                                                          self.tip_fillet,
                                                          start_locations=[0+self.tip_fillet/self.ra_circle.length,
                                                                           1-self.tip_fillet/self.ra_circle.length],
                                                          method=crv.IntersectMethod.MINDISTANCE)
                    if sol.success:
                        # this is almost guaranteed to succeed, the middle of this arc should be on the x axis
                        # the length-proportion based curve parameterization might make it off by a tiny bit so solver is used instead
                        sol2 = crv.find_curve_plane_intersect(arc,plane_normal=UP,guess=0.5)
                        arc.set_end_on(sol2.x[0])
                        self.tooth_curve.set_end_on(t1)
                        self.tooth_curve.append(arc)


    def update_root_fillet(self):

        def angle_check(p):
            return angle_between_vector_and_plane(p,UP) < self.pitch_angle/2
        
        if self.root_fillet>0:
            sol1 = crv.find_curve_intersect(self.tooth_curve,self.rd_circle,guess=[0,0])
            if sol1.success and angle_check(self.rd_circle(sol1.x[1])):
                sharp_root = False
                guesses = np.asarray([0.5,1,1.5])*self.root_fillet
                for guess in guesses:
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.rd_circle,
                                                        self.tooth_curve,
                                                        self.root_fillet,
                                                        start_locations=[sol1.x[1]-guess/self.rd_circle.length,
                                                                         sol1.x[0]+guess/self.tooth_curve.length])
                    if sol.success:
                        break
                if angle_check(arc(0)):
                    self.tooth_curve.set_start_on(t2)
                    self.tooth_curve.insert(0,arc)
                else:
                    sharp_root = True
            else:
                sharp_root = True

            if sharp_root:
                mirror_curve = crv.MirroredCurve(self.tooth_curve,plane_normal=rotate_vector(UP,-self.pitch_angle/2))
                mirror_curve.reverse()
                arc, t1,t2,sol = crv.calc_tangent_arc(mirror_curve,
                                                      self.tooth_curve,
                                                      self.root_fillet,
                                                      start_locations=[1-self.root_fillet/self.tooth_curve.length,
                                                                       0+self.root_fillet/self.tooth_curve.length])
                if sol.success:
                    sol2 = crv.find_curve_plane_intersect(arc,plane_normal=rotate_vector(UP,-self.pitch_angle/2),guess=0.5)
                    arc.set_start_on(sol2.x[0])
                    self.tooth_curve.set_start_on(t2)
                    self.tooth_curve.insert(0,arc)
            
            
class GearSegmentGeneratorSpherical(GearSegmentGenerator):
    def __init__(self,tooth_curve_generator: Callable, center_z: float = 0.0, **kwargs):
        self.center = center_z*OUT
        super().__init__(tooth_curve_generator=tooth_curve_generator,**kwargs)

    @property
    def R(self):
        return np.sqrt(self.rp**2+self.center[2]**2)
    
    def r_height_func(self,point):
        return self.R*(PI-xyz_to_spherical(point,center=self.center)[2])
    
    def generate_base_circles(self):
        rp_circe = crv.ArcCurve(radius=self.rp,angle=2*PI)
        pa = scp_Rotation.from_euler('y',-self.h_a/self.R).apply(self.rp*RIGHT-self.center)+self.center
        pd = scp_Rotation.from_euler('y',self.h_d/self.R).apply(self.rp*RIGHT-self.center)+self.center
        ra_circle = crv.ArcCurve.from_point_center_angle(p0=pa,center=pa[2]*OUT,angle=2*PI)
        rd_circle = crv.ArcCurve.from_point_center_angle(p0=pd,center=pd[2]*OUT,angle=2*PI)
        return rp_circe, rd_circle, ra_circle

class InvoluteFlankGenerator(InvoluteParamMin):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.involute_curve = crv.InvoluteCurve(active=False)
        self.involute_connector = crv.LineCurve(active=False)
        self.undercut_curve = crv.InvoluteCurve(active=False)

        self.involute_curve_sph = crv.SphericalInvoluteCurve(active=False)
        self.involute_connector_arc = crv.ArcCurve(active=False)
        self.undercut_curve_sph = crv.SphericalInvoluteCurve(active=False)

        

        if self.cone_angle==0:
            self.rd = self.rp - self.h_d + self.profile_shift
            self.calculate_involutes_cylindric()
            self.tooth_curve = crv.CurveChain(self.undercut_curve,self.involute_connector,self.involute_curve)
        else:
            # gamma is cone angle / 2 property
            self.R = self.rp/np.sin(self.gamma)
            self.C_sph = 1/self.R
            self.center = OUT*np.sqrt(self.R**2-self.rp**2)
            self.an_d = (self.h_d-self.profile_shift ) /self.R
            # 180deg cone is a flat circle... leads to similar result like infinite radius cylinder... which would be a straight rack
            if self.cone_angle==PI:
                self.tooth_curve = self.calculate_rack_spherical()
            else:
                self.calculate_involutes_spherical()
                self.tooth_curve = crv.CurveChain(self.undercut_curve_sph,self.involute_connector_arc,self.involute_curve_sph)

    def __call__(self):
        return self.tooth_curve

    def calculate_involutes_cylindric(self):
        self.involute_curve.active=True
        
        pitch_circle = crv.ArcCurve(radius=self.rp,angle=2*PI)

        # base circle of involute function
        self.rb = self.rp * np.cos(self.alpha)
        # setup base circle
        self.involute_curve.r = self.rb

        # find the angular position of the involute curve start point
        # this is the point where the involute curve intersects the pitch circle
        sol2 = crv.find_curve_intersect(self.involute_curve,pitch_circle,guess=[0.5,0])
        involute_angle_0 = angle_between_vectors(RIGHT,self.involute_curve(sol2.x[0]))

        # based on tec-science article
        # https://www.tec-science.com/mechanical-power-transmission/involute-gear/profile-shift/
        # angle change from profile shift
        da = self.profile_shift * np.tan(self.alpha)/ self.rp - self.profile_reduction/self.rp
        # angle to move the involute into standard construction position
        # by convention moving clockwise, which is negative angular direction
        # the tooth shall be symmetrical on the x-axis, so the base angle is quarter of pitch angle
        # added angular components to compensate for profile shift and the involute curve's travel from base to pitch circle
        self.involute_curve.angle = -(self.pitch_angle / 4 + involute_angle_0 + da)

        # hence the tooth shall be on the x axis, the involute shall not cross the x axis
        # find the point where the involute curve reaches the x axis, that shall be the end of the segment
        x_line = crv.Curve(arc_from_2_point,params={'p0':ORIGIN,'p1':self.rp*2*RIGHT,'curvature':0})
        sol1 = crv.find_curve_intersect(self.involute_curve,x_line)
        self.involute_curve.t_1 = self.involute_curve.p2t(sol1.x[0])
        self.involute_curve.update_lengths()

        if self.rd<self.rb:
            p_invo_base = self.involute_curve(0)
            p_invo_d = p_invo_base*self.rd/self.rb

            self.involute_connector.p0 = p_invo_d
            self.involute_connector.p1 = p_invo_base
            self.involute_connector.active=True
            if self.enable_undercut:
                # when undercut is used, there is no line between undercut and involute in 2D
                
                self.undercut_curve.active=True
                # the undercut is an involute curve with an offset vector (sometimes called trochoid)
                # radial and tangential elements of the offset vector
                rad_ucut = self.rd - self.rp
                tan_ucut = +self.rd * np.tan(self.alpha)
                ucut_ofs_v = np.array((rad_ucut,tan_ucut,0)).reshape(VSHAPE)
                
                self.undercut_curve.r = self.rp
                self.undercut_curve.angle = self.involute_curve.angle - self.alpha
                self.undercut_curve.v_offs = ucut_ofs_v
                self.undercut_curve.t_0=0.3
                self.undercut_curve.t_1=-0.35
                self.undercut_curve.update_lengths()

                loc_curve = crv.CurveChain(self.involute_connector,self.involute_curve)
                t_invo = loc_curve.get_length_portions()[1]

                # find intersection of undercut curve and involute curve, might need multiple guesses from different starting points
                guess = 0.1
                for k in range(10):
                    sol1 = root(lambda p: (loc_curve(p[0])-self.undercut_curve(p[1])),[guess+t_invo,guess,0])
                    if abs(sol1.x[1])>0.01+t_invo and sol1.success:
                        break
                    guess = (k+1) * 0.1

                # loc_curve.set_start_on(sol1.x[0])
                # find lowest point of ucut
                sol2 = minimize(lambda t: np.linalg.norm(self.undercut_curve(t)),0)
                self.undercut_curve.set_start_and_end_on(sol2.x[0],sol1.x[1])
                self.involute_connector.active=False
            else:
                self.undercut_curve.active=False
        else:
            self.involute_connector.active=False
            self.undercut_curve.active=False

    def calculate_involutes_spherical(self):
        def involute_angle_func(x):
            t=x[0]
            r=x[1]
            p0=involute_sphere(t,      r,angle=0,C=self.C_sph)
            p1=involute_sphere(t+DELTA,r,angle=0,C=self.C_sph)
            p2=involute_sphere(t-DELTA,r,angle=0,C=self.C_sph)
            tan = normalize_vector(p1-p2)
            center = np.array([0,0,np.sqrt(self.R**2-r**2)])
            sph_tan = normalize_vector(np.cross(p0-center,np.array([p0[0],p0[1],0])))
            # angle = np.arctan2(np.linalg.norm(np.cross(tan,sph_tan)),np.dot(tan,sph_tan))
            angle = angle_between_vectors(tan,sph_tan)

            return [p0[0]**2+p0[1]**2-self.rp**2, angle-PI/2-self.alpha]
    
        self.involute_curve_sph.active=True
        base_res = root(involute_angle_func,[self.alpha/2,self.rp*np.cos(self.alpha)],tol=1E-14)
        self.rb = base_res.x[1]
        
        self.involute_curve_sph.r = self.rb
        self.involute_curve_sph.c_sphere = self.C_sph
        
        angle_0 = angle_between_vectors(involute_sphere(base_res.x[0],self.rb,angle=0,C=self.C_sph)*np.array([1,1,0]),
                                        RIGHT)
        angle_offset = -angle_0 - (self.pitch_angle/4 + self.profile_shift*np.tan(self.alpha)/2 /self.rp)
        self.involute_curve_sph.angle = angle_offset
        self.involute_curve_sph.z_offs = -involute_sphere(base_res.x[0],base_res.x[1],C=self.C_sph)[2]
        self.involute_curve_sph.t_0 = 0
        self.involute_curve_sph.t_1 = 1
        sol1 = crv.find_curve_plane_intersect(self.involute_curve_sph,offset=ORIGIN,plane_normal=UP,guess=1)
        self.involute_curve_sph.set_end_on(sol1.x[0])


        ## undercut
        p0=self.involute_curve_sph(0)
        axis = normalize_vector(np.cross(p0,OUT))
        # by convention the pitch circle is in the x-y plane
        # the involute goes partially below the pitch circle
        # calculate the angle to go until the dedendum circle
        p0_xy = (p0-self.center)*np.array([1,1,0])
        an_diff = self.an_d-angle_between_vectors(p0-self.center,p0_xy)+(PI/2-self.gamma)
        if an_diff<0:
            self.involute_connector_arc.active=False
            self.undercut_curve_sph.active=False
        else:
            p1 = scp_Rotation.from_rotvec(-axis*an_diff).apply(p0-self.center)+self.center
            self.involute_connector_arc = crv.ArcCurve.from_2_point_center(p0=p1,p1=p0,center=self.center)
            self.involute_connector_arc.active=True

            if not self.enable_undercut:
                self.undercut_curve_sph.active=False
            else:
                self.undercut_curve_sph.active=True
                ref_rack = self.calculate_rack_spherical()
                self.undercut_curve_sph.r = self.rp
                self.undercut_curve_sph.angle = 0
                self.undercut_curve_sph.z_offs = 0
                self.undercut_curve_sph.v_offs = scp_Rotation.from_euler('y',PI/2 * np.sign(self.C_sph)).apply(ref_rack(0)-self.R*RIGHT)
                self.undercut_curve_sph.c_sphere = self.C_sph
                self.undercut_curve_sph.t_0 = 1
                self.undercut_curve_sph.t_1 = -1

                self.undercut_curve_sph.update_lengths()


                loc_curve = crv.CurveChain(self.involute_connector_arc,self.involute_curve_sph)
                rb_curve = crv.ArcCurve.from_2_point_curvature(p0=self.involute_curve_sph(0),
                                                               p1=self.involute_curve_sph(0)*np.array([1,-1,1]),
                                                               curvature=1/self.involute_curve_sph.r,
                                                               revolutions=0)

                sol0 = crv.find_curve_intersect(self.undercut_curve_sph,rb_curve,guess=[0.1,0])
                sol1 = crv.find_curve_intersect(loc_curve,self.undercut_curve_sph, guess=[0.3,sol0.x[0]*2])
                loc_curve.set_start_on(sol1.x[0],preserve_inactive_curves=True)
                self.undercut_curve_sph.set_end_on(sol1.x[1])

                # sol2 = crv.find_curve_intersect(self.undercut_curve_sph,self.rd_curve)
                sol2 = minimize(lambda t: np.linalg.norm(self.undercut_curve_sph(t)[:2]),0)
                self.undercut_curve_sph.set_start_on(sol2.x[0])
                
    
    def calculate_rack_spherical(self):
        def rack_flanc_func(t,a):
            axis1 = scp_Rotation.from_euler('x',-self.alpha).apply(OUT)
            v0 = self.R*RIGHT
            an1 = t*np.cos(self.alpha)
            v1 = scp_Rotation.from_rotvec(-axis1*an1).apply(v0)
            v2 = scp_Rotation.from_euler('z',t+a).apply(v1)
            return v2
        
        an_tooth_sph = (self.pitch_angle/2 + self.profile_shift*np.tan(self.alpha) /self.rp )* self.rp / self.R
        curve1 = crv.Curve(rack_flanc_func,
                           t0=-1,
                           t1=1,
                           params={'a':-an_tooth_sph/2})
               
        
        sol2 = root(lambda t: np.arcsin(curve1(t[0])[2]/self.R)+self.an_d,[0])

        sol1 = crv.find_curve_plane_intersect(curve1,plane_normal=UP,guess=1)
        curve1.set_start_and_end_on(sol2.x[0],sol1.x[0])
        return curve1
    
class GearProfile2D(InvoluteProfileParam):
    def __init__(self,**kwargs):
        '''
        alpha: pressure angle in degrees, affects tooth curvature. Suggested values between 10-30
        h_a: addendum / module coefficient (tooth height above pitch circle)
        h_d: dedendum / module coefficient (tooth height below pitch circle)
        inside_teeth: generate ring gear where the teeth point inward (for example for planetary gear setup)
        profile_shift: profile shift coefficient x.
            Simulates the cutting rack profile being shifted away from the gear by x*m.
            Changes shape and diameter slightly, reduces undercut.
        profile_reduction: reduction coefficient b. Simulates the cutting rack profile being slightly asymmetric,
            pushing together the cutting flanks by b*m/2 left and right side each.
            Used for generating backlash
        fillet_radius: coefficient of module, this fillet is applied to the tooth tip and the edge between
            undercut and involute, or edge between root circle and involute when there is no undercut.
        
        '''
        # init all parameters of ancestor class
        super().__init__(**kwargs)

        self.h_d_min = PI / 4 / np.tan(self.alpha)
        if self.h_d > self.h_d_min:
            self.h_d = self.h_d_min
        self.tooth_h = (self.h_a + self.h_d)

        # pitch circle radius
        self.rp = PI/self.pitch_angle

        # base circle of involute function
        self.rb = self.rp * np.cos(self.alpha)
        # addendum and dedendum radii
        self.ra = self.rp +  (self.h_a + self.profile_shift)
        self.rd = self.rp - (self.h_d - self.profile_shift)

        # reference circles
        self.tip_circle =   crv.Curve(lambda t: self.ra*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.root_circle =  crv.Curve(lambda t: self.rd*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.pitch_circle = crv.Curve(lambda t: self.rp*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.base_circle =  crv.Curve(lambda t: self.rb*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))

        # empty curves representing tooth flank segments
        self.involute_curve = crv.InvoluteCurve()
        self.undercut_curve = crv.InvoluteCurve()
        self.undercut_connector_line = crv.LineCurve(active=False)

        # curve representing one side of the tooth, without addendum and dedendum
        self.tooth_curve = crv.CurveChain(self.undercut_curve, self.undercut_connector_line,self.involute_curve)

        # addendum and dedendum arcs
        self.rd_curve = crv.ArcCurve.from_2_point_center(p0=rotate_vector(self.rd*RIGHT,-self.pitch_angle/2).reshape(VSHAPE),
                                                         p1=self.rd*RIGHT,
                                                         center=ORIGIN)
        self.ra_curve = crv.ArcCurve.from_2_point_center(p1=self.ra*RIGHT,
                                                         p0=rotate_vector(self.ra*RIGHT,-self.pitch_angle/2).reshape(VSHAPE),
                                                         center=ORIGIN)
        # perform math calculations to generate the tooth profile
        self.calculate_involute_base()
        self.calculate_uncercut()
        self.calculate_arcs()

        self.tooth_mirror = crv.CurveChain(*[crv.MirroredCurve(curve,plane_normal=UP) for curve in self.tooth_curve])
        self.tooth_mirror.reverse()
        self.profile = crv.CurveChain(self.rd_curve,
                                      *self.tooth_curve,
                                      self.ra_curve,
                                      *self.tooth_mirror)
        
    def calculate_involute_base(self):
        # setup base circle
        self.involute_curve.r = self.rb

        # find the angular position of the involute curve start point
        # this is the point where the involute curve intersects the pitch circle
        sol2 = crv.find_curve_intersect(self.involute_curve,self.pitch_circle)
        involute_angle_0 = angle_between_vectors(RIGHT,self.involute_curve(sol2.x[0]))

        # based on tec-science article
        # https://www.tec-science.com/mechanical-power-transmission/involute-gear/profile-shift/
        # angle change from profile shift
        da = self.profile_shift * np.tan(self.alpha)/ self.rp - self.profile_reduction/self.rp
        # angle to move the involute into standard construction position
        # by convention moving clockwise, which is negative angular direction
        # the tooth shall be symmetrical on the x-axis, so the base angle is quarter of pitch angle
        # added angular components to compensate for profile shift and the involute curve's travel from base to pitch circle
        self.involute_curve.angle = -(self.pitch_angle / 4 + involute_angle_0 + da)

        # hence the tooth shall be on the x axis, the involute shall not cross the x axis
        # find the point where the involute curve reaches the x axis, that shall be the end of the segment
        x_line = crv.Curve(arc_from_2_point,params={'p0':ORIGIN,'p1':self.ra*2*RIGHT,'curvature':0})
        sol1 = crv.find_curve_intersect(self.involute_curve,x_line)
        self.involute_curve.t_1 = self.involute_curve.p2t(sol1.x[0])
        self.involute_curve.update_lengths()

        p_invo_base = self.involute_curve(0)
        p_invo_d = p_invo_base*self.rd/self.rb

        #set up extension line from involute to dedendum circle along radial direction
        if self.rd<self.rb:
            self.undercut_connector_line.p0 = p_invo_d
            self.undercut_connector_line.p1 = p_invo_base
            self.undercut_connector_line.active=True
            self.undercut_connector_line.update_lengths()
        else:
            self.undercut_connector_line.p0 = p_invo_base
            self.undercut_connector_line.p1 = p_invo_base
            self.undercut_connector_line.active=False

    def calculate_uncercut(self):
        # undercut can only happen if dedendum is smaller than base circle - and it is allowed
        if self.rd < self.rb and self.enable_undercut:
            
            # when undercut is used, there is no line between undercut and involute in 2D
            self.undercut_connector_line.active=False
            self.undercut_curve.active=True
            # the undercut is an involute curve with an offset vector (sometimes called trochoid)
            # radial and tangential elements of the offset vector
            rad_ucut = self.rd - self.rp
            tan_ucut = +self.rd * np.tan(self.alpha)
            ucut_ofs_v = np.array((rad_ucut,tan_ucut,0)).reshape(VSHAPE)
            
            self.undercut_curve.r = self.rp
            self.undercut_curve.angle = self.involute_curve.angle - self.alpha
            self.undercut_curve.v_offs = ucut_ofs_v
            self.undercut_curve.t_0=0
            self.undercut_curve.t_1=-0.35
            self.undercut_curve.update_lengths()

            # find intersection of undercut curve and involute curve, might need multiple guesses from different starting points
            guess = 0.1
            for k in range(10):
                sol1 = root(lambda p: (self.involute_curve(p[0])-self.undercut_curve(p[1])),[guess,guess,0])
                if abs(sol1.x[1])>0.01 and sol1.success:
                    break
                guess = (k+1) * 0.1

            self.involute_curve.set_start_on(sol1.x[0])
            # self.undercut_curve.t_1 = self.undercut_curve.p2t(sol1.x[1])
            self.undercut_curve.set_end_on(sol1.x[1])
            self.undercut_curve.t_0=0.6

            # find lowest point of ucut
            sol2 = minimize(lambda t: np.linalg.norm(self.undercut_curve(t)),0)
            # sol3 = root(lambda t: (self.undercut_curve(t[0])-self.root_circle(t[1])),[0,0,0])

            # self.undercut_curve.t_0 = self.undercut_curve.p2t(sol2.x[0])
            self.undercut_curve.set_start_on(sol2.x[0])

            # self.tooth_curve.insert(0,self.undercut_curve)
            self.tooth_curve.update_lengths()

        else:
            # find the intersection of the involute and undercut connector curve and the root circle
            # clip tooth curve to the intersection point
            sol1 = crv.find_curve_intersect(self.tooth_curve,self.rd_curve)
            self.tooth_curve=self.tooth_curve.cut(sol1.x[0],preserve_inactive_curves=True)[1]
            
            if self.root_fillet>0:
                rd_cut_curve = self.rd_curve.cut(sol1.x[1])[0]
                self.tooth_curve.insert(0,rd_cut_curve)
                location = self.tooth_curve.get_t_for_index(0)[1]
                self.tooth_curve = self.tooth_curve.fillet(self.root_fillet,location)
                self.tooth_curve.pop(0)
                

            else:
                # add placeholder of undercut curve
                self.undercut_curve.active=False


    def calculate_arcs(self):
        sol1 = crv.find_curve_intersect(self.tooth_curve,self.ra_curve)
        guesses = [0.1,0.5,0.9]
        sol2 = []
        for guess in guesses:
            sol2.append(crv.find_curve_line_intersect(self.tooth_curve,offset=ORIGIN,line_direction=RIGHT, guess=guess))
        ttooth1 = sol1.x[0]
        # ttooth2 = sol2.x[0]
        # lowest positive value where the tooth curve intersectsthe x-line
        ttooth2 = np.min([sol.x[0] for sol in sol2 if sol.success and sol.x[0]>0])
        if ttooth1<ttooth2:
            self.ra_curve = self.ra_curve.cut(sol1.x[1])[1]
            self.tooth_curve = self.tooth_curve.cut(sol1.x[0])[0]
            # reconfigure ra_curve
            p0 = self.ra_curve(0)
            p1 = self.ra_curve(0) * np.array([1,-1,1])
            self.ra_curve = crv.ArcCurve.from_2_point_center(p0=p0,p1=p1,center=ORIGIN)

        else:
            self.tooth_curve = self.tooth_curve.cut(ttooth2)[0]
            self.ra_curve.active=False
            self.ra = np.linalg.norm(self.tooth_curve(1))


        p1 = self.tooth_curve(0)
        angle1 = abs(angle_between_vectors(RIGHT,p1))
        angle0 = self.pitch_angle-angle1
        p0 = rotate_vector(RIGHT*self.rd,-angle0)
        self.rd_curve=crv.ArcCurve.from_2_point_center(p0,p1,ORIGIN)

        # self.tooth_curve.insert(0,self.rd_curve)

@dataclasses.dataclass
class SphericalInvoluteProfileParam(InvoluteProfileParam):
    gamma: float = PI/4

class GearProfileSpherical(SphericalInvoluteProfileParam):
    def __init__(self,
                 **kwargs # just to catch unexpected kwargs error
                 ):
        
        super().__init__(**kwargs)

        
        # D = m*z
        # z=2pi/pitch_angle
        # r = D/2 = m*2pi/pitch_angle/2

        self.rp = PI/self.pitch_angle
        self.R = self.rp/np.sin(self.gamma)
        self.C_sph = 1/self.R
        self.center = OUT*np.sqrt(self.R**2-self.rp**2)

        
        # base circle of involute function: needs to be solved later
        self.rb = 0
        self.X = self.profile_shift
        self.B = self.profile_reduction

        # spherical angle parameter related to addendum and dedendum circles
        self.an_a = (self.h_a+self.X ) /self.R
        self.an_d = (self.h_d-self.X ) /self.R

        # self.tip_circle =   Curve(lambda t: self.ra*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        # self.root_circle =  Curve(lambda t: self.rd*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.pitch_circle = crv.Curve(lambda t: self.rp*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        # self.base_circle =  Curve(lambda t: self.rb*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))

        p0 = RIGHT*self.rp
        p1 = scp_Rotation.from_euler('y',self.an_d).apply(p0-self.center)+self.center
        p2 = scp_Rotation.from_euler('z',-self.pitch_angle).apply(p1)
        p3 = scp_Rotation.from_euler('y',-self.an_a).apply(p0-self.center)+self.center
        p4 = scp_Rotation.from_euler('z',-self.pitch_angle).apply(p3)
        self.rd_curve = crv.ArcCurve.from_2_point_center(p0=p1,
                                                         p1=p2,
                                                         center=p1*np.array([0,0,1]))
        self.ra_curve = crv.ArcCurve.from_2_point_center(p0=p3,
                                                         p1=p4,
                                                         center=self.center)
        self.rd = self.rd_curve.radius
        self.ra = self.ra_curve.radius

        self.involute_curve = crv.SphericalInvoluteCurve()
        self.undercut_curve = crv.SphericalInvoluteCurve()
        self.undercut_connector_arc = crv.ArcCurve(active=False)




        self.calculate_ref_rack_disc()
        self.calculate_involute_base()
        self.calculate_undercut()

        self.tooth_curve = crv.CurveChain(self.undercut_curve,
                                          self.undercut_connector_arc,
                                          self.involute_curve)

        self.calculate_arcs()

        self.tooth_mirror = crv.CurveChain(*[crv.MirroredCurve(curve,plane_normal=UP) for curve in self.tooth_curve])
        self.tooth_mirror.reverse()
        self.profile = crv.CurveChain(self.rd_curve,
                                  *self.tooth_curve,
                                  self.ra_curve,
                                  *self.tooth_mirror)

        self.ra = np.linalg.norm(self.tooth_curve(1)*np.array([1,1,0]))
        self.rd = np.linalg.norm(self.tooth_curve(0)*np.array([1,1,0]))

    def calculate_ref_rack_disc_bad(self):
        # pitch_angle = 2pi / z
        # D = m*z; rp = m*z/2
        # dra = h_a*m = R*dan_a 
        # dan_a = h_a*m/R
        # dan_d = h_d*m/R

        

        h_a = self.R*np.sin(self.an_a)
        h_d = self.R*np.sin(self.an_d)

        p0 = RIGHT*self.R
        p1 = p0+scp_Rotation.from_euler('x',-self.alpha).apply(IN) # IN = -1 in z direction
        p1 = normalize_vector(p1)*self.R

        curve_1 = crv.ArcCurve.from_2_point_center(p0=p0,p1=p1,center=ORIGIN)
        sol1 = root(lambda t: curve_1(t[0])[2]+h_d,[1])
        sol2 = root(lambda t: curve_1(t[0])[2]-h_a,[1])
        
        curve_1.p0, curve_1.p1  = curve_1(sol1.x[0]), curve_1(sol2.x[0])
        curve_1.update_lengths()
        p0 = curve_1(0)
        p1 = curve_1(1)
        
        an_tooth_sph = (self.pitch_angle/2 + self.X*np.tan(self.alpha) /self.rp )* self.rp / self.R

        # an_shift = self.X/self.R

        # mirror y values
        p2,p3 = p1*np.array([1,-1,1]), p0*np.array([1,-1,1])
        # shift (rotate) for other side of tooth
        p2 = (scp_Rotation.from_euler('z',an_tooth_sph/2)).apply(p2)
        p3 = (scp_Rotation.from_euler('z',an_tooth_sph/2)).apply(p3)
        p0 = (scp_Rotation.from_euler('z',-an_tooth_sph/2)).apply(p0)
        p1 = (scp_Rotation.from_euler('z',-an_tooth_sph/2)).apply(p1)

        curve_1 = crv.ArcCurve.from_2_point_center(p0=p0,p1=p1,center=ORIGIN)
        curve_2 = crv.ArcCurve.from_2_point_center(p0=p1,p1=p2,center=ORIGIN)
        curve_3 = crv.ArcCurve.from_2_point_center(p0=p2,p1=p3,center=ORIGIN)

        self.ref_rack_curve=crv.CurveChain(curve_1,curve_2,curve_3)

    def calculate_ref_rack_disc(self):
        def rack_flanc_func(t,a):
            axis1 = scp_Rotation.from_euler('x',-self.alpha).apply(OUT)
            v0 = self.R*RIGHT
            an1 = t*np.cos(self.alpha)
            v1 = scp_Rotation.from_rotvec(-axis1*an1).apply(v0)
            v2 = scp_Rotation.from_euler('z',t+a).apply(v1)
            return v2
        
        an_tooth_sph = (self.pitch_angle/2 + self.X*np.tan(self.alpha) /self.rp )* self.rp / self.R
        curve1 = crv.Curve(rack_flanc_func,
                           t0=-1,
                           t1=1,
                           params={'a':-an_tooth_sph/2})
               
        
        sol2 = root(lambda t: np.arcsin(curve1(t[0])[2]/self.R)+self.an_d,[0])
        curve1.set_start_on(sol2.x[0])
        sol1 = root(lambda t: np.arcsin(curve1(t[0])[2]/self.R)-self.an_a,[0])
        curve1.set_end_on(sol1.x[0])
        curve3 = crv.MirroredCurve(curve1,plane_normal=UP).reverse()
        curve2 = crv.ArcCurve.from_2_point_curvature(p0=curve1(1),p1=curve3(0),curvature=self.C_sph,axis=OUT)
        self.ref_rack_curve = crv.CurveChain(curve1,curve2,curve3)

    def calculate_involute_base(self):
        def involute_angle_func(x):
            t=x[0]
            r=x[1]
            p0=involute_sphere(t,      r,angle=0,C=self.C_sph)
            p1=involute_sphere(t+DELTA,r,angle=0,C=self.C_sph)
            p2=involute_sphere(t-DELTA,r,angle=0,C=self.C_sph)
            tan = normalize_vector(p1-p2)
            center = np.array([0,0,np.sqrt(self.R**2-r**2)])
            sph_tan = normalize_vector(np.cross(p0-center,np.array([p0[0],p0[1],0])))
            # angle = np.arctan2(np.linalg.norm(np.cross(tan,sph_tan)),np.dot(tan,sph_tan))
            angle = angle_between_vectors(tan,sph_tan)

            return [np.sqrt(p0[0]**2+p0[1]**2)-self.rp, angle-PI/2-self.alpha]
    
        base_res = root(involute_angle_func,[self.alpha/2,self.rp*np.cos(self.alpha)])
        self.rb = base_res.x[1]

        self.involute_curve.r = self.rb
        self.involute_curve.c_sphere = self.C_sph
        
        angle_0 = angle_between_vectors(involute_sphere(base_res.x[0],self.rb,angle=0,C=self.C_sph)*np.array([1,1,0]),
                                        RIGHT)
        angle_offset = -angle_0 - (self.pitch_angle/4 + self.X*np.tan(self.alpha)/2 /self.rp)
        # self.involute_param['z_offs'] = -involute_sphere(base_res.x[0],base_res.x[1],C=self.C_sph)
        # self.involute_param['angle'] = angle_offset
        self.involute_curve.angle = angle_offset
        self.involute_curve.z_offs = -involute_sphere(base_res.x[0],base_res.x[1],C=self.C_sph)[2]
        self.involute_curve.t_0 = 0
        self.involute_curve.t_1 = 1
        self.involute_curve.update_lengths()
        sol1 = crv.find_curve_plane_intersect(self.involute_curve,offset=ORIGIN,plane_normal=UP,guess=1)
        self.involute_curve.set_end_on(sol1.x[0])

        if self.rb<self.rd:
            self.undercut_connector_arc.active=False
            self.undercut_curve.active=False
            sol2 = crv.find_curve_intersect(self.involute_curve,self.rd_curve)
            self.involute_curve.set_start_on(sol2.x[0])


    def calculate_undercut(self):
        p0=self.involute_curve(0)
        axis = normalize_vector(np.cross(p0,OUT))
        # by convention the pitch circle is in the x-y plane
        # the involute goes partially below the pitch circle
        # calculate the angle to go until the dedendum circle
        p0_xy = (p0-self.center)*np.array([1,1,0])
        an_diff = self.an_d-angle_between_vectors(p0-self.center,p0_xy)+(PI/2-self.gamma)
        if an_diff<0:
            self.undercut_connector_arc.active=False
            self.undercut_curve.active=False
        else:
            p1 = scp_Rotation.from_rotvec(-axis*an_diff).apply(p0-self.center)+self.center
            self.undercut_connector_arc = crv.ArcCurve.from_2_point_center(p0=p1,p1=p0,center=self.center)
            if self.enable_undercut:
                self.undercut_curve.r = self.rp
                self.undercut_curve.angle = 0
                self.undercut_curve.z_offs = 0
                self.undercut_curve.v_offs = scp_Rotation.from_euler('y',PI/2 * np.sign(self.C_sph)).apply(self.ref_rack_curve(0)-self.R*RIGHT)
                self.undercut_curve.c_sphere = self.C_sph
                self.undercut_curve.t_0 = 1
                self.undercut_curve.t_1 = -1

                self.undercut_curve.update_lengths()


                loc_curve = crv.CurveChain(self.undercut_connector_arc,self.involute_curve)
                rb_curve = crv.ArcCurve.from_2_point_curvature(p0=self.involute_curve(0),
                                                               p1=self.involute_curve(0)*np.array([1,-1,1]),
                                                               curvature=1/self.involute_curve.r,
                                                               revolutions=0)

                sol0 = crv.find_curve_intersect(self.undercut_curve,rb_curve,guess=[0.1,0])
                sol1 = crv.find_curve_intersect(loc_curve,self.undercut_curve, guess=[0.3,sol0.x[0]*2])
                loc_curve.set_start_on(sol1.x[0],preserve_inactive_curves=True)
                self.undercut_curve.set_end_on(sol1.x[1])

                sol2 = crv.find_curve_intersect(self.undercut_curve,self.rd_curve)
                self.undercut_curve.set_start_on(sol2.x[0])
            else:
                self.undercut_curve.active=False
                self.undercut_connector_arc.active=True
                
    def calculate_arcs(self):

        sol1 = crv.find_curve_intersect(self.tooth_curve,self.ra_curve)
        guesses = [0.1,0.5,0.9]
        sol2 = []
        for guess in guesses:
            sol2.append(crv.find_curve_plane_intersect(self.tooth_curve,offset=ORIGIN, plane_normal=UP, guess=guess))
        ttooth1 = sol1.x[0]
        # ttooth2 = sol2.x[0]
        # lowest positive value where the tooth curve intersectsthe x-line
        ttooth2 = np.min([sol.x[0] for sol in sol2 if sol.success and sol.x[0]>0])
        if ttooth1<ttooth2:
            self.ra_curve = self.ra_curve.cut(sol1.x[1])[1]
            # self.tooth_curve = self.tooth_curve.cut(sol1.x[0])[0]
            self.tooth_curve.set_end_on(sol1.x[0])
            # reconfigure ra_curve
            p0 = self.ra_curve(0)
            p1 = self.ra_curve(0) * np.array([1,-1,1])
            self.ra_curve = crv.ArcCurve.from_2_point_center(p0=p0,p1=p1,center=ORIGIN)

        else:
            self.tooth_curve = self.tooth_curve.cut(ttooth2)[0]
            self.ra_curve.active=False
            self.ra = np.linalg.norm(self.tooth_curve(1))


        p1 = self.tooth_curve(0)
        p0 = rotate_vector(p1*np.array([1,-1,1]),-self.pitch_angle)
        self.rd_curve=crv.ArcCurve.from_2_point_center(p0,p1,ORIGIN)






@dataclasses.dataclass
class GearParam2D():
    num_of_teeth: float = 16
    cutout_teeth_num: int = 0
    angle: float = 0
    center: np.ndarray = ORIGIN
    module: float = 1
    h_o: float = 3
    inside_teeth: bool =False
    profile_overlap: float = 0.0
    profile_symmetry_shift: float = 0.0
    profile_param: InvoluteProfileParam = InvoluteProfileParam() # using all default values of the class

    def __post_init__(self):
        '''Derived parameters and value checks'''
        # limit values to +-1
        self.profile_symmetry_shift = np.max([-1.0,np.min([1.0,self.profile_symmetry_shift])])
        self.profile_overlap = np.max([-1.0,np.min([1.0,self.profile_overlap])])
        if self.num_of_teeth<0:
            self.num_of_teeth = np.abs(self.num_of_teeth)
            self.inside_teeth = not(self.inside_teeth)
        self.cutout_teeth_num = int(abs(self.cutout_teeth_num))
        self.profile_param.pitch_angle = self.pitch_angle

    
    # derived parameters
    @property
    def rp(self):
        return  self.m*self.num_of_teeth/2
    
    @property
    def pitch_angle(self):
        return 2*PI/self.num_of_teeth
    
    # shorthand for module
    @property
    def m(self):
        return self.module



@dataclasses.dataclass
class GearParam2DSpherical(GearParam2D):
    profile_param: SphericalInvoluteProfileParam = SphericalInvoluteProfileParam()

class Gear2D(GearParam2D):
    '''
    Class used for representing and handling 2D gears.
    Takes a reference profile of 1 tooth, and generates the full gear profile.
    Applies rotation, module scaling and center-offset to the profile.
    '''
    def __init__(self,
                 **kwargs 
                 ):
        super().__init__(**kwargs)

        self.generate_profile_reference()
        self.calc_outer_radius()


        self.teeth_curve = crv.Curve(self.teeth_generator,t1=(self.num_of_teeth-self.cutout_teeth_num)/self.num_of_teeth)
        # self.teeth_curve.t_1 = (self.num_of_teeth-self.cutout_teeth_n)/self.num_of_teeth
        self.r_d_padding = crv.Curve(lambda t: arc_from_2_point_center(t, p0=self.teeth_curve(1),p1=self.teeth_curve(0), center=self.center))
        if self.r_d_padding.length<DELTA/10:
            self.r_d_padding.active=False
        self.boundary = crv.CurveChain(self.teeth_curve,self.r_d_padding)
        self.profile = self.generate_profile()
        self.profile_closed = self.generate_profile_closed()

    def calc_outer_radius(self):
        self.ra = self.profile_reference.ra * self.m
        self.rd = self.profile_reference.rd * self.m
        if self.inside_teeth:
            self.ro = self.rp+self.h_o/self.m
        else:
            self.ro = self.rp-self.h_o*self.m
    
    def generate_profile_reference(self):
        self.profile_reference = GearProfile2D(**self.profile_param.__dict__)
        return self.profile_reference
    
    def generate_profile(self):
        # profile of 1 tooth to be repeated, but with applied rotation, module scaling and center-offset
        curves = []
        for curve in self.profile_reference.profile.get_curves():
            if isinstance(curve,crv.ArcCurve):
                curves.append(crv.ArcCurve.from_2_point_center(p0=self.base_curve_transform(curve.p0),
                                                               p1=self.base_curve_transform(curve.p1),
                                                               center=self.base_curve_transform(curve.center),
                                                               active=curve.active))
            elif isinstance(curve,crv.LineCurve):
                curves.append(crv.LineCurve(p0=self.base_curve_transform(curve.p0),
                                            p1=self.base_curve_transform(curve.p1),
                                            active=curve.active))
            else:
                curves.append(crv.TransformedCurve(self.base_curve_transform,curve=curve))
        self.profile = crv.CurveChain(*curves)
        return self.profile

    def teeth_generator(self,t):
        z = np.floor(self.num_of_teeth)
        t2 = (t*z)%1
        angle_mod = np.floor(t*z)
        rotmat = scp_Rotation.from_euler('z',angle_mod*self.pitch_angle+self.angle)
        return rotmat.apply(self.profile_reference.profile(t2)*self.m)+self.center
    
    def base_curve_transform(self,point):
        return scp_Rotation.from_euler('z',self.angle).apply(point)*self.m+self.center
    
    def generate_profile_closed(self):
        self.profile_closed = crv.CurveChain(*self.profile.copy()[0:])
        rot_pitch = scp_Rotation.from_euler('z',self.pitch_angle)
        
        rd_copy = crv.ArcCurve.from_2_point_center(p0=rot_pitch.apply(self.profile_closed[0](0)),
                                                    p1=rot_pitch.apply(self.profile_closed[0](1)),
                                                    center=self.center)
        
        self.profile_closed[0].set_start_on(0.5*(1-self.profile_symmetry_shift)-self.profile_overlap/2)
        rd_copy.set_end_on(0.5*(1-self.profile_symmetry_shift)+self.profile_overlap/2)
        self.profile_closed.append(rd_copy)
        


        p1 = self.profile_closed(1)
        p0 = self.profile_closed(0)
        center = self.center 
        p1_o = p1+normalize_vector(p1-center)*(self.ro-np.linalg.norm(p1-center))
        p0_o = p0+normalize_vector(p0-center)*(self.ro-np.linalg.norm(p0-center))
        
        self.profile_closed.extend([crv.LineCurve(p0=p1,p1=p1_o),
                                    crv.ArcCurve.from_2_point_center(p0=p1_o,p1=p0_o,center=center),
                                    crv.LineCurve(p0=p0_o,p1=p0)])
        
        return self.profile_closed

class Gear2DSpherical(Gear2D):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        

    def generate_profile_reference(self):
        self.profile_reference = GearProfileSpherical(**self.profile_param.__dict__)
        return self.profile_reference
    
    def calc_outer_radius(self):
        self.ra = self.profile_reference.ra * self.m
        self.rd = self.profile_reference.rd * self.m

        

        if self.inside_teeth:
            self.an_o = self.profile_reference.an_a * 0 + self.h_o/self.profile_reference.R
        else:
            self.an_o = -self.profile_reference.an_d * 0 - self.h_o/self.profile_reference.R

    def generate_profile_closed(self):
        self.profile_closed = crv.CurveChain(*self.profile.copy()[0:])
        rot_pitch = scp_Rotation.from_euler('z',self.pitch_angle)
        rd_copy = crv.ArcCurve.from_2_point_center(p0=rot_pitch.apply(self.profile_closed[0](0)),
                                            p1=rot_pitch.apply(self.profile_closed[0](1)),
                                            center=self.profile_closed[0].center)
        
        self.profile_closed[0].set_start_on(0.5*(1-self.profile_symmetry_shift)-self.profile_overlap/2)
        rd_copy.set_end_on(0.5*(1-self.profile_symmetry_shift)+self.profile_overlap/2)
        self.profile_closed.append(rd_copy)

        p1 = self.profile_closed(1)
        p0 = self.profile_closed(0)
        center = self.center + OUT*((p0+p1)/2)[2]
        center_sph =self.center + self.profile_reference.center*self.m

        self.gamma = self.profile_reference.gamma
        ax0 = normalize_vector(np.cross(p0-center_sph,OUT))
        ax1 = normalize_vector(np.cross(p1-center_sph,OUT))
        p0_o = center_sph + (scp_Rotation.from_rotvec(ax0*(self.an_o+self.profile_reference.an_d))).apply(p0-center_sph)
        p1_o = center_sph + (scp_Rotation.from_rotvec(ax1*(self.an_o+self.profile_reference.an_d))).apply(p1-center_sph)
        

        self.profile_closed.extend([crv.ArcCurve.from_2_point_center(p0=p1,p1=p1_o,center=center_sph),
                                    crv.ArcCurve.from_2_point_center(p0=p1_o,p1=p0_o,center=OUT*p1_o[2]+self.center*np.array([1,1,0])),
                                    crv.ArcCurve.from_2_point_center(p0=p0_o,p1=p0,center=center_sph)])

        self.ro = np.linalg.norm((p1_o-center)[:2])
        return self.profile_closed

class InvoluteProfileHandler(InvoluteProfileParam,ZFunctionMixin):
    pass
class SphericalInvoluteProfileHandler(SphericalInvoluteProfileParam,ZFunctionMixin):
    pass

class GearParamHandler(GearParam2D,ZFunctionMixin):
    center: lambda z: OUT*z
    profile_param: InvoluteProfileHandler = InvoluteProfileHandler()
    
class GearParamHandlerSpherical(GearParam2DSpherical,ZFunctionMixin):
    center: lambda z: OUT*z
    module: lambda z: self.default_module(z)
    profile_param: SphericalInvoluteProfileHandler = SphericalInvoluteProfileHandler()

    def default_module(self,z):
        return 1-np.tan(self.gamma)*z


class GearCylindric():
    def __init__(self,

                 z_vals = np.array([0,1]),
                 params : GearParamHandler = GearParamHandler(),
                 **kwargs):
        
        self.params = params
        self.pitch_angle = 2*PI/self.params.num_of_teeth
        self.params.profile_param.pitch_angle = self.pitch_angle

        self.z_vals = z_vals

    def generate_gear_slice(self,z) -> Gear2D:
        paramdict = self.params(z).__dict__
        return Gear2D(**paramdict)
    
    def gen_point(self,z,t):
        return self.generate_gear_slice(z).profile(t)
    
class GearSpherical(GearCylindric):
    def generate_gear_slice(self,z) -> Gear2DSpherical:
        paramdict = self.params(z).__dict__
        return Gear2DSpherical(**paramdict)
    

# class Gear():
#     def __init__(self,
#                  z_vals=np.array([0]),
#                  params: GearParamHandler = GearParamHandler(),
#                  flank_generator: InvoluteFlankGenerator = InvoluteFlankGenerator(),
#                  segment_generator: GearSegmentGenerator = GearSegmentGenerator()
#     ):
#         self.params = params
#         self.flank_generator = flank_generator
#         self.segment_generator = segment_generator
#         self.z_vals = z_vals