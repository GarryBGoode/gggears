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

from gggears.function_generators import *
from gggears.defs import *
from scipy.optimize import root
from scipy.optimize import minimize
import dataclasses
import gggears.curve as crv
import copy

@dataclasses.dataclass
class ZFunctionMixin():
    '''
    This class is used to add capability to gear parameter classes to define
    changing parameters along Z axis of 3D gear as functions of Z value.
    Simple example would be changing the angle to create helical gears.
    '''
    z_vals: np.ndarray = np.array([0,1])

    def __call__(self,z):
        dict_loc = copy.deepcopy(self.__dict__)
        class_loc = copy.deepcopy(self)
        for key,value in dict_loc.items():
            if callable(value):
                class_loc = dataclasses.replace(class_loc,**{key:value(z)})
        return class_loc

    def __add__(self,other):
        if isinstance(other,ZFunctionMixin):
            z_vals = np.unique(np.concatenate((self.z_vals,other.z_vals)))
            dict_loc = copy.deepcopy(self.__dict__)
            dict_other = copy.deepcopy(other.__dict__)
            class_loc = copy.deepcopy(self)

            def lambda_adder(key):
                if callable(dict_loc[key]) and callable(dict_other[key]):
                    return lambda z: dict_loc[key](z) + dict_other[key](z)
                elif callable(dict_loc[key]):
                    return lambda z: dict_loc[key](z) + dict_other[key]
                elif callable(dict_other[key]):
                    return lambda z: dict_loc[key] + dict_other[key](z)
                else:
                    return dict_loc[key] + dict_other[key]

            for key,value in dict_loc.items():
                if callable(value):
                    # dict_loc[key] = value(z)
                    if callable(value):
                        class_loc = dataclasses.replace(class_loc,
                                                        **{key: lambda_adder(key)})
                else:
                    if not isinstance(value,bool):
                        class_loc = dataclasses.replace(class_loc,
                                                        **{key:value+dict_other[key]})
            class_loc = dataclasses.replace(class_loc,**{'z_vals':z_vals})
            return class_loc
        else:
            raise(TypeError('ZFunctionMixin can only be added to \
                             another ZFunctionMixin'))


@dataclasses.dataclass
class InvoluteGearParam():
    n_teeth: float = 16
    n_cutout_teeth: int = 0
    module: float = 1
    pressure_angle: float = 20 * PI / 180
    cone_angle: float = 0
    axis_offset: float = 0
    center: np.ndarray = ORIGIN
    angle: float = 0
    orientation: np.ndarray = np.eye(3)
    h_a: float = 1
    h_d: float = 1.2
    h_o: float = 2
    profile_shift: float = 0.0
    profile_reduction: float = 0.0
    tip_fillet: float = 0.0
    root_fillet: float = 0.0
    tip_reduction: float = 0.0
    inside_teeth: bool = False
    enable_undercut: bool = True

    @property
    def rp(self):
        return self.n_teeth/2*self.module

    @property
    def gamma(self):
        return self.cone_angle/2

    @property
    def R(self):
        return self.rp/np.sin(self.gamma)

    @property
    def pitch_angle(self):
        return 2*PI/self.n_teeth

    @property
    def axis(self):
        return self.orientation[:,2]

    @property
    def x_axis(self):
        return self.orientation[:,0]

    @classmethod
    def null(cls):
        return cls(n_teeth=0,
                n_cutout_teeth=0,
                module=0,
                pressure_angle=0,
                cone_angle=0,
                axis_offset=0,
                center=np.array([0, 0, 0]),
                angle=0,
                orientation=np.zeros((3,3)),
                h_a=0,
                h_d=0,
                h_o=0,
                profile_shift=0,
                profile_reduction=0,
                tip_fillet=0,
                root_fillet=0,
                tip_reduction=0,
                inside_teeth=False,
                enable_undercut=False)



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
                 orientation: np.ndarray = np.eye(3),
                 h_a: float = 1,
                 h_d:float = 1.2,
                 h_o: float = 2,
                 tip_fillet: float = 0.0,
                 root_fillet: float = 0.0,
                 tip_reduction: float = 0.0,
                 inside_teeth: bool = False,
                 **kwargs
                 ) -> None:
        self.n_teeth = n_teeth
        self.n_cutout_teeth = n_cutout_teeth
        self.module = module
        self.cone_angle = cone_angle
        self.axis_offset = axis_offset
        self.center = center
        self.angle = angle
        self.orientation = orientation
        self.h_a = h_a
        self.h_d = h_d
        self.h_o = h_o
        self.tip_fillet = tip_fillet
        self.root_fillet = root_fillet
        self.tip_reduction = tip_reduction
        self.inside_teeth = inside_teeth
        self.tooth_curve = reference_tooth_curve

        self.ra_curve: crv.ArcCurve
        self.rd_curve: crv.ArcCurve
        self.ro_curve: crv.ArcCurve
        self.ra_circle: crv.ArcCurve
        self.rd_circle: crv.ArcCurve
        self.ro_circle: crv.ArcCurve

        self.generate_ref_base_circles()
        self.update_tip()
        self.update_tip_fillet()
        self.update_root_fillet()
        self.generate_profile()

    @property
    def axis(self):
        return self.orientation[:,2]
    
    @property
    def axis_x(self):
        return self.orientation[:,0]
    
    @property
    def axis_y(self):
        return self.orientation[:,1]

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
        return self.R_ref*np.cos(self.gamma)*OUT

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

    # these are post-calculated values that may differ from input parameters
    @property
    def ra(self):
        return self.ra_curve.r*self.module
    @property
    def rd(self):
        return self.rd_curve.r*self.module
    @property
    def ro(self):
        return self.ro_curve.r*self.module

    def base_transform(self,point):
        # rot_axis = np.cross(OUT,self.axis)
        # if np.linalg.norm(rot_axis)<1E-12:
        #     # this can be if axis is OUT or -OUT
        #     if all(abs(self.axis-OUT)<1E-12):
        #         rot_angle = 0
        #         rot_axis = RIGHT
        #     else:
        #         rot_angle = PI
        #         rot_axis = RIGHT
        # else:
        #     rot_angle = angle_between_vectors(OUT,self.axis)
        #     rot_axis = normalize_vector(rot_axis)

        rot_z = scp_Rotation.from_euler('z',self.angle).as_matrix()
        # rot_axis = scp_Rotation.from_rotvec(rot_angle*rot_axis)
        rot_axis = self.orientation

        # return rot_axis@rot_z.apply(point)*self.module + self.center
        return point @ rot_z.transpose() @ rot_axis.transpose() *self.module + self.center

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
        return self.polar_transform(point)[0]
    def r_height_func_spherical(self,point):
        return self.polar_transform(point)[0]

    def generate_ref_base_circles(self):

        p0 = RIGHT*self.rp_ref
        pa = self.inverse_polar_transform(self.polar_transform(p0) + \
                                          np.array([self.h_a,0,0]))
        pd = self.inverse_polar_transform(self.polar_transform(p0) - \
                                          np.array([self.h_d,0,0]))
        if self.inside_teeth:
            po = self.inverse_polar_transform(self.polar_transform(p0) + \
                                              np.array([self.h_o,0,0]))
        else:
            po = self.inverse_polar_transform(self.polar_transform(p0) - \
                                              np.array([self.h_o,0,0]))

        self.rp_circle = crv.ArcCurve.from_point_center_angle(p0=p0,
                                                              center=OUT*p0[2],
                                                              angle=2*PI)
        self.ra_circle = crv.ArcCurve.from_point_center_angle(p0=pa,
                                                              center=OUT*pa[2],
                                                              angle=2*PI)
        self.rd_circle = crv.ArcCurve.from_point_center_angle(p0=pd,
                                                              center=OUT*pd[2],
                                                              angle=2*PI)
        self.ro_circle = crv.ArcCurve.from_point_center_angle(p0=po,
                                                              center=OUT*po[2],
                                                              angle=2*PI)

    def update_tip(self):
        sols = []
        rdh = self.r_height_func(self.rd_circle(0))
        rah = self.r_height_func(self.ra_circle(0))
        for guess in np.linspace(0.1,0.9,4):
            sol1 = crv.find_curve_plane_intersect(self.tooth_curve,
                                                  plane_normal=UP,
                                                  guess=guess)
            r_sol = self.r_height_func(self.tooth_curve(sol1.x[0]))
            if sol1.success and r_sol>rdh:
                sols.append(sol1)

        if len(sols) > 0:
            sol = sols[np.argmin([sol.x[0] for sol in sols])]
            r_sol = self.r_height_func(self.tooth_curve(sol.x[0]))

            if r_sol-self.tip_reduction<rah:
                if self.tip_reduction>0:
                    sol2 = root(lambda t: self.r_height_func(self.tooth_curve(t[0])) - \
                                          r_sol+self.tip_reduction,
                                [sol.x[0]])
                    self.tooth_curve.set_end_on(sol2.x[0])
                else:
                    self.tooth_curve.set_end_on(sol.x[0])
                self.ra_circle = crv.ArcCurve.from_point_center_angle(
                        p0=self.tooth_curve(1),
                        center=self.tooth_curve(1)*np.array([0,0,1]),
                        angle=2*PI
                        )

    def generate_profile(self):

        # if tip fillet is used, tooth curve tip is already settled
        # in fact this solver tends to fail due to tangential nature of fillet
        if not self.tip_fillet>0:
            sol_tip_2 = crv.find_curve_intersect(self.tooth_curve,
                                                 self.ra_circle,
                                                 guess=[1,0],
                                                 method=crv.IntersectMethod.EQUALITY)
            if not sol_tip_2.success:
                # try the other way
                sol_tip_2 = crv.find_curve_intersect(self.tooth_curve,
                                                     self.ra_circle,
                                                     guess=[1,0],
                                                     method=crv.IntersectMethod.MINDISTANCE)
            solcheck = np.linalg.norm(self.tooth_curve(sol_tip_2.x[0]) - \
                                      self.ra_circle(sol_tip_2.x[1]))
            if sol_tip_2.success or solcheck<1E-5:
                self.tooth_curve.set_end_on(sol_tip_2.x[0])
            else:
                sol_mid = crv.find_curve_plane_intersect(self.tooth_curve,
                                                         plane_normal=UP,
                                                         guess=1)
                self.tooth_curve.set_end_on(sol_mid.x[0])

        if not self.root_fillet>0:
            sol_root_1 = crv.find_curve_intersect(self.tooth_curve,
                                                  self.rd_circle,guess=[0.3,-0.01],
                                                  method=crv.IntersectMethod.EQUALITY)
            solcheck = np.linalg.norm(self.tooth_curve(sol_root_1.x[0]) - \
                                      self.rd_circle(sol_root_1.x[1]))
            if not sol_root_1.success:
                # try the other way
                sol_root_2 = crv.find_curve_intersect(self.tooth_curve,
                                                      self.rd_circle,guess=[0.3,-0.01],
                                                      method=crv.IntersectMethod.MINDISTANCE)
                solcheck2 = np.linalg.norm(self.tooth_curve(sol_root_1.x[0]) - \
                                           self.rd_circle(sol_root_1.x[1]))
                if sol_root_2.success or solcheck2<1E-5:
                    solcheck = solcheck2
                    sol_root_1 = sol_root_2

            if sol_root_1.success or solcheck<1E-5:
                self.tooth_curve.set_start_on(sol_root_1.x[0])
            else:
                plane_norm =  rotate_vector(UP,-self.pitch_angle/2)
                sol_mid2 = crv.find_curve_plane_intersect(self.tooth_curve,
                                                          plane_normal=plane_norm,
                                                          guess=0)
                self.tooth_curve.set_start_on(sol_mid2.x[0])

        self.tooth_mirror = crv.MirroredCurve(self.tooth_curve,plane_normal=UP)
        self.tooth_mirror.reverse()
        tooth_rotate = crv.RotatedCurve(self.tooth_mirror,
                                        angle=-self.pitch_angle,
                                        axis=OUT)

        pa1 = self.tooth_curve(1)
        pa2 = self.tooth_mirror(0)
        center_a = ((pa1+pa2)/2*np.array([0,0,1]))*OUT
        self.ra_curve = crv.ArcCurve.from_2_point_center(p0=pa1,
                                                         p1=pa2,
                                                         center=center_a)

        pd1 = self.tooth_curve(0)
        pd2 = tooth_rotate(1)
        center_d = ((pd1+pd2)/2*np.array([0,0,1]))*OUT
        self.rd_curve = crv.ArcCurve.from_2_point_center(p0=pd2,
                                                         p1=pd1,
                                                         center=center_d)

        self.profile = crv.CurveChain(self.rd_curve,
                                      self.tooth_curve,
                                      self.ra_curve,
                                      self.tooth_mirror)
        return self.profile

    def update_tip_fillet(self):
        if self.tip_fillet>0:
            sol1 = crv.find_curve_intersect(self.tooth_curve,
                                            self.ra_circle,
                                            guess=[0.9,0],
                                            method=crv.IntersectMethod.EQUALITY)
            # if sol is found and the intersection is below the x line
            if sol1.success and self.ra_circle(sol1.x[1])[1]<0:
                sharp_tip = False
                guesses = np.asarray([0.5,1,1.5])*self.tip_fillet
                for guess in guesses:
                    start_locations=[sol1.x[0]-guess/self.tooth_curve.length,
                                     sol1.x[1]+guess/self.ra_circle.length]
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.tooth_curve,
                                                          self.ra_circle,
                                                          self.tip_fillet,
                                                          start_locations=start_locations,
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
                    start_locations=[0+self.tip_fillet/self.ra_circle.length,
                                     1-self.tip_fillet/self.ra_circle.length]
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.tooth_curve,
                                                          mirror_curve,
                                                          self.tip_fillet,
                                                          start_locations=start_locations,
                                                          method=crv.IntersectMethod.MINDISTANCE)
                    if sol.success:
                        # this is almost guaranteed to succeed,
                        #  the middle of this arc should be on the x axis
                        # the length-proportion based curve parameterization still
                        #  might make it off by a tiny bit so solver is used instead
                        sol2 = crv.find_curve_plane_intersect(arc,
                                                              plane_normal=UP,
                                                              guess=0.5)
                        arc.set_end_on(sol2.x[0])
                        self.tooth_curve.set_end_on(t1)
                        self.tooth_curve.append(arc)

    def update_root_fillet(self):

        def angle_check(p):
            return angle_between_vector_and_plane(p,UP) < self.pitch_angle/2

        if self.root_fillet>0:
            sol1 = crv.find_curve_intersect(self.tooth_curve,
                                            self.rd_circle,
                                            guess=[0.1,self.pitch_angle/4/(2*PI)*1.01])
            if sol1.success and angle_check(self.rd_circle(sol1.x[1])):
                sharp_root = False
                guesses = np.asarray([0.5,1,1.5])*self.root_fillet
                for guess in guesses:
                    start_locations=[sol1.x[1]-guess/self.rd_circle.length,
                                     sol1.x[0]+guess/self.tooth_curve.length]
                    arc, t1,t2,sol = crv.calc_tangent_arc(self.rd_circle,
                                                        self.tooth_curve,
                                                        self.root_fillet,
                                                        start_locations=start_locations)
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
                plane_normal=rotate_vector(UP,-self.pitch_angle/2)
                mirror_curve = crv.MirroredCurve(self.tooth_curve,
                                                 plane_normal=plane_normal)
                mirror_curve.reverse()
                start_locations=[1-self.root_fillet/self.tooth_curve.length,
                                 0+self.root_fillet/self.tooth_curve.length]
                arc, t1,t2,sol = crv.calc_tangent_arc(mirror_curve,
                                                      self.tooth_curve,
                                                      self.root_fillet,
                                                      start_locations=start_locations)
                if sol.success:
                    plane_normal=rotate_vector(UP,-self.pitch_angle/2)
                    sol2 = crv.find_curve_plane_intersect(arc,
                                                          plane_normal=plane_normal,
                                                          guess=0.5)
                    arc.set_start_on(sol2.x[0])
                    self.tooth_curve.set_start_on(t2)
                    self.tooth_curve.insert(0,arc)

    def generate_profile_closed(self,rd_coeff_right=1.0,rd_coeff_left=0.0):
        # mirroring but making sure its a distinct curve,
        #   MirroredCurve remains linked to the original
        rd_curve_left = crv.ArcCurve.from_2_point_center(
                p0=self.rd_curve(1)*np.array([1,-1,1]),
                p1=self.rd_curve(0)*np.array([1,-1,1]),
                center=self.rd_curve.center)
        rd_curve_left.set_end_on(rd_coeff_left)
        if rd_coeff_left>0:
            rd_curve_left.active=True
        else:
            rd_curve_left.active=False

        # right-side rd curve is the original rd curve so it exists already
        if rd_coeff_right<1:
            self.rd_curve.set_start_on(1-rd_coeff_right)
            if rd_coeff_right<=0:
                self.rd_curve.active=False

        v0 = np.cross(self.profile(0),OUT)
        v1 = np.cross(rd_curve_left(1),OUT)
        sol0 = crv.find_curve_plane_intersect(self.ro_circle,v0,guess=0)
        sol1 = crv.find_curve_plane_intersect(self.ro_circle,v1,guess=0)

        p0 = self.ro_circle(sol0.x[0])
        p1 = self.ro_circle(sol1.x[0])

        if self.cone_angle==0:
            self.ro_connector_1 = crv.LineCurve(rd_curve_left(1),p1)
            self.ro_connector_0 = crv.LineCurve(p0,self.profile(0))
        else:
            self.ro_connector_1 = crv.ArcCurve.from_2_point_center(p0=rd_curve_left(1),
                                                           p1=p1,
                                                           center=self.center_sphere_ref)
            self.ro_connector_0 = crv.ArcCurve.from_2_point_center(p0=p0,
                                                           p1=self.profile(0),
                                                           center=self.center_sphere_ref)

        self.ro_curve = crv.ArcCurve.from_2_point_center(p0=p1,
                                                         p1=p0,
                                                         center=self.ro_circle.center)

        profile_closed = crv.CurveChain(self.profile,
                                        rd_curve_left,
                                        self.ro_connector_1,
                                        self.ro_curve,
                                        self.ro_connector_0)

        return profile_closed

    def generate_gear_pattern(self,profile:crv.Curve):
        def func(t):
            t2,k = self.tooth_moduler(t)
            p = profile(t2)
            return self.base_transform(
                scp_Rotation.from_euler('z',k*self.pitch_angle).apply(p))
        return crv.Curve(func,0,1)

    def tooth_moduler(self,t):
        t2 = ((np.floor(self.n_teeth)-self.n_cutout_teeth)*t)
        return t2%1, t2//1


class InvoluteFlankGenerator():
    def __init__(self,
                pitch_angle: float = 2 * PI / 16,
                cone_angle: float = 0.0,
                alpha: float = 20.0 * PI / 180,
                profile_shift: float = 0.0,
                profile_reduction: float = 0.0,
                h_d: float = 1.2,
                enable_undercut: bool = True,
                **kwargs
                ):
        self.pitch_angle = pitch_angle
        self.cone_angle = cone_angle
        self.alpha = alpha
        self.profile_shift = profile_shift
        self.profile_reduction = profile_reduction
        self.h_d = h_d
        self.enable_undercut = enable_undercut

        self.involute_curve = crv.InvoluteCurve(active=False)
        self.involute_connector = crv.LineCurve(active=False)
        self.undercut_curve = crv.InvoluteCurve(active=False)

        self.involute_curve_sph = crv.SphericalInvoluteCurve(active=False)
        self.involute_connector_arc = crv.ArcCurve(active=False)
        self.undercut_curve_sph = crv.SphericalInvoluteCurve(active=False)



        if self.cone_angle==0:
            self.rd = self.rp - self.h_d + self.profile_shift
            self.calculate_involutes_cylindric()
            self.tooth_curve = crv.CurveChain(self.undercut_curve,
                                              self.involute_connector,
                                              self.involute_curve)
        else:
            # gamma is cone angle / 2 property
            self.R = self.rp/np.sin(self.gamma)
            self.C_sph = 1/self.R
            self.center = OUT*np.sqrt(self.R**2-self.rp**2)
            self.an_d = (self.h_d-self.profile_shift ) /self.R
            # 180deg cone is a flat circle... leads to similar result like
            #  infinite radius cylinder... which would be a straight rack
            if self.cone_angle==PI:
                self.tooth_curve = self.calculate_rack_spherical()
            else:
                self.calculate_involutes_spherical()
                self.tooth_curve = crv.CurveChain(self.undercut_curve_sph,
                                                  self.involute_connector_arc,
                                                  self.involute_curve_sph)

    @property
    def rp(self):
        return PI/self.pitch_angle

    @property
    def gamma(self):
        return self.cone_angle/2


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
        da = self.profile_shift * np.tan(self.alpha) / self.rp - \
            self.profile_reduction/self.rp
        # angle to move the involute into standard construction position
        # by convention moving clockwise, which is negative angular direction
        # the tooth shall be symmetrical on the x-axis,
        #  so the base angle is quarter of pitch angle
        # added angular components to compensate for profile shift and the
        #  involute curve's travel from base to pitch circle
        self.involute_curve.angle = -(self.pitch_angle / 4 + involute_angle_0 + da)

        # hence the tooth shall be on the x axis,
        #  the involute shall not cross the x axis
        # find the point where the involute curve reaches the x axis,
        #  that shall be the end of the segment
        sol1 = crv.find_curve_plane_intersect(self.involute_curve,
                                              plane_normal=UP,
                                              guess=1)
        self.involute_curve.t_1 = self.involute_curve.p2t(sol1.x[0])
        self.involute_curve.update_lengths()

        if self.rd<self.rb:
            p_invo_base = self.involute_curve(0)
            p_invo_d = p_invo_base*self.rd/self.rb

            self.involute_connector.p0 = p_invo_d
            self.involute_connector.p1 = p_invo_base
            self.involute_connector.active=True
            if self.enable_undercut:
                # when undercut is used,
                #  there is no line between undercut and involute in 2D
                self.involute_connector.active=False
                self.undercut_curve.active=True
                # the undercut is an involute curve with an offset vector
                #  (sometimes called trochoid)
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

                # find intersection of undercut curve and involute curve,
                # might need multiple guesses from different starting points
                guess = 0.1
                for k in range(10):
                    sol1 = crv.find_curve_intersect(loc_curve,
                                                    self.undercut_curve,
                                                    guess=[guess+t_invo,guess])
                    # loc curve is the involute with a straight line down
                    # the undercut curve will cross it at 2 points
                    # need to find the point that hits the involute part, not the line
                    if abs(sol1.x[0])>t_invo and sol1.success:
                        break
                    guess = (k+1) * 0.1

                loc_curve.set_start_on(sol1.x[0])
                # find lowest point of ucut
                sol2 = minimize(lambda t: np.linalg.norm(self.undercut_curve(t)),0)
                self.undercut_curve.set_start_and_end_on(sol2.x[0],sol1.x[1])
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
            angle = angle_between_vectors(tan,sph_tan)

            return [p0[0]**2+p0[1]**2-self.rp**2, angle-PI/2-self.alpha]

        self.involute_curve_sph.active=True
        base_res = root(involute_angle_func,
                        [self.alpha/2,self.rp*np.cos(self.alpha)],
                        tol=1E-14)
        self.rb = base_res.x[1]

        self.involute_curve_sph.r = self.rb
        self.involute_curve_sph.c_sphere = self.C_sph

        angle_0 = angle_between_vectors(
                    involute_sphere(base_res.x[0],
                                    self.rb,angle=0,
                                    C=self.C_sph)*np.array([1,1,0]),
                    RIGHT)
        angle_offset =  - (self.pitch_angle/4 + \
                           self.profile_shift * np.tan(self.alpha) / self.rp) + \
                           self.profile_reduction/self.rp - \
                        angle_0
        self.involute_curve_sph.angle = angle_offset
        self.involute_curve_sph.z_offs = -involute_sphere(base_res.x[0],
                                                          base_res.x[1],
                                                          C=self.C_sph)[2]
        self.involute_curve_sph.t_0 = 0
        self.involute_curve_sph.t_1 = 1
        sol1 = crv.find_curve_plane_intersect(self.involute_curve_sph,
                                              offset=ORIGIN,
                                              plane_normal=UP,guess=1)
        self.involute_curve_sph.set_end_on(sol1.x[0])


        ## undercut
        p0=self.involute_curve_sph(0)
        axis = normalize_vector(np.cross(p0,OUT))
        # by convention the pitch circle is in the x-y plane
        # the involute goes partially below the pitch circle
        # calculate the angle to go until the dedendum circle
        p0_xy = (p0-self.center)*np.array([1,1,0])
        an_diff = self.an_d - \
                  angle_between_vectors(p0-self.center,p0_xy) + \
                  (PI/2-self.gamma)
        if an_diff<0:
            self.involute_connector_arc.active=False
            self.undercut_curve_sph.active=False
        else:
            p1 = self.center + \
                 scp_Rotation.from_rotvec(-axis*an_diff).apply(p0-self.center)
            self.involute_connector_arc = \
                crv.ArcCurve.from_2_point_center(p0=p1,p1=p0,
                                                 center=self.center)
            self.involute_connector_arc.active=True

            if not self.enable_undercut:
                self.undercut_curve_sph.active=False
            else:
                self.undercut_curve_sph.active=True
                ref_rack = self.calculate_rack_spherical()
                self.undercut_curve_sph.r = self.rp
                self.undercut_curve_sph.angle = 0
                self.undercut_curve_sph.z_offs = 0
                self.undercut_curve_sph.v_offs = scp_Rotation.from_euler(
                    'y',PI/2 * np.sign(self.C_sph)).apply(ref_rack(0)-self.R*RIGHT)
                self.undercut_curve_sph.c_sphere = self.C_sph
                self.undercut_curve_sph.t_0 = 1
                self.undercut_curve_sph.t_1 = -1

                self.undercut_curve_sph.update_lengths()


                loc_curve = crv.CurveChain(self.involute_connector_arc,
                                           self.involute_curve_sph)
                rb_curve = crv.ArcCurve.from_2_point_curvature(
                    p0=self.involute_curve_sph(0),
                    p1=self.involute_curve_sph(0)*np.array([1,-1,1]),
                    curvature=1/self.involute_curve_sph.r,
                    revolutions=0)

                sol0 = crv.find_curve_intersect(self.undercut_curve_sph,
                                                rb_curve,
                                                guess=[0.1,0])
                for guess in np.linspace(0.1,0.9,4):
                    sol1 = crv.find_curve_intersect(loc_curve,
                                                    self.undercut_curve_sph,
                                                    guess=[0.3,sol0.x[0]+guess],
                                                    method=crv.IntersectMethod.EQUALITY)
                    #direction check
                    d1 = self.undercut_curve_sph.derivative(sol1.x[1])
                    d2 =loc_curve.derivative(sol1.x[0])
                    solcheck = np.dot(np.cross(d1,d2),
                                      self.undercut_curve_sph(sol1.x[1])-self.center)
                    solcheck2 = np.linalg.norm(self.undercut_curve_sph(sol1.x[1]) - \
                                               loc_curve(sol1.x[0]))
                    if solcheck<0 and solcheck2<1E-7:
                        break
                loc_curve.set_start_on(sol1.x[0],preserve_inactive_curves=True)
                self.undercut_curve_sph.set_end_on(sol1.x[1])

                sol2 = minimize(lambda t:np.linalg.norm(self.undercut_curve_sph(t)[:2]),
                                0)
                self.undercut_curve_sph.set_start_on(sol2.x[0])


    def calculate_rack_spherical(self):
        def rack_flanc_func(t,a):
            axis1 = scp_Rotation.from_euler('x',-self.alpha).apply(OUT)
            v0 = self.R*RIGHT
            an1 = t*np.cos(self.alpha)
            v1 = scp_Rotation.from_rotvec(-axis1*an1).apply(v0)
            v2 = scp_Rotation.from_euler('z',t+a).apply(v1)
            return v2

        an_tooth_sph = self.rp / self.R * \
            (self.pitch_angle/2 + self.profile_shift*np.tan(self.alpha) *2/self.rp - \
             self.profile_reduction*2/self.rp)
        curve1 = crv.Curve(rack_flanc_func,
                           t0=-1,
                           t1=1,
                           params={'a':-an_tooth_sph/2})

        sol2 = root(lambda t: np.arcsin(curve1(t[0])[2]/self.R)+self.an_d,[0])

        sol1 = crv.find_curve_plane_intersect(curve1,plane_normal=UP,guess=1)
        curve1.set_start_and_end_on(sol2.x[0],sol1.x[0])
        return curve1


@dataclasses.dataclass
class InvoluteGearParamManager(InvoluteGearParam, ZFunctionMixin):
    z_vals: np.ndarray = np.array([0,1])

class InvoluteGear():
    def __init__(self,
                 params : InvoluteGearParamManager = None,
                 **kwargs):
        # use defaults
        if params is None:
            params=InvoluteGearParamManager()
        self.params = params
        self.paramref = self.params(0)
        self.z_vals = self.params.z_vals

    def setup_generator(self,params):
        paramdict = params.__dict__
        tooth_curve = self.get_tooth_curve(
            pitch_angle = 2*PI/params.n_teeth,
            cone_angle = params.cone_angle,
            alpha = params.pressure_angle,
            profile_shift = params.profile_shift,
            profile_reduction = params.profile_reduction,
            h_d = params.h_d,
            enable_undercut = params.enable_undercut)

        paramdict['h_a'] = paramdict['h_a']+paramdict['profile_shift']
        paramdict['h_d'] = paramdict['h_d']-paramdict['profile_shift']

        curve_generator = GearCurveGenerator(reference_tooth_curve=tooth_curve,
                                             **paramdict)
        return curve_generator

    def curve_gen_at_z(self,z):
        return self.setup_generator(self.params(z))

    # somehow this is slower with cache!!! wtf
    # @lru_cache(maxsize=12)
    def get_tooth_curve(self,
                        pitch_angle,
                        cone_angle,
                        alpha ,
                        profile_shift,
                        profile_reduction,
                        h_d,
                        enable_undercut):
        return InvoluteFlankGenerator(
            pitch_angle = pitch_angle,
            cone_angle = cone_angle,
            alpha = alpha,
            profile_shift = profile_shift,
            profile_reduction = profile_reduction,
            h_d = h_d,
            enable_undercut = enable_undercut).tooth_curve

    def mesh_to(self,other:'InvoluteGear',
                target_dir=RIGHT,
                distance_offset=0):
        '''
        Move this gear into a meshing position with other gear,
        so that the point of contact of the pitch circles is in target_dir direction.
        '''
        target_dir_norm = target_dir - \
            np.dot(target_dir,other.paramref.axis)*other.paramref.axis
        if np.linalg.norm(target_dir_norm)<1E-12:
            # target_dir is parallel to x axis
            target_dir_norm = other.paramref.x_axis
        else:
            target_dir_norm = normalize_vector(target_dir_norm)

        target_plane_norm = np.cross(other.paramref.axis,target_dir_norm)

        target_angle_other = angle_between_vectors(other.paramref.x_axis,
                                                   target_dir_norm) * \
                            np.sign(np.dot(np.cross(other.paramref.x_axis,
                                                    target_dir_norm),
                                            other.paramref.axis))
        target_phase_other = ((target_angle_other-other.paramref.angle) / other.paramref.pitch_angle)%1


        if self.paramref.gamma==0 and other.paramref.gamma==0:
            # both are cylindrical
            self.params.orientation=other.params.orientation

            if self.paramref.inside_teeth or other.paramref.inside_teeth:
                phase_offset = 0
                angle_turnaround = 0
                phase_sign = -1
            else:
                phase_offset = 0.5
                angle_turnaround = PI
                phase_sign = 1

            target_angle_self = target_angle_other + angle_turnaround
            angle_offs = target_angle_self +\
                         (phase_sign*(target_phase_other)-phase_offset)*self.paramref.pitch_angle
            r1 = self.paramref.rp + self.paramref.profile_shift * self.paramref.module
            r2 = other.paramref.rp + other.paramref.profile_shift * other.paramref.module
            if self.paramref.inside_teeth:
                distance_ref = r2 - r1 + distance_offset
            elif other.paramref.inside_teeth:
                distance_ref = r1 - r2 - distance_offset
            else:
                distance_ref = r1 + r2 + distance_offset

            center_offs = distance_ref*target_dir_norm
            params_upd = InvoluteGearParamManager.null()
            params_upd.z_vals = self.params.z_vals
            params_upd.angle = angle_offs
            params_upd.center = center_offs
            self.params = self.params + params_upd

        elif self.paramref.gamma!=0 and other.paramref.gamma!=0:
            # both are spherical
            # start off by identical orientation
            self.params.orientation=other.params.orientation
            # angle-phase math is the same as cylindrical
            if self.paramref.inside_teeth or other.paramref.inside_teeth:
                phase_offset = 0
                angle_turnaround = 0
                phase_sign = -1
            else:
                phase_offset = 0.5
                angle_turnaround = PI
                phase_sign = 1

            target_angle_self = target_angle_other + angle_turnaround
            angle_offs = target_angle_self +\
                         (phase_sign*(target_phase_other)-phase_offset)*self.paramref.pitch_angle
            r1 = self.paramref.rp + \
                self.paramref.profile_shift * self.paramref.module
            r2 = other.paramref.rp + \
                other.paramref.profile_shift * other.paramref.module
            
            # compatible bevel gears should have the same spherical radius
            # and the same center sphere when placed on xy plane at the orgin

            if self.paramref.inside_teeth:
                distance_ref = r2 - r1 + distance_offset
            elif other.paramref.inside_teeth:
                distance_ref = r1 - r2 - distance_offset
            else:
                distance_ref = r1 + r2 + distance_offset
            
            # angle_ref = distance_ref / self.paramref.R
            angle_ref =self.paramref.gamma + other.paramref.gamma
            center_sph = np.sqrt(self.paramref.R**2-self.paramref.rp**2) * OUT
            center_sph_other = np.sqrt(other.paramref.R**2-other.paramref.rp**2) * OUT
            rot1 = scp_Rotation.from_rotvec(-target_plane_norm*angle_ref)
            center_offs = rot1.apply(-center_sph)+center_sph_other
            params_upd = InvoluteGearParamManager.null()
            params_upd.z_vals = self.params.z_vals
            params_upd.angle = angle_offs
            params_upd.center = center_offs
            self.params = self.params + params_upd
            self.params.orientation = other.params.orientation @ rot1.as_matrix()
            self.params.center = lambda z: self.params.module(0)*z*rot1.as_matrix()[:,2]*np.cos(self.paramref.gamma)+center_offs
            
        else:
            # one is cylindrical, the other is spherical
            Warning('Meshing cylindrical and spherical gears are not supported')
