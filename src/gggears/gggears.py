from function_generators import *
from curve import *
from defs import *
from scipy.optimize import root
from scipy.optimize import minimize
from functools import cache
import dataclasses


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


class GearProfile2D(InvoluteProfileParam):
    def __init__(self,
                 **kwargs
                 ):
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
        self.tip_circle =   Curve(lambda t: self.ra*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.root_circle =  Curve(lambda t: self.rd*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.pitch_circle = Curve(lambda t: self.rp*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.base_circle =  Curve(lambda t: self.rb*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))

        # empty curves representing tooth flank segments
        self.involute_curve = InvoluteCurve()
        self.undercut_curve = InvoluteCurve()
        self.undercut_connector_line = LineCurve(active=False)

        # curve representing one side of the tooth, without addendum and dedendum
        self.tooth_curve = CurveChain(self.undercut_curve, self.undercut_connector_line,self.involute_curve)

        # addendum and dedendum arcs
        self.rd_curve = ArcCurve.from_2_point_center(p0=rotate_vector(self.rd*RIGHT,-self.pitch_angle/2).reshape(VSHAPE),
                                                     p1=self.rd*RIGHT,
                                                     center=ORIGIN)
        self.ra_curve = ArcCurve.from_2_point_center(p1=self.ra*RIGHT,
                                                     p0=rotate_vector(self.ra*RIGHT,-self.pitch_angle/2).reshape(VSHAPE),
                                                     center=ORIGIN)
        # perform math calculations to generate the tooth profile
        self.calculate_involute_base()
        self.calculate_uncercut()
        self.calculate_arcs()

        self.tooth_mirror = CurveChain(*[MirroredCurve(curve,plane_normal=UP) for curve in self.tooth_curve])
        self.tooth_mirror.reverse()
        self.profile = CurveChain(self.rd_curve,
                                  *self.tooth_curve,
                                  self.ra_curve,
                                  *self.tooth_mirror)
        
    def calculate_involute_base(self):
        # setup base circle
        self.involute_curve.r = self.rb

        # find the angular position of the involute curve start point
        # this is the point where the involute curve intersects the pitch circle
        sol2 = find_curve_intersect(self.involute_curve,self.pitch_circle)
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
        x_line = Curve(arc_from_2_point,params={'p0':ORIGIN,'p1':self.ra*2*RIGHT,'curvature':0})
        sol1 = find_curve_intersect(self.involute_curve,x_line)
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
            sol1 = find_curve_intersect(self.tooth_curve,self.rd_curve)
            self.tooth_curve=self.tooth_curve.cut(sol1.x[0],preserve_inactive_curves=True)[1]
            
            if self.root_fillet>0:
                # rd_cut_curve = self.rd_curve.cut(sol1.x[1])[0]
                # prep_chain = CurveChain(rd_cut_curve,self.tooth_curve)
                # location = prep_chain.get_t_for_index(0)[1]
                # temp = fillet_curve(prep_chain,self.root_fillet,location)
                # # self.undercut_curve = temp.get_curves()[1]
                # # self.undercut_connector_line.p0 = self.undercut_curve(1)
                # # self.tooth_curve.update_lengths()
                # self.tooth_curve = CurveChain(*temp.get_curves()[1:])
                rd_cut_curve = self.rd_curve.cut(sol1.x[1])[0]
                # prep_chain = CurveChain(rd_cut_curve,self.tooth_curve)
                # location = prep_chain.get_t_for_index(0)[1]

                # self.undercut_curve, t1, t2 = fillet_curve(prep_chain,self.root_fillet,location
                self.tooth_curve.insert(0,rd_cut_curve)
                location = self.tooth_curve.get_t_for_index(0)[1]
                self.tooth_curve = self.tooth_curve.fillet(self.root_fillet,location)
                self.tooth_curve.pop(0)
                
                 

            else:
                # add placeholder of undercut curve
                # self.undercut_curve=Curve(arc_from_2_point,params={'p0': self.tooth_curve(0),
                #                                                    'p1': self.tooth_curve(0),
                #                                                    'curvature':0},
                #                           active=False)
                # self.tooth_curve.insert(0,self.undercut_curve)
                self.undercut_curve.active=False


    def calculate_arcs(self):
        sol1 = find_curve_intersect(self.tooth_curve,self.ra_curve)
        guesses = [0.1,0.5,0.9]
        sol2 = []
        for guess in guesses:
            sol2.append(find_curve_line_intersect(self.tooth_curve,offset=ORIGIN,line_direction=RIGHT, guess=guess))
        ttooth1 = sol1.x[0]
        # ttooth2 = sol2.x[0]
        ttooth2 = np.min([sol.x[0] for sol in sol2 if sol.success])
        if ttooth1<ttooth2:
            self.ra_curve = self.ra_curve.cut(sol1.x[1])[1]
            self.tooth_curve = self.tooth_curve.cut(sol1.x[0])[0]
            # reconfigure ra_curve
            # p2 = self.ra_curve(0) * np.array([1,-1,1])
            # self.ra_curve.params['p0'] = self.ra_curve(0)
            # self.ra_curve.params['p1'] = p2
            # self.ra_curve.t_0,self.ra_curve.t_1 = 0,1
            p0 = self.ra_curve(0)
            p1 = self.ra_curve(0) * np.array([1,-1,1])
            self.ra_curve = ArcCurve.from_2_point_center(p0=p0,p1=p1,center=ORIGIN)

        else:
            self.tooth_curve = self.tooth_curve.cut(ttooth2)[0]
            self.ra_curve.active=False
            self.ra = np.linalg.norm(self.tooth_curve(1))


        p1 = self.tooth_curve(0)
        angle1 = abs(angle_between_vectors(RIGHT,p1))
        angle0 = self.pitch_angle-angle1
        p0 = rotate_vector(RIGHT*self.rd,-angle0)
        self.rd_curve=ArcCurve.from_2_point_center(p0,p1,ORIGIN)

        # self.tooth_curve.insert(0,self.rd_curve)




class GearProfileSpherical():
    def __init__(self,
                 pitch_angle = 2 * PI / 16,
                 alpha=20,
                 gamma = 45,
                 h_a=1,
                 h_d=1.2,
                 profile_shift=0,
                 profile_reduction=0,
                 fillet_radius=0,
                 **kwargs # just to catch unexpected kwargs error
                 ):
        
        # self.module = 1
        # the profile will be calculated to unit module, and then post-scaled when used in gear generation.
        self.alpha = alpha*np.pi/180
        self.gamma = gamma*np.pi/180
        self.r_fillet = fillet_radius

        self.involute_params = {}
        self.undercut_params = {}
        self.gearparam = {}


        # angular period of teeth
        self.pitch_angle = pitch_angle
        
        # D = m*z
        # z=2pi/pitch_angle
        # r = D/2 = m*2pi/pitch_angle/2

        self.rp = PI/pitch_angle
        # pressure angle
        self.alpha = alpha*DEG2RAD
        self.C_sph = self.rp/np.sin(self.gamma)
        self.R = 1/self.C_sph

        # addendum and dedendum coefficients
        self.h_a = h_a
        self.h_d = h_d
        
        # base circle of involute function: needs to be solved later
        self.rb = 0

        self.X = profile_shift
        self.B = profile_reduction

        # self.h_d_min = PI / 4 / np.tan(self.alpha)
        # self.tooth_h = (self.h_a + self.h_d)

        # if self.h_d > self.h_d_min:
        #     self.h_d = self.h_d_min
        # self.ra = self.rp +  (self.h_a + profile_shift)

        # self.rd = self.rp - (self.h_d - profile_shift)

        self.tip_circle =   Curve(lambda t: self.ra*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.root_circle =  Curve(lambda t: self.rd*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.pitch_circle = Curve(lambda t: self.rp*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))
        self.base_circle =  Curve(lambda t: self.rb*np.array((np.cos(t),np.sin(t),0)).reshape(VSHAPE))

        self.involute_curve = Curve(involute_sphere,params=self.involute_params)
        self.undercut_curve = Curve(involute_circle,params=self.undercut_params)
        self.tooth_curve = CurveChain(self.involute_curve)
        self.rd_curve = Curve(arc_from_2_point,params={'p1':self.rd*RIGHT,
                                                 'p0':rotate_vector(self.rd*RIGHT,-self.pitch_angle/2).reshape(VSHAPE),
                                                 'curvature':1/self.rd})
        self.ra_curve = Curve(arc_from_2_point,params={'p1':self.ra*RIGHT,
                                                 'p0':rotate_vector(self.ra*RIGHT,-self.pitch_angle/2).reshape(VSHAPE),
                                                 'curvature':1/self.ra})
        self.calculate_involute_base()
        self.calculate_uncercut()
        self.calculate_arcs()

        self.tooth_mirror = Curve(lambda t: self.tooth_curve(1-t)*np.array([1,-1,1]).reshape(VSHAPE))
        self.profile = CurveChain(self.tooth_curve,self.tooth_mirror)

        def calculate_ref_rack_disc(self):
            # pitch_angle = 2pi / z
            # D = m*z; rp = m*z/2
            # dra = h_a*m = R*dan_a 
            # dan_a = h_a*m/R
            # dan_d = h_d*m/R

            an_a = (self.h_a+self.X ) /self.R
            an_d = (self.h_d-self.X ) /self.R

            h_a = self.R*np.sin(an_a)
            h_d = self.R*np.sin(an_d)

            p0 = RIGHT*self.R
            p1 = p0+scp_Rotation.from_euler('x',-self.alpha).apply(DOWN)
            p1 = normalize_vector(p1)*self.R

            curve_1 = Curve(arc_from_2_point, params={'p0':p0, 'p1':p1,'curvature':1/self.R})
            sol1 = root(lambda t: curve_1(t)[1]-h_a,[1])
            sol2 = root(lambda t: curve_1(t)[1]+h_d,[1])
            curve_1.params.p1, curve_1.params.p2  = curve_1(sol1.x[0]), curve_1(sol2.x[0])
            curve_1.update_lengths()
            
            an_tooth_sph = (self.pitch_angle/2 + self.X*np.tan(self.alpha) /self.rp )* self.rp / self.R

            # an_shift = self.X/self.R

            # mirror y values
            p2,p3 = p1*np.array([1,-1,1]), p0*np.array([1,-1,1])
            # shift (rotate) for other side of tooth + handling profile shift up
            p2 = (scp_Rotation.from_euler('z',an_tooth_sph/2)).apply(p2)
            p3 = (scp_Rotation.from_euler('z',an_tooth_sph/2)).apply(p3)
            p0 = (scp_Rotation.from_euler('z',-an_tooth_sph/2)).apply(p0)
            p1 = (scp_Rotation.from_euler('z',-an_tooth_sph/2)).apply(p1)

            curve_1 = Curve(arc_from_2_point_center(p0=p0,p1=p1,center=ORIGIN))
            curve_2 = Curve(arc_from_2_point_center(p0=p1,p1=p2,center=ORIGIN))
            curve_3 = Curve(arc_from_2_point_center(p0=p2,p1=p3,center=ORIGIN))

            self.ref_rack_curve=CurveChain(curve_1,curve_2,curve_3)

            

            pass

        def calculate_involute_base(self):
            def involute_angle_func(x):
                t=x[0]
                r=x[1]
                p0=involute_sphere(t,r,0,C=self.C_sph)
                p1=involute_sphere(t+DELTA,r,0,C=self.C_sph)
                p2=involute_sphere(t-DELTA,r,0,C=self.C_sph)
                tan = normalize_vector(p1-p2)
                center = np.array([0,0,np.sqrt(self.R**2-r**2)])
                sph_tan = normalize_vector(np.cross(p0-center,np.array([p0[0],p0[1],0])))
                # angle = np.arctan2(np.linalg.norm(np.cross(tan,sph_tan)),np.dot(tan,sph_tan))
                angle = angle_between_vectors(tan,sph_tan)

                return [np.sqrt(p0[0]**2+p0[1]**2)-self.rp, angle-PI/2-self.alpha]
        
            base_res = root(involute_angle_func,[self.alpha/2,self.rp*np.cos(self.alpha)])
            self.rb = base_res.x[1]
            self.involute_param['r']=self.rb
            angle_0 = base_res.x[0]
            angle_offset = -angle_0 - (self.pitch_angle/2 + self.X*np.tan(self.alpha) /self.rp)
            self.involute_param['z_offs'] = -involute_sphere(base_res.x[0],base_res.x[1],C=self.C_sph)
            self.involute_param['angle'] = angle_offset



        def calcuate_undercut(self):
            pass
        def calculate_arcs(self):
            pass






@dataclasses.dataclass
class GearParam2D():
    num_of_teeth: float = 16
    cutout_teeth_num: int = 0
    angle: float = 0
    center: np.ndarray = ORIGIN
    module: float = 1
    h_o: float = 1
    inside_teeth: bool =False
    profile_overlap: float = 0
    profile_param: InvoluteProfileParam = InvoluteProfileParam() # using all default values of the class

    def __post_init__(self):
        if self.num_of_teeth<0:
            self.num_of_teeth = np.abs(self.num_of_teeth)
            self.inside_teeth = not(self.inside_teeth)
        self.cutout_teeth_num = int(abs(self.cutout_teeth_num))




class Gear2D(GearParam2D):
    '''Class used for representing and handling 2D gears'''
    def __init__(self,
                 **kwargs 
                 ):
        super().__init__(**kwargs)
        
        self.rp = self.m*self.num_of_teeth
        
        # pitch angle has 2 roles and thus part of 2 classes
        # Gear2D needs to know it to be able to rotate exactly 1 tooth over
        # GearProfile2D needs to know it as a parameter of the profile
        self.pitch_angle = 2*PI/self.num_of_teeth
        self.profile_param.pitch_angle = self.pitch_angle

        self.profile_reference = GearProfile2D(**self.profile_param.__dict__)
        
        self.ra = self.profile_reference.ra * self.m
        self.rd = self.profile_reference.rd * self.m
        if self.inside_teeth:
            self.ro = self.ra+self.h_o*self.m
        else:
            self.ro = self.rd-self.h_o*self.m

        self.teeth_curve = Curve(self.teeth_generator,t1=(self.num_of_teeth-self.cutout_teeth_num)/self.num_of_teeth)
        # self.teeth_curve.t_1 = (self.num_of_teeth-self.cutout_teeth_n)/self.num_of_teeth
        self.r_d_padding = Curve(lambda t: arc_from_2_point_center(t, p0=self.teeth_curve(1),p1=self.teeth_curve(0), center=self.center))
        if self.r_d_padding.length<DELTA/10:
            self.r_d_padding.active=False
        self.boundary = CurveChain(self.teeth_curve,self.r_d_padding)

        # profile of 1 tooth to be repeated, but with applied rotation, module scaling and center-offset
        self.profile = CurveChain(*[Curve(self.base_curve_transform,params={'curve': curve}, active=curve.active)
                                    for curve in self.profile_reference.profile.get_curves()])
        self.profile_closed = self.generate_profile_closed()
    
    # shorthand for module
    @property
    def m(self):
        return self.module

    def teeth_generator(self,t):
        z = np.floor(self.num_of_teeth)
        t2 = (t*z)%1
        angle_mod = np.floor(t*z)
        rotmat = scp_Rotation.from_euler('z',angle_mod*self.pitch_angle+self.angle)
        return rotmat.apply(self.profile_reference.profile(t2)*self.m)+self.center
    
    def base_curve_transform(self,t,curve=Curve(lambda t: ORIGIN)):
        return scp_Rotation.from_euler('z',self.angle).apply(curve(t))*self.m+self.center
    
    def generate_profile_closed(self):
        self.profile_closed = CurveChain(*self.profile.copy()[0:])
        self.profile_closed[0].t_0 = -self.profile_overlap
        self.profile_closed.update_lengths()
        p1 = self.profile_closed(1)
        p0 = self.profile_closed(0)
        p1_o = p1+normalize_vector(p1-self.center)*(self.ro-np.linalg.norm(p1-self.center))
        p0_o = p0+normalize_vector(p0-self.center)*(self.ro-np.linalg.norm(p0-self.center))

        self.profile_closed.extend([Curve(arc_from_2_point, params={'p0':p1,'p1':p1_o,'curvature':0}),
                                    Curve(arc_from_2_point_center, params={'p0':p1_o,'p1':p0_o,'center':self.center}),
                                    Curve(arc_from_2_point, params={'p0':p0_o,'p1':p0,'curvature':0})])
        
        return self.profile_closed



class InvoluteProfileHandler(InvoluteProfileParam,ZFunctionMixin):
    pass


class GearParamHandler(GearParam2D,ZFunctionMixin):
    center: lambda z: OUT*z
    profile_param: InvoluteProfileHandler = InvoluteProfileHandler()
    

class GearCylindric():
    def __init__(self,

                 z_vals = np.array([0,1]),
                 params : GearParamHandler = GearParamHandler(),
                 **kwargs):
        
        self.params = params
        self.pitch_angle = 2*PI/self.params.num_of_teeth
        self.params.profile_param.pitch_angle = self.pitch_angle

        self.z_vals = z_vals

    
    @cache
    def generate_gear_slice(self,z) -> Gear2D:
        paramdict = self.params(z).__dict__
        return Gear2D(**paramdict)
    
    def gen_point(self,z,t):
        return self.generate_gear_slice(z).profile(t)