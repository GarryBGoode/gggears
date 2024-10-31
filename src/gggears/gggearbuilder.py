import numpy as np
from defs import *
from function_generators import *
import curve as crv
from scipy.optimize import root
from scipy.optimize import minimize
import dataclasses
from typing import Protocol
from collections.abc import Callable
import warnings 

class CurveBuilderBase(Protocol):
    def gen_curve(self) -> crv.Curve:
        pass

class NullBuilder(CurveBuilderBase):
    def gen_curve(self) -> crv.Curve:
        return crv.Curve(lambda t: ORIGIN,active=False)

class PatternBuilderBase(CurveBuilderBase):
    def __init__(self,
                 pattern: CurveBuilderBase,
                 num_repeat: int,
                 repeater_method: Callable[[np.ndarray,int],np.ndarray],
                 starter: CurveBuilderBase = NullBuilder() ,
                 ender: CurveBuilderBase = NullBuilder() ,
                 ):
        self.pattern = pattern
        self.num_repeat = num_repeat
        self.repeater_method = repeater_method
        self.starter = starter
        self.ender = ender
    
    def gen_curve(self) -> crv.CurveChain:
        starter_crv = self.starter.gen_curve()
        ender_crv = self.ender.gen_curve()
        pattern_crv = self.pattern.gen_curve()
        crv_list = [starter_crv]
        for k in range(self.num_repeat):
            crv_list.append(crv.TransformedCurve(lambda p: self.repeater_method(p,k), pattern_crv))
        crv_list.append(ender_crv)

        return crv.CurveChain(*crv_list)
    

class TrapezoidPatternBuilder(CurveBuilderBase):
    def __init__(self,
                 left_flank_builder:CurveBuilderBase,
                 top_builder:CurveBuilderBase,
                 right_flank_builder:CurveBuilderBase,
                 bottom_builder:CurveBuilderBase,
                 ) -> None:
        self.left_flank_builder = left_flank_builder
        self.top_builder = top_builder
        self.right_flank_builder = right_flank_builder
        self.bottom_builder = bottom_builder
    
    def gen_curve(self) -> crv.Curve:
        # return crv.CurveChain(self.left_flank_builder.gen_curve(),
        #                       self.top_builder.gen_curve(),
        #                       self.right_flank_builder.gen_curve(),
        #                       self.bottom_builder.gen_curve())
        return crv.CurveChain(self.bottom_builder.gen_curve(),
                              self.right_flank_builder.gen_curve(),
                              self.top_builder.gen_curve(),
                              self.left_flank_builder.gen_curve()
                              )

class GearToothTrapezoidAdapter(CurveBuilderBase):
    def __init__(self, 
                 trapezoid_builder: TrapezoidPatternBuilder,
                 root_fillet=0.0,
                 tip_fillet=0.0,
                 tip_reduction=0.0,
                 height_func:Callable=np.linalg.norm
                 ):
        self.trapezoid_builder = trapezoid_builder
        # negative values make no sense, using abs
        self.root_fillet =   np.abs(root_fillet)
        self.tip_fillet =    np.abs(tip_fillet)
        self.tip_reduction = np.abs(tip_reduction)
        self.height_func = height_func

 
    def gen_curve(self) -> crv.CurveChain:

        ref_curves = self.trapezoid_builder.gen_curve()
        botcrv = ref_curves.curves[0]
        topcrv = ref_curves.curves[2]
        leftcrv = ref_curves.curves[3]
        rightcrv = ref_curves.curves[1]

        r_bot = self.height_func(botcrv(1))
        r_top = self.height_func(topcrv(1))
        r_tip_red = r_top+self.tip_reduction
        sols = []
        for guess in np.linspace(0.01,0.99,4):
            sol = crv.find_curve_intersect(leftcrv,rightcrv,[1-guess,guess])
            if r_bot < np.linalg.norm(rightcrv(sol.x[1])) < r_tip_red and sol.success:
                sols.append(sol.x)

        if len(sols) > 0:
            
            sol = sols[np.argmin([sol[0] for sol in sols])]

            r_sol = self.height_func(rightcrv(sol[1]))

            if self.tip_reduction>0:
                angle_0 = -angle_between_vectors(topcrv(0),RIGHT)
                angle_1 = angle_between_vectors(topcrv(1),RIGHT)
                topcrv2 = crv.ArcCurve.from_radius_center_angle(radius=r_sol-self.tip_reduction,
                                                               center=ORIGIN,
                                                               angle_start=angle_0,
                                                               angle_end=angle_1)

                rightcrv.set_end_on(sol[1])
                leftcrv.set_start_on(sol[0])
                sol2 = crv.find_curve_intersect(topcrv2,rightcrv,[0.5,1])
                rightcrv.set_end_on(sol2.x[1])
                topcrv2.set_start_on(sol2.x[0])
                sol3 = crv.find_curve_intersect(topcrv2,leftcrv,[0.5,0])
                leftcrv.set_start_on(sol3.x[1])
                topcrv2.set_end_on(sol3.x[0])
                topcrv = topcrv2
                ref_curves.curves[2] = topcrv
            else:
                topcrv.active = False
                leftcrv.set_start_on(sol[0])
                rightcrv.set_end_on(sol[1])
        
        else:
            sol2 = crv.find_curve_intersect(topcrv,rightcrv,[0.5,1])
            rightcrv.set_end_on(sol2.x[1])
            topcrv.set_start_on(sol2.x[0])
            sol3 = crv.find_curve_intersect(topcrv,leftcrv,[0.5,0])
            leftcrv.set_start_on(sol3.x[1])
            topcrv.set_end_on(sol3.x[0])

        sol4 = crv.find_curve_intersect(leftcrv,botcrv,[1,1])
        leftcrv.set_end_on(sol4.x[0])
        sol5 = crv.find_curve_intersect(botcrv,rightcrv,[0.5,0])
        rightcrv.set_start_on(sol5.x[1])
        botcrv.set_end_on(sol5.x[0])

        if self.tip_fillet>0:
            props = ref_curves.get_length_portions()
            ref_curves = ref_curves.fillet(self.tip_fillet,props[-2])
            props = ref_curves.get_length_portions()
            ref_curves = ref_curves.fillet(self.tip_fillet,props[-4])

        if self.root_fillet>0:
            botcrv2 = botcrv.copy()
            botcrv2.set_end_on(4)
            sol4 = crv.find_curve_intersect(leftcrv,botcrv2,[1,1])
            botcrv2.set_start_on(sol4.x[1])
            ref_curves.append(botcrv2)
            props = ref_curves.get_length_portions()
            ref_curves = ref_curves.fillet(self.root_fillet,props[-2])
            props = ref_curves.get_length_portions()
            ref_curves = ref_curves.fillet(self.root_fillet,props[1])
            ref_curves.pop(-1)

        angle_end = angle_between_vectors(ref_curves(1),RIGHT)
        angle_start = -angle_between_vectors(ref_curves(0),RIGHT)
        angle_target = angle_start+angle_end
        sol = crv.find_curve_line_intersect(ref_curves,ORIGIN,scp_Rotation.from_euler('z',angle_target).apply(RIGHT))
        ref_curves.set_start_on(sol.x[0])

        return ref_curves


