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

from gggearbuilder import *
import curve as crv
import matplotlib.pyplot as plt
import numpy as np
from defs import *

n=12
pitch_angle = 2*PI/n

class ref_arc_builder(CurveBuilderBase):
    def __init__(self,pitch_angle,radius=1):
        self.pitch_angle = pitch_angle
        self.radius = radius
    def gen_curve(self):
        return crv.ArcCurve.from_radius_center_angle(self.radius,ORIGIN,-self.pitch_angle,0)
    
class ref_tooth_line_builder(CurveBuilderBase):
    def __init__(self,pitch_angle,radius=1,height=0.2,alpha=20*np.pi/180, reverse=False):
        self.pitch_angle = pitch_angle
        self.radius = radius
        self.alpha = alpha
        self.height=height
        self.reverse = reverse
    def gen_curve(self):
        p0 = RIGHT * self.radius + rotate_vector(RIGHT,self.alpha)*self.height/2
        p1 = RIGHT * self.radius + rotate_vector(RIGHT,self.alpha)*(-self.height/2)
        p2 = rotate_vector(p0,-self.pitch_angle/4)
        p3 = rotate_vector(p1,-self.pitch_angle/4)
        if self.reverse:
            return crv.LineCurve(p3,p2)
        else:
            return crv.LineCurve(p2,p3)
        
test1 = ref_arc_builder(pitch_angle,1)
test2 = ref_arc_builder(pitch_angle,1.7)
test3 = ref_tooth_line_builder(pitch_angle,1.05,reverse=True)
test4 = ref_tooth_line_builder(-pitch_angle,1.05,alpha=-20*np.pi/180,reverse=False)

test5 = GearToothTrapezoidAdapter(TrapezoidPatternBuilder(test4,test2,test3,test1),
                                  root_fillet=0.05*0,
                                  tip_fillet=0.0,
                                  tip_reduction=0.1)

points1 = test1.gen_curve()(np.linspace(0,1,100))
points2 = test2.gen_curve()(np.linspace(0,1,100))
points3 = test3.gen_curve()(np.linspace(0,1,100))
points4 = test4.gen_curve()(np.linspace(0,1,100))
points5 = test5.gen_curve()(np.linspace(0,1,1000))


ax = plt.axes()
ax.axis('equal')
ax.plot(points1[:,0],points1[:,1])
ax.plot(points2[:,0],points2[:,1])
ax.plot(points3[:,0],points3[:,1])
ax.plot(points4[:,0],points4[:,1])
ax.plot(points5[:,0],points5[:,1])

plt.show()
