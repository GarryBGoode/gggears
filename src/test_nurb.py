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


import curve as crv
import numpy as np
from defs import *
from matplotlib import pyplot as plt


curve1 = crv.ArcCurve.from_radius_center_angle(radius=0.5,center=RIGHT*0.5,angle_start=-PI/2,angle_end=0)
curve2 = crv.ArcCurve.from_2_point_center(p0=RIGHT,p1=RIGHT*2+DOWN,center=RIGHT+DOWN)

chain1 = crv.CurveChain(curve1,curve2)
# chain1 = chain1.fillet(radius=0.25,location=chain1.get_length_portions()[1])

nurb1 = crv.convert_curve_nurbezier(chain1)

points1 = nurb1(np.linspace(0,1,101))
points2 = chain1(np.linspace(0,1,101))

points3 = nurb1.points
plt.plot(points1[:,0],points1[:,1])
plt.plot(points3[:,0],points3[:,1],marker='+')

print(f'nurb points: \n {points3}')
print(f'knots: \n {nurb1.knots}')

plt.show()