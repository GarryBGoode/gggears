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

import gggears as gg
import matplotlib.pyplot as plt
from defs import *
n = 8
gamma = PI/2 *0.5


param1 = gg.InvoluteParamMin(cone_angle=gamma*2,
                             pitch_angle=2*PI/n,
                             h_d=1.2,
                             enable_undercut=True)
profile1 = gg.InvoluteFlankGenerator(**param1.__dict__)


curvgen = gg.GearCurveGenerator(n_teeth=n,cone_angle=gamma*2,reference_tooth_curve=profile1.tooth_curve)
curvgen.generate_profile_closed()
profile_full = curvgen.generate_gear_pattern(curvgen.profile_closed)
print(curvgen.inverse_polar_transform(curvgen.polar_transform(np.array([RIGHT,UP]))))

points1 = curvgen.rp_circle(np.linspace(0,1,300))
points2 = curvgen.ra_circle(np.linspace(0,1,300))
points3 = curvgen.rd_circle(np.linspace(0,1,300))
points4 = curvgen.ro_circle(np.linspace(0,1,300))
points5 = profile_full(np.linspace(0,1,300*n))

ax = plt.axes(projection='3d')

ax.plot(points1[:,0],points1[:,1], points1[:,2], marker='',linestyle='-', color='red')
ax.plot(points2[:,0],points2[:,1], points2[:,2], marker='',linestyle='-', color='green')
ax.plot(points3[:,0],points3[:,1], points3[:,2], marker='',linestyle='-', color='blue')
ax.plot(points4[:,0],points4[:,1], points4[:,2], marker='',linestyle='-', color='black')
ax.plot(points5[:,0],points5[:,1], points5[:,2], marker='.',linestyle='-', color='orange')

ax.axis('equal')
plt.show()