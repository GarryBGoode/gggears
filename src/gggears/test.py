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

p0 = normalize_vector(np.array([1,0,0]))
p1 = normalize_vector(np.array([0,1,0]))
p2 = normalize_vector(np.array([1,1,2]))

curve1 = crv.ArcCurve2.from_2_point_center(p0,p1)
curve2 = crv.ArcCurve2.from_2_point_center(p1,p2)
curve3 = crv.ArcCurve2.from_2_point_center(p2,p0)
# curve4 = copy.deepcopy(curve3)
# curve4.set_start_on(0.5)
# curve4.set_end_on(3)

curve4 = crv.CurveChain(curve1,curve2,curve3)
curve4 = curve4.fillet(radius=0.1,location=curve4.get_length_portions()[2])

points1 = curve1(np.linspace(0,1,31))
points2 = curve2(np.linspace(0,1,31))
points3 = curve3(np.linspace(0,1,31))
points4 = curve4(np.linspace(0,1,201))


ax = plt.axes(projection='3d')

ax.plot(points1[:,0],points1[:,1],points1[:,2])
ax.plot(points2[:,0],points2[:,1],points2[:,2])
ax.plot(points3[:,0],points3[:,1],points3[:,2])
ax.plot(points4[:,0],points4[:,1],points4[:,2])
ax.axis('equal')


plt.show()
