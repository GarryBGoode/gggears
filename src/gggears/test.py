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
import matplotlib.pyplot as plt
import numpy as np

n=40

param1 = SphericalInvoluteProfileParam(gamma=PI/2*0.6,
                                       pitch_angle=2*PI/n,
                                       h_a=1,h_d=1.5,
                                       profile_shift=0,
                                       enable_undercut=True,
                                       root_fillet=0.0)

gearprofile1 = GearProfileSpherical(**param1.__dict__)

gear1 = Gear2DSpherical(num_of_teeth=-n,module=1,profile_param=param1)

# points = scp_Rotation.from_euler('y',PI/2-gearprofile1.gamma).apply(gearprofile1.ref_rack_curve(np.linspace(0,1,100)))+gearprofile1.center
# points2 = gearprofile1.involute_curve(np.linspace(0,1,100))
# points3 = gearprofile1.undercut_connector_arc(np.linspace(0,1,100))
# points4 = gearprofile1.undercut_curve(np.linspace(0,1.0,100))
# points5 = gearprofile1.rd_curve(np.linspace(0,1,100))
# points6 = gearprofile1.ra_curve(np.linspace(0,1,100))
# points7 = gearprofile1.profile(np.linspace(0,1,100))


points = gear1.profile_closed(np.linspace(0,1,1000))

ax = plt.axes(projection='3d')

ax.plot(points[:,0],points[:,1],points[:,2])
# ax.plot(points2[:,0],points2[:,1],points2[:,2])
# ax.plot(points3[:,0],points3[:,1],points3[:,2])
# ax.plot(points4[:,0],points4[:,1],points4[:,2])
# ax.plot(points5[:,0],points5[:,1],points5[:,2])
# ax.plot(points6[:,0],points6[:,1],points6[:,2])
# ax.plot(points7[:,0],points7[:,1],points7[:,2])

ax.axis('equal')
plt.show()
