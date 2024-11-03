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

import gggears.gggears_convert as ggc
import gggears.gggears_core as gg
import numpy as np
import matplotlib.pyplot as plt

n_z = 18
gamma = 0.5*np.pi/2
axis = gg.OUT
m=2

param = gg.InvoluteGearParamManager(z_vals=[0,1,2],
                                    n_teeth=n_z,
                                    module=lambda z: m*(1-np.tan(gamma)*z/n_z*2),
                                    center=lambda z: m*z*axis,
                                    cone_angle=gamma*2,
                                    angle=lambda z: 0.25*(z-1)**2 *0+0.1,
                                    axis=axis,
                                    h_d=1.4,
                                    h_a=1.2,
                                    h_o=2,
                                    root_fillet=0.0,
                                    tip_fillet=0.0,
                                    tip_reduction=0.1,
                                    profile_reduction=0,
                                    profile_shift=0.0,
                                    enable_undercut=True,
                                    inside_teeth=False)
gear1 = ggc.GearToNurbs(params=param,
                        n_points_vert=3,
                        n_points_hz=4)



points = gear1.nurb_profile_stacks[0][0](np.linspace(0,1,301))
ax = plt.axes(projection='3d')
ax.plot(points[:,0],points[:,1],points[:,2])
ax.axis('equal')
plt.show()