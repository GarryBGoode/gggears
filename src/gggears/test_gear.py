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

n = 24
# gear1 = GearCylindric(num_of_teeth=16,z_vals=[0,1],n_tweens=1,params=GearParamHandler(angle=lambda z: z*PI/23))


asd1 = GearParamHandler(num_of_teeth=16,angle=lambda z: z*2,profile_param=InvoluteProfileHandler(profile_shift= lambda z: z/10))
print(asd1)
print(asd1(0.5))

# gear1 = GearProfile2D(pitch_angle=2*PI/n,enable_undercut=True,h_a=3)
# gear2 = GearProfile2D(pitch_angle=2*PI/n,enable_undercut=False,root_fillet=0.25,h_a=3)
gear3 = Gear2D(num_of_teeth=n, profile_param=InvoluteProfileParam(h_a=1,h_d=0.2,profile_shift=0,enable_undercut=False,root_fillet=0.5))
# gear4 = Gear2D(num_of_teeth=n, profile_param=InvoluteProfileParam(h_a=2,
#                                                                   root_fillet=0.5,
#                                                                   enable_undercut=False))

print(gear3.profile_reference.tooth_curve(0))
# gear5 = GearCylindric(params=GearParamHandler(center=lambda z: z*OUT))
point_arr =  gear3.profile_reference.profile(np.linspace(0,1,100*n))
# point_arr2 = gear4.boundary(np.linspace(0,1,100*n))
# # point_arr2 = gear1.gear_stack[6].boundary(np.linspace(0,1,100*n))


def_param = InvoluteProfileParam(h_a=1,
                                 h_d=1.2,
                                 profile_shift=0,
                                 enable_undercut=False,
                                 root_fillet=0)

val_range = np.linspace(0,0.5,4)
attr = 'root_fillet'

for val in val_range:
    mod_param = def_param
    setattr(mod_param,attr,val)
    gear3 = Gear2D(num_of_teeth=n, 
                   profile_symmetry_shift=-0.5,
                   profile_param=mod_param)
    point_arr =  gear3.profile_closed(np.linspace(0,1,101*n))
    plt.plot(point_arr[:,0],point_arr[:,1])

# plt.plot(point_arr[:,0],point_arr[:,1])
# plt.plot(point_arr2[:,0],point_arr2[:,1],'-')
plt.axis('equal')
plt.show()