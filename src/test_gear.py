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
import gggears.curve as crv
import gggears.gggears_core as gg
import matplotlib.pyplot as plt
import numpy as np
from gggears.defs import *
import pytest as pytest
from scipy.spatial.transform import Rotation as scp_Rotation

def test_rotation():
    '''
    Test the rotation matrix of scipy.
    Not really a test but rather learning how to use the rotation matrix
      with np dimensions.
    '''
    rot = scp_Rotation.from_euler('y',np.pi/2)
    assert rot.as_matrix() == pytest.approx(np.array([[0,0,1],[0,1,0],[-1,0,0]]))
    assert rot.as_matrix() @ np.array([1,0,0]) == pytest.approx(np.array([0,0,-1]))
    # multiplying on the right with the transpose of the rotation matrix is the way to go
    assert np.array([RIGHT,UP,LEFT,IN]) @ rot.as_matrix().transpose() == \
        pytest.approx(np.array([IN,UP,OUT,LEFT]),rel=1e-12,abs=1e-12)

def test_gear(num_teeth=16,module=1,gamma=0.0,axis_pitch=0):
    m=module
    orient_mat = scp_Rotation.from_euler('y',axis_pitch).as_matrix()
    axis = orient_mat[:,2]
    param = gg.InvoluteGearParamManager(z_vals=[0,1],
                                        n_teeth=num_teeth,
                                        module=lambda t: m* (1-t*np.sin(gamma)/num_teeth*2),
                                        center=lambda z: m*z*axis*np.cos(gamma),
                                        orientation=orient_mat ,
                                        cone_angle=gamma*2,
                                        angle=0,
                                        h_d=1.0,
                                        h_a=1.4,
                                        h_o=2.5,
                                        root_fillet=0.3,
                                        tip_fillet=0.0,
                                        tip_reduction=0.0,
                                        profile_reduction=0,
                                        profile_shift=0.0,
                                        enable_undercut=False,
                                        inside_teeth=False)
    
    gear = gg.InvoluteGear(param)
    gear_gen = gear.curve_gen_at_z(0)
    outer_curve = gear_gen.generate_gear_pattern(gear_gen.profile)
    points = outer_curve(np.linspace(0,1,101*num_teeth))
    # points = gear_gen.profile(np.linspace(0,1,101))

    gear_gen2 = gear.curve_gen_at_z(1)
    outer_curve2 = gear_gen2.generate_gear_pattern(gear_gen.profile)
    points2 = outer_curve2(np.linspace(0,1,101*num_teeth))

    ax = plt.axes(projection='3d')
    ax.plot(points[:,0],points[:,1],points[:,2])
    ax.plot(points2[:,0],points2[:,1],points2[:,2])
    ax.axis('equal')
    plt.show()

if __name__ == '__main__':
    test_rotation()
    test_gear(gamma=PI/4,axis_pitch=PI/4)
    