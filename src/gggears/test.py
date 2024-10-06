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

import numpy as np
from scipy.spatial.transform import Rotation as scp_Rotation
from function_generators import *
from curve import *
from defs import *
from gggears import *
import matplotlib.pyplot as plt



points =np.array([[ORIGIN,RIGHT,2*RIGHT],
                  [ORIGIN+UP,RIGHT+UP,RIGHT+UP],
                  [ORIGIN+2*UP,RIGHT+2*UP,2*RIGHT+2*UP]
                  ])
weights = np.ones((3,3))
weights[1,1] = 2
asd = nurbezier_surface(0.7,0.55,points,weights)
asd2 = nurbezier_surface_2(0.55,0.7,points,weights)

arccurve = Curve(arc_from_2_point_center,params={'p0':UP,'p1':RIGHT,'center':ORIGIN})
testpoints = arccurve(np.linspace(0,1,20))

testres,testpoints,testweights = fit_nurb_points(testpoints,n_points=3,force_2D=True)



curve_aprox = Curve(nurbezier,params={'points':testpoints,'weights':testweights})


asd=bezierdc(np.linspace(0,1,4),np.stack([np.zeros((3,3,3)),np.ones((3,3,3))]))

print(asd)
# print(asd2)