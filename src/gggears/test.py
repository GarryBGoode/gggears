import numpy as np
from scipy.spatial.transform import Rotation as scp_Rotation
from function_generators import *
from curve import *
from defs import *
from gggears2 import *
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