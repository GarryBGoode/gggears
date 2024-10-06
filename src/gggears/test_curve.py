from curve import *
import matplotlib.pyplot as plt
# curve1 = Curve(arc_from_2_point_center,params={'p0': DOWN,'p1':RIGHT,'center':ORIGIN})
# curve2 = Curve(arc_from_2_point_center,params={'p0': RIGHT,'p1':RIGHT*2+DOWN,'center':RIGHT*2})

curve1 = ArcCurve.from_2_point_center(p0=DOWN,p1=RIGHT,center=ORIGIN)
curve2 = ArcCurve.from_2_point_center(p0=RIGHT,p1=RIGHT*2+DOWN,center=RIGHT+DOWN)

chain1 = CurveChain(curve1,curve2)

chain1(0.2)
chain1(0.8)

print(len(chain1))
p1 = chain1(np.linspace(0,1,101))

chain2 = chain1.fillet(radius=0.2,location=0.5)
# chain2.set_start_on(0.5,preserve_inactive_curves=True)
# chain2.set_end_on(0.8)

chain3 = MirroredCurve(chain2)
print(curve1)
# chain2[1].active=0
# chain2(0.8)
p2 = chain2(np.linspace(0,1,101))

plt.plot(p1[:,0],p1[:,1])
plt.plot(p2[:,0],p2[:,1])
plt.show()