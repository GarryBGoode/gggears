from sympy import *

init_printing() 
psi,theta,phi,x,y,z,x2,y2,z2 = symbols('psi theta phi x y z x2 y2 z2')


Rz = Matrix([[cos(psi), -sin(psi), 0],
                [sin(psi), cos(psi), 0],
                [0, 0, 1]])

Ry = Matrix([[cos(theta), 0, sin(theta)],
                [0, 1, 0],
                [-sin(theta), 0, cos(theta)]])

Rx = Matrix([[1, 0, 0],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi), cos(phi)]])

z = Matrix([[0],[0],[1]])

p,p0,p1,p2 = symbols('p p0 p1 p2')
ps = symbols('ps')
ps = Matrix([[psi],[theta]])
p = Matrix([[p0],[p1],[p2]])

pprint(Rx*Ry*(-z))

# I dont get it why it wont solve this properly
pprint(solve(Eq(Rx*Ry*(-z),p),ps))

