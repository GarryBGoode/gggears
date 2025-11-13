# Copyright 2024 Gergely Bencsik
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from gggears.defs import *
from scipy.spatial.transform import Rotation as scp_Rotation
from scipy.special import comb


def rotate_vector(v, angle):
    rot1 = scp_Rotation.from_euler("z", angles=angle)
    return rot1.apply(v)


def normalize_vector(v):
    return v / np.linalg.norm(v)


def angle_between_vectors(v1, v2):
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


def angle_between_vector_and_plane(v, plane_normal):
    v_plane = v - np.dot(v, plane_normal) * plane_normal
    if np.linalg.norm(v_plane) == 0:
        return PI / 2
    else:
        return angle_between_vectors(v_plane, v)


def project_vector_to_plane(v, plane_normal):
    return v - np.dot(v, plane_normal) * plane_normal


def angle_of_vector_in_xy(v):
    return np.arctan2(v[1], v[0])


def involute_func(t, r, a=0, rad_offs=0, tan_offs=0, z_offs=0, csph=0):
    """
    Returns the x-y-z values of the involute function.
    t: input angle
    r: base circle radius
    a: offset angle, angle of the starting point of the involute on the base circle
    rad_offs, tan_offs: radial and tangential offset values to generate the trochoid of the undercut curve.
    csph: curvature of spherical involute for bevel gears. Zero curvature means straight (spur) gear.
    """

    def involute_val(val):

        rot_mat_z = np.asarray(
            [
                [np.cos(val + a), -np.sin(val + a), 0],
                [np.sin(val + a), np.cos(val + a), 0],
                [0, 0, 1],
            ]
        )
        rot_z = scp_Rotation.from_euler("z", angles=val + a)
        rot_mat_z = rot_z.as_matrix()

        # offset vector
        ov = np.asarray([[rad_offs], [tan_offs], [z_offs]])

        if csph == 0:

            z = 0
            beta = 0

            d = r * val * DOWN
            # rotation likes row vectors, I like column vectors, hence transpose()
            return rot_z.apply((r * RIGHT + d + ov).transpose()).transpose()
        else:

            R = 1 / csph
            gamma = r * val / R

            v1 = ov + R * RIGHT

            l = R * np.sin(gamma)
            z = R * (1 - np.cos(gamma))
            beta = np.arcsin(r / R)
            # d = l * np.array([[0], [-1], [0]]) + np.array([[0], [0], [z]])
            d = l * DOWN + z * OUT

            # z_center = np.array([[0],[0],[np.cos(beta)*R]])
            z_center = np.cos(beta) * R * OUT

            # rot_mat_y0 = np.array([[np.cos(PI/2-beta), 0, np.sin(PI/2-beta)],
            #                          [0, 1, 0],
            #                          [-np.sin(PI/2-beta), 0, np.cos(PI/2-beta)]])

            rot_mat_y0 = scp_Rotation.from_euler("y", beta - PI / 2).as_matrix()

            # rot_mat_z0 = np.array([[np.cos(gamma), np.sin(gamma), 0],
            #                          [-np.sin(gamma), np.cos(gamma), 0],
            #                          [0, 0, 1]])

            rot_mat_z0 = scp_Rotation.from_euler("z", -gamma).as_matrix()
            v2 = rot_mat_y0 @ rot_mat_z0 @ v1 + z_center

            # rot_mat_x = np.array([[1,0,0],
            #                         [0, np.cos(gamma), np.sin(gamma)],
            #                         [0, -np.sin(gamma), np.cos(gamma)]])
            # rot_mat_y = np.array([[np.cos(beta), 0, -np.sin(beta)],
            #                         [0, 1, 0],
            #                         [np.sin(beta), 0, np.cos(beta)]])

            # ov = rot_mat_y @ rot_mat_x @ ov
            # d = rot_mat_y @ d

            return np.reshape(rot_mat_z @ (v2), (1, 3))[0]

    if hasattr(t, "__iter__"):
        ret = np.empty((0, 3, 1))
        for u in t:
            point = involute_val(u)
            ret = np.concatenate([ret, point[np.newaxis, :, :]], 0)
        return ret
    else:
        return involute_val(t)


def involute_circle(t, r=1, angle=0, v_offs=ORIGIN, z_offs=0):
    """
    Returns the x-y-z values of the involute function.
    t: input angle
    r: base circle radius
    a: offset angle, angle of the starting point of the involute on the base circle
    v_offs: offset vector used for trochoid (undercut) curve generation. The offset rotates with the tangent line of the involute.
    z_offs: offset value applied in the z direction
    """

    rot_z = scp_Rotation.from_euler("z", angles=t + angle)
    d = r * t * DOWN
    return rot_z.apply((r * RIGHT + d + v_offs)) + z_offs * OUT


def involute_sphere(t, r=1, C=0.5, angle=0, v_offs=ORIGIN, z_offs=0):
    """
    Returns the x-y-z values of the involute function.
    The base circle is positioned in the x-y plane, the spherical center is placed in the positive Z direction for positive C values.
    t: input angle
    r: base circle radius
    C: spherical curvature. C=1/R, R is the sphere radius. Curvature is used for supporting C==0 or infinite R, reverting to circle invoulte.
    Negative values result in a cone with a center in negative Z direction.
    abs(C)<=1/r should be kept.
    a: offset angle, angle of the starting point of the involute on the base circle
    v_offs: offset vector used for trochoid (undercut) curve generation. The offset rotates with the tangent line of the involute.
    z_offs: offset value applied in the z direction
    """
    if C == 0:
        return involute_circle(t, r, angle, v_offs)
    else:
        rot_z = scp_Rotation.from_euler("z", angles=t + angle)

        R = 1 / np.abs(C)
        Csig = np.sign(C)
        phi = r / R * t
        # to keep similarity to circle involute for small conic angles (beta~0) this
        # rotation of v_offs is needed.
        v1 = scp_Rotation.from_euler("y", -PI / 2 * Csig).apply(v_offs) + R * RIGHT
        # v1 = v_offs + R*RIGHT

        beta = np.arcsin(r / R)
        # z_center = np.cos(beta) * R * OUT  * Csig
        z_center = np.sqrt(R**2 - r**2) * OUT * Csig
        rot_y0 = scp_Rotation.from_euler("y", (PI / 2 - beta) * Csig)
        rot_z0 = scp_Rotation.from_euler("z", -phi)

        rot_chain = rot_z * rot_y0 * rot_z0
        rot_chain2 = scp_Rotation.from_euler(
            "zyz", [-phi, (PI / 2 - beta) * Csig, t + angle]
        )
        v2 = rot_chain.apply(v1) + z_center + z_offs * OUT
        return v2


def cycloid_circle(t, rb=1, rc=1, angle=0, v_offs=ORIGIN, z_offs=0):
    """
    Returns the x-y-z values of the cycloid function.

    The cycloid is calculated for a circle of rc rolling on the outside of a circle of
    rb. If negative radius is supplied for rc, the circle rolls on the inside of the
    base circle.

    t: input angle
    rb: base circle radius
    rc: rolling circle radius
    a: offset angle, angle of the starting point of the involute on the base circle
    v_offs: offset vector used for trochoid curve generation. The offset rotates with the rolling circle.
    z_offs: offset value applied in the z direction
    """

    rot_z1 = scp_Rotation.from_euler("z", t + angle)
    beta = t * rb / rc
    rot_z2 = scp_Rotation.from_euler("z", beta)
    v0 = (rb + rc) * RIGHT
    v1 = rc * LEFT + v_offs

    return rot_z1.apply((v0) + rot_z2.apply(v1)) + z_offs * OUT


def cycloid_line(t, rc=1, angle=0, v_offs=ORIGIN, z_offs=0):
    pass


def cycloid_cone(t, rb=1, rc=1, C=0.5, angle=0, v_offs=ORIGIN, z_offs=0):
    """
    Returns the x-y-z values of the cycloid function.
    The base circle is positioned in the x-y plane, the spherical center is placed in the positive Z direction for positive C values.
    t: input angle
    rb: base circle radius
    rc: rolling circle radius
    C: spherical curvature. C=1/R, R is the sphere radius. Curvature is used for supporting C==0 or infinite R, reverting to circle cycloid.
    Negative values result in a cone with a center in negative Z direction.
    abs(C)<=1/rb should be kept.
    a: offset angle, angle of the starting point of the involute on the base circle
    v_offs: offset vector used for trochoid curve generation. The offset rotates with the tangent line of the involute.
    z_offs: offset value applied in the z direction
    """
    if C == 0:
        return cycloid_circle(t, r, angle, v_offs)
    else:
        rot_z = scp_Rotation.from_euler("z", t + angle)

        R = 1 / np.abs(C)
        Csig = np.sign(C)
        gamma1 = np.arcsin(rb / R)
        gamma2 = np.arcsin(rc / R)
        z_center = np.sqrt(R**2 - rb**2) * OUT * Csig

        gamma12_rot_y = scp_Rotation.from_euler("y", -Csig * (gamma1 + gamma2))
        v0 = rb * RIGHT + gamma12_rot_y.apply(RIGHT * rc)
        beta = t * rb / rc
        axis = gamma12_rot_y.apply(OUT)

        v1 = scp_Rotation.from_euler("z", beta).apply(LEFT * rc + v_offs)
        v3 = gamma12_rot_y.apply(v1 - z_center * 0) + z_center * 0

        v2 = rot_z.apply(v0 + v3) + z_offs * OUT

        return v2


def vectorize(func: callable):
    def vector_handler(t, **kwargs):
        if hasattr(t, "__iter__"):
            return np.stack([func(u, **kwargs) for u in t])
        else:
            return func(t, **kwargs)

    return vector_handler


def arc_from_2_point(t, p0=RIGHT, p1=UP, curvature=1.0, axis=OUT, revolutions=0.0):
    if curvature == 0:
        return p0 + t * (p1 - p0)
    else:

        # axis = normalize_vector( np.cross((p0-center),(p1-center)))
        if any((p1 - p0) != 0):
            dp = normalize_vector(p1 - p0)
            axis = normalize_vector(axis - np.dot(axis, dp) * dp)
        r = 1 / curvature
        if abs(r) < np.linalg.norm(p1 - p0) / 2:
            r = np.linalg.norm(p1 - p0) / 2 * np.sign(r)

        h = np.sqrt(r**2 - (np.linalg.norm(p1 - p0) / 2) ** 2) * np.sign(r)

        center = (p0 + p1) / 2 - np.cross(dp, axis) * h

        if any(np.isnan(axis)):
            axis = OUT
        d_angle = np.arctan2(
            np.linalg.norm(np.cross((p0 - center), (p1 - center))),
            np.dot((p1 - center), (p0 - center)),
        )
        d_angle += revolutions * PI * 2

        angles = t * axis * d_angle
        rot1 = scp_Rotation.from_rotvec(angles)
        return center + rot1.apply((p0 - center)).reshape(VSHAPE)


def arc_from_2_point_center(t, p0=RIGHT, p1=UP, center=ORIGIN, revolutions=0.0):

    axis = normalize_vector(np.cross(p0 - center, p1 - center))

    d_angle = np.arctan2(
        np.linalg.norm(np.cross((p0 - center), (p1 - center))),
        np.dot((p1 - center), (p0 - center)),
    )
    d_angle += revolutions * PI * 2

    angles = t * axis * d_angle
    rot1 = scp_Rotation.from_rotvec(angles)
    return center + rot1.apply((p0 - center)).reshape(VSHAPE)


def bezier(t, points):
    # n is the number of points, degree of polynomial is n-1
    n = np.shape(points)[0]
    d = n - 1
    output = np.zeros(points.shape[1:])
    for k in range(n):
        output = output + points[k] * comb(d, k) * (t**k) * ((1 - t) ** (d - k))
    return output


def bezier_coeff(t, n: int):
    return np.stack(
        [comb((n - 1), k) * (t**k) * ((1 - t) ** ((n - 1) - k)) for k in range(n)],
        axis=-1,
    )


def bezierdc(t, points: np.ndarray) -> np.ndarray:
    """
    Bezier curve evaluation using decasteljau algorithm. Works faster for arrays of t than iterating naive Bezier algorithm with bernstein polynomials.
    t: float or np 1darray
    points: 2d array of control points
    """

    if not isinstance(t, np.ndarray):
        t = np.array([t])
        single_t = True
    else:
        single_t = False

    # points axis 0: control points listing, axis 1+ : coordinates
    # t axis 0: t values
    # working axis 0: t values, axis 1: control points listing, axis 2+ : coordinates

    n = t.shape[0]
    d = points.shape[0]
    # Add as many singleton dimensions as needed
    t_shape = t.shape + (1,) * (points.ndim)
    t_expanded = t.reshape(t_shape)

    points2 = np.repeat(np.expand_dims(points, 0), n, axis=0)

    for k in range(d - 1):
        points2 = (1 - t_expanded) * points2[:, :-1] + (t_expanded) * points2[:, 1:]

    if single_t:
        return points2.squeeze()
    else:
        return points2.squeeze(1)


def lerp(t, a, b):
    t_shape = t.shape + (1,) * a.ndim  # Add as many singleton dimensions as needed
    t_expanded = t.reshape(t_shape)
    return (1 - t_expanded) * a + (t_expanded) * b


def interpolate(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def nurbezier(t, points, weights):
    n = points.shape[0]
    point2 = points * weights.reshape(n, 1)
    if hasattr(t, "__iter__"):
        out_points = bezierdc(t, point2)
        out_weights = bezierdc(t, weights)
        return out_points / out_weights.reshape(
            out_weights.shape + (1,) * (out_points.ndim - 1)
        )
    else:
        return bezierdc(t, point2) / bezierdc(t, weights)


def bezier_diff_t(t, points, n=1):
    d = points.shape[0]
    diffpoints = points
    if n > d:
        return np.zeros(t.shape + points.shape)
    for k in range(n):
        diffpoints = (d - k - 1) * (diffpoints[1:] - diffpoints[:-1])
    # diffpoints = points[1:] - points[:-1]
    diffpoints_bez = bezierdc(t, diffpoints)
    return diffpoints_bez


def nurbezier_diff_t(t, points, weights):
    """Derivative of a NURB segment by t"""
    n = points.shape[0]
    wpoints = points * weights.reshape(n, 1)
    diffpoints = (n - 1) * (wpoints[1:] - wpoints[:-1])
    diffweights = (n - 1) * (weights[1:] - weights[:-1])

    points_bez = bezierdc(t, wpoints)
    weights_bez = bezierdc(t, weights)

    if hasattr(t, "__iter__"):
        diffpoints_bez = bezierdc(t, diffpoints)
        diffweight_bez = bezierdc(t, diffweights)
        diffweight_bez = diffweight_bez.reshape(
            diffweight_bez.shape + (1,) * (diffpoints_bez.ndim - 1)
        )
        weights_bez = weights_bez.reshape(
            weights_bez.shape + (1,) * (points_bez.ndim - 1)
        )
        out_points = (
            weights_bez * diffpoints_bez - diffweight_bez * points_bez
        ) / weights_bez**2

    else:
        diffpoints_bez = bezierdc(t, diffpoints)
        diffweight_bez = bezierdc(t, diffweights)

        out_points = (
            diffpoints_bez * weights_bez - points_bez * diffweight_bez
        ) / weights_bez**2
    return out_points


def bezier_diff_p(t, n):
    """The coefficients of points for a bezier same as differentiation by points"""
    if not isinstance(t, np.ndarray):
        t = np.array([t])
        single_t = True
    else:
        single_t = False

    nt = t.shape[0]
    out_points = np.zeros((nt, n))
    for k in range(n):
        zero_one_arr = np.zeros(n)
        zero_one_arr[k] = 1
        out_points[:, k] = bezierdc(t, zero_one_arr)
    if single_t:
        return out_points.squeeze()
    else:
        return out_points


def nurbezier_diff_points(t, n, weights):
    """the derivative of a nurb segment by points"""
    if not isinstance(t, np.ndarray):
        t = np.array([t])
        single_t = True
    else:
        single_t = False

    outval = (
        bezier_diff_p(t, n)
        * weights[np.newaxis, :]
        / bezierdc(t, weights)[:, np.newaxis]
    )
    if single_t:
        return outval.squeeze()
    else:
        return outval


def nurbezier_diff_weights(t, points, weights):
    """Derivative of a NURB segment by weights"""
    if not isinstance(t, np.ndarray):
        t = np.array([t])
        single_t = True
    else:
        single_t = False

    # axis 0: t values, , axis 1 : weights/num degree, axis 2: control point dim

    n = points.shape[0]
    nurbez = nurbezier(t, points, weights)
    bezier_diffp = bezier_diff_p(t, n)
    bezier_weights = bezierdc(t, weights)
    diffpart_1 = (
        bezier_diffp[:, :, np.newaxis]
        * points[np.newaxis, :, :]
        / bezier_weights[:, np.newaxis, np.newaxis]
    )
    diffpart_2 = (
        bezier_diffp[:, :, np.newaxis] / bezier_weights[:, np.newaxis, np.newaxis]
    )

    out = diffpart_1 - diffpart_2 * nurbez[:, np.newaxis, :]
    if single_t:
        return out.squeeze()
    else:
        return out


def nurbezier_surface(u, v, points, weights):
    # u is for first coord (dimension), v is for second dimension
    n_0 = points.shape[0]
    n_1 = points.shape[1]
    q = []
    qw = []
    for ii in range(n_0):
        rowpoints = points[ii, :]
        rowweights = weights[ii, :]
        q.append(nurbezier(v, rowpoints, rowweights))
        qw.append(bezier(v, rowweights))

    qpoints = np.stack(q, axis=0)
    qweights = np.stack(qw, axis=0)
    p_out = nurbezier(u, qpoints, qweights)
    return p_out


def nurbezier_surface_2(u, v, points, weights):
    n, m = points.shape[:2]
    cn = bezier_coeff(u, n)
    cm = bezier_coeff(v, m)
    parts = []
    weights_bezier = []

    for ii, jj in np.ndindex((n, m)):
        weights_bezier.append(weights[ii, jj] * cn[ii] * cm[jj])
        parts.append(points[ii, jj] * weights[ii, jj] * cn[ii] * cm[jj])

    return np.sum(parts, axis=0) / np.sum(weights_bezier)


def calc_nurbezier_arc(p0, p2, center):
    pmid = (p0 + p2) / 2
    h_angle = angle_between_vectors(p0 - center, p2 - center) / 2
    # p0 and p2 should be on the same radius, but adding little numeric redundancy here with the average
    radius = (np.linalg.norm(p0 - center) + np.linalg.norm(p2 - center)) / 2
    p1 = (
        normalize_vector(pmid - center)
        * (np.sin(h_angle) ** 2 / np.cos(h_angle) + np.cos(h_angle))
        * radius
        + center
    )
    bz_points = np.array([p0, p1, p2])
    bz_weights = np.array([1, np.cos(h_angle), 1])
    return bz_points, bz_weights


def calc_quadratic_bezier_interp(p0, p1, p2):
    return np.array([p0, (4 * p1 - p0 - p2), p2])


def xyz_to_spherical(v, center=ORIGIN):
    """Convert to spherical coordinates. Spherical center can be set as kwarg. Zero angles are at the north pole and along the x axis.
    Returns: r: radius, phi: azimuth angle (rotation around z), theta: polar angle (altitude angle)
    """
    v = np.asarray(v)
    center = np.asarray(center)
    single_vector = v.ndim == 1
    if single_vector:
        v = v[np.newaxis, :]
    r = np.linalg.norm(v - center, axis=-1)
    theta = np.arccos((v - center)[:, 2] / r)
    phi = np.arctan2((v - center)[:, 1], (v - center)[:, 0])
    result = np.stack([r, phi, theta], axis=-1)
    return result[0] if single_vector else result


def spherical_to_xyz(s, center=ORIGIN):
    """
    Convert to cartesian coordinates. Spherical center can be set as kwarg. Zero angles are at the north pole and along the x axis.
    Input convention: s= [r, phi, theta] (radius, azimuth angle, polar angle)
    Returns: x, y, z coordinates
    """
    s = np.asarray(s)
    center = np.asarray(center)
    single_vector = s.ndim == 1
    if single_vector:
        s = s[np.newaxis, :]
    r, phi, theta = s[:, 0], s[:, 1], s[:, 2]
    x = r * np.sin(theta) * np.cos(phi) + center[0]
    y = r * np.sin(theta) * np.sin(phi) + center[1]
    z = r * np.cos(theta) + center[2]
    result = np.stack([x, y, z], axis=-1)
    return result[0] if single_vector else result


def xyz_to_cylindrical(v, center=ORIGIN):
    """
    Convert to cylindrical coordinates. Cylindrical center can be set as kwarg. Zero angles are at the x axis.
    Returns:
    r: radius,
    phi: azimuth angle (rotation around z),
    z: height"""
    v = np.asarray(v)
    center = np.asarray(center)
    single_vector = v.ndim == 1
    if single_vector:
        v = v[np.newaxis, :]
    r = np.linalg.norm((v - center)[:, :2], axis=-1)
    phi = np.arctan2((v - center)[:, 1], (v - center)[:, 0])
    z = (v - center)[:, 2]
    result = np.stack([r, phi, z], axis=-1)
    return result[0] if single_vector else result


def cylindrical_to_xyz(c, center=ORIGIN):
    """
    Convert to cartesian coordinates. Cylindrical center can be set as kwarg. Zero angles are at the x axis.
    Input convention: c= [r, phi, z]
    Returns: x, y, z coordinates
    """
    c = np.asarray(c)
    center = np.asarray(center)
    single_vector = c.ndim == 1
    if single_vector:
        c = c[np.newaxis, :]
    r, phi, z = c[:, 0], c[:, 1], c[:, 2]
    x = r * np.cos(phi) + center[0]
    y = r * np.sin(phi) + center[1]
    z = z + center[2]
    result = np.stack([x, y, z], axis=-1)
    return result[0] if single_vector else result


def octoid(
    t,
    base_rad=0.5,
    sphere_rad=1.0,
    alpha=20 * PI / 180,
    angle=0.0,
    v_offs=ORIGIN,
    z_offs=0.0,
):
    # pitch angle: beta
    # pressure angle: phi
    # r: fundamental sphere radius
    # alpha1: rolling parameter
    # alpha: flat tooth flank angle
    beta = np.arcsin(base_rad / sphere_rad)
    alpha_1 = t
    r = sphere_rad

    def s(x):
        return np.sin(x)

    def c(x):
        return np.cos(x)

    def s2(x):
        return np.sin(x) ** 2

    def c2(x):
        return np.cos(x) ** 2

    a1sb = alpha_1 * s(beta)

    R = -r / np.sqrt(s2(alpha) * s2(alpha_1 * s(beta)) + c2(alpha_1 * s(beta)))

    v0 = (
        -c2(alpha) * c(alpha_1) * s(a1sb) * c(a1sb)
        + s(beta) * s2(alpha) * s(alpha_1) * s2(a1sb)
        + c(beta) * s(alpha) * c(alpha) * s(alpha_1) * s(a1sb)
        + s(beta) * s(alpha_1) * c2(a1sb)
    )

    v1 = (
        +c2(alpha) * s(alpha_1) * s(a1sb) * c(a1sb)
        + s(beta) * s2(alpha) * c(alpha_1) * s2(a1sb)
        + c(beta) * s(alpha) * c(alpha) * c(alpha_1) * s(a1sb)
        + s(beta) * c(alpha_1) * c2(a1sb)
    )

    v2 = (
        -c(beta) * s2(alpha) * s2(a1sb)
        + s(beta) * s(alpha) * c(alpha) * s(a1sb)
        - c(beta) * c2(a1sb)
    )
    v = np.array([v0, v1, v2]) * R

    rot = scp_Rotation.from_euler("yz", angles=[PI, PI / 2 + angle])
    zshift = np.sqrt(sphere_rad**2 - base_rad**2)
    v = rot.apply(v)
    return v + zshift * OUT
