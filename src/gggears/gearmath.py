import numpy as np
from gggears.gggears_base_classes import *
from gggears.function_generators import *
from scipy.optimize import root
from scipy.spatial.transform import Rotation as scp_Rotation


def cone_angle_from_teeth(
    num_teeth_1: int, num_teeth_2: int, axis_angle: float = np.pi / 2
):
    """
    Calculate the cone angles for two bevel gears based on their number of teeth
    and the prescribed angle between their axes.

    Parameters
    ----------
    num_teeth_1 : int
        Number of teeth on gear 1.
    num_teeth_2 : int
        Number of teeth on gear 2.
    axis_angle : float, optional
        Angle between the axes of the two gears (in radians), by default np.pi / 2

    Returns
    -------
    np.ndarray
        Array containing the cone angles (in radians) for gear 1 and gear 2.
    """
    cosfi = np.cos(axis_angle)
    a = float(num_teeth_1)
    b = float(num_teeth_2)
    e = np.sqrt((-(b**2) - 2 * a * b * cosfi - a**2 * cosfi) / (cosfi - 1))
    gamma2 = np.arctan(e / a)
    gamma1 = np.pi - axis_angle - gamma2
    return np.array((gamma1 * 2, gamma2 * 2))


def calc_involute_mesh_distance(
    r_base_1,
    r_base_2,
    angle_base_1,
    angle_base_2,
    pitch_angle_2,
    inside_ring=False,
    backlash=0.0,
):
    """
    Calculate the required axial distance for two involute gears to mesh,
    taking into account prescribed backlash.

    Parameters
    ----------
    r_base_1 : float
        Base radius of the involute curve of gear 1.
    r_base_2 : float
        Base radius of the involute curve of gear 2.
    angle_base_1 : float
        Angular coordinate of the base of the involute curve of gear 1 (in radians).
    angle_base_2 : float
        Angular coordinate of the base of the involute curve of gear 2 (in radians).
    pitch_angle_2 : float
        Pitch angle of gear 2 (in radians).
    inside_ring : bool, optional
        Any of the 2 gears is an inside-ring gear, by default False
    backlash : float, optional
        Prescribed backlash between the gears, by default 0.0
        Backlash is defined as the distance along the line of action
        between the inactive flanks of the two gears.
        Angular backlash can be derived as backlash / base radius.

    Returns
    -------
    Dist : float
        Required axial distance for the two gears to mesh with the specified backlash.
    """

    if inside_ring:
        d1 = r_base_1 * np.abs(angle_base_1)
        d2 = r_base_2 * np.abs(angle_base_2)
        bl_sign = np.sign(d2 - d1)
        sol = root(
            lambda a: np.tan(a)
            - a
            + (d2 - d1 - backlash / 2 * bl_sign) / (r_base_2 - r_base_1),
            0,
        )
        Dist = (r_base_2 - r_base_1) / np.cos(sol.x[0])
    else:

        # d1 = r_base_1 * (-angle_base_1)
        # d2 = r_base_2 * (pitch_angle_2 / 2 + angle_base_2)
        # sol = root(
        #     lambda a: np.tan(a) - a + (d2 - d1 - backlash / 2) / (r_base_1 + r_base_2),
        #     0.1,
        # )
        d1 = r_base_1 * ((angle_base_1))
        d2 = r_base_2 * (pitch_angle_2 / 2 - (angle_base_2))
        sol = root(
            lambda a: a - np.tan(a) + (d1 - d2 + backlash / 2) / (r_base_1 + r_base_2),
            0.0,
        )
        Dist = (r_base_1 + r_base_2) / np.cos(sol.x[0])

    return Dist


def backlash_from_ax_distance(
    Distance: float,
    r_base_1: float,
    r_base_2: float,
    angle_base_1: float,
    angle_base_2: float,
    pitch_angle_2: float,
    inside_ring=False,
):
    """
    Calculate the backlash between two involute gears based on their axial distance.
    Inverse of calc_involute_mesh_distance.

    Parameters
    ----------
    Distance : float
        Axial distance between the two gears.
    r_base_1 : float
        Base radius of the involute curve of gear 1.
    r_base_2 : float
        Base radius of the involute curve of gear 2.
    angle_base_1 : float
        Angular coordinate of the base of the involute curve of gear 1 (in radians).
    angle_base_2 : float
        Angular coordinate of the base of the involute curve of gear 2 (in radians).
    pitch_angle_2 : float
        Pitch angle of gear 2 (in radians).
    inside_ring : bool, optional
        Any of the 2 gears is an inside-ring gear, by default False

    Returns
    -------
    backlash : float
        Backlash between the two gears.
    """
    if inside_ring:
        d1 = r_base_1 * (-angle_base_1)
        d2 = r_base_2 * (-angle_base_2)
        a = np.arccos((r_base_2 - r_base_1) / Distance)
        #  np.tan(a) - a + (d2 - d1 - backlash) / (r_base_2 - r_base_1) = 0
        backlash = d2 - d1 - (-np.tan(a) + a) * (r_base_2 - r_base_1)
    else:
        d1 = r_base_1 * (-angle_base_1)
        d2 = r_base_2 * (pitch_angle_2 / 2 + angle_base_2)
        a = np.arccos((r_base_1 + r_base_2) / Distance)
        #  np.tan(a) - a + (d2 - d1 - backlash) / (r_base_1 + r_base_2) = 0
        # (d2 - d1 - backlash) / (r_base_1 + r_base_2) = -np.tan(a) + a
        # (d2 - d1 - backlash) = ( -np.tan(a) + a) * (r_base_1 + r_base_2)
        backlash = d2 - d1 - (-np.tan(a) + a) * (r_base_1 + r_base_2)
    return backlash


def calc_nominal_mesh_distance(
    pitch_radius_1,
    pitch_radius_2,
    profile_shift_1,
    profile_shift_2,
    module,
    inside_ring_1=False,
    inside_ring_2=False,
):
    """
    Calculate the nominal mesh distance for two gears,
    taking into account profile shifts and whether any gear is an inside-ring gear.

    Parameters
    ----------
    pitch_radius_1 : float
        Pitch radius of gear 1.
    pitch_radius_2 : float
        Pitch radius of gear 2.
    profile_shift_1 : float
        Profile shift of gear 1.
    profile_shift_2 : float
        Profile shift of gear 2.
    module : float
        Module of the gears.
    inside_ring_1 : bool, optional
        Whether gear 1 is an inside-ring gear, by default False
    inside_ring_2 : bool, optional
        Whether gear 2 is an inside-ring gear, by default False

    Returns
    -------
    Distance : float
    """
    # gear 1 being inside-ring means profile shift of the other
    # gear will reduce axial distance

    # gear 1 being inside-ring and having profile shift
    # still increases the axial distance
    if inside_ring_1:
        ps_mult_2 = -1
    else:
        ps_mult_2 = 1
    if inside_ring_2:
        ps_mult_1 = -1
    else:
        ps_mult_1 = 1

    Dist = (
        pitch_radius_1 * ps_mult_1
        + pitch_radius_2 * ps_mult_2
        + profile_shift_1 * ps_mult_1 * module
        + profile_shift_2 * ps_mult_2 * module
    )
    return Dist


def calc_mesh_angle(
    geartransform_1: GearTransform,
    geartransform_2: GearTransform,
    pitch_angle_1: float,
    pitch_angle_2: float,
    gear1_inside_ring=False,
    gear2_inside_ring=False,
):
    """Calculate the rotation angle for this gear to mesh with the other gear. A
    bias value can be used for positioning within backlash.

    Parameters
    ----------
    geartransform_1 : GearTransform
        Transform-data of gear 1.
    geartransform_2 : GearTransform
        Transform-data of gear 2.
    pitch_angle_1 : float
        Pitch angle of gear 1 (in radians).
    pitch_angle_2 : float
        Pitch angle of gear 2 (in radians).
    gear1_inside_ring : bool, optional
        Whether gear 1 is an inside-ring gear, by default False
    gear2_inside_ring : bool, optional
        Whether gear 2 is an inside-ring gear, by default False

    Returns
    -------
    angle : float
        Rotation angle for gear 1 to mesh with gear 2 (in radians).
    """

    center_diff_dir = geartransform_2.center - geartransform_1.center
    if np.linalg.norm(center_diff_dir) < 1e-12:
        # if gears are co-axial, use x axis of other gear as reference
        center_diff_dir = geartransform_2.x_axis
    else:
        center_diff_dir = normalize_vector(center_diff_dir)

    if gear1_inside_ring:
        contact_dir_gear1 = center_diff_dir
        contact_dir_other = center_diff_dir
        phase_offset = 0
        phase_sign = 1
    elif gear2_inside_ring:
        contact_dir_gear1 = -center_diff_dir
        contact_dir_other = -center_diff_dir
        phase_offset = 0
        phase_sign = 1
    else:
        contact_dir_gear1 = center_diff_dir
        contact_dir_other = -center_diff_dir
        phase_offset = 0.5
        phase_sign = -1

    # mult from right: vector in other's coordinate system
    angle_of_other = angle_of_vector_in_xy(
        contact_dir_other @ geartransform_2.orientation
    )
    target_angle_gear1 = angle_of_vector_in_xy(
        contact_dir_gear1 @ geartransform_1.orientation
    )

    phase_of_other = ((geartransform_2.angle - angle_of_other) / pitch_angle_2) % 1

    angle = (
        target_angle_gear1
        + ((phase_sign * phase_of_other + phase_offset) % 1) * pitch_angle_1
    )

    return angle


def calc_mesh_orientation(
    gear_1_cone_angle: float,
    gear_2_cone_angle: float,
    R: float,
    gear2: GearTransform,
    inside_ring_1=False,
    inside_ring_2=False,
    target_dir: np.ndarray = RIGHT,
    offset: float = 0,
):
    """
    Calculate the orientation matrix for gear1 to mesh with gear 2.
    Meant ot be used with bevel gears, returns exact orientation of gear2 for
    cylindrical gears.

    Parameters
    ----------
    gear_1_cone_angle : float
        The full cone angle of gear 1 in radians
    gear_2_cone_angle : float
        The full cone angle of gear 2 in radians
    R : float
        The spherical radius of bevel gears
    gear2 : GearTransform
        Transform object containing the position and orientation of gear 2
    inside_ring_1 : bool, optional
        Whether gear 1 is an internal ring gear, by default False
    inside_ring_2 : bool, optional
        Whether gear 2 is an internal ring gear, by default False
    target_dir : np.ndarray, optional
        Target direction vector for mesh alignment, by default RIGHT
    offset : float, optional
        Additional angular offset for the mesh in radians, by default 0

    Returns
    -------
    np.ndarray
        3x3 orientation matrix for gear 1 to properly mesh with gear 2

    Notes
    -----
    Use calc_bevel_gear_placement_vector() to get the center position for gear 1.
    """
    target_dir_norm = target_dir - np.dot(target_dir, gear2.z_axis) * gear2.z_axis
    gamma_1 = gear_1_cone_angle / 2
    gamma_2 = gear_2_cone_angle / 2
    if np.linalg.norm(target_dir_norm) < 1e-12:
        # target_dir is parallel to x axis
        target_dir_norm = gear2.x_axis
    else:
        target_dir_norm = normalize_vector(target_dir_norm)
    if gear_1_cone_angle == 0 and gear_2_cone_angle == 0:
        return gear2.orientation
    else:
        if inside_ring_2:
            angle_ref = gamma_2 - gamma_1 + offset / R
        elif inside_ring_1:
            angle_ref = gamma_2 - gamma_1 - offset / R
        else:
            angle_ref = gamma_1 + gamma_2 + offset / R
        rot_ax = normalize_vector(np.cross(target_dir_norm, gear2.z_axis))
        rot1 = scp_Rotation.from_rotvec(rot_ax * angle_ref)
        return rot1.as_matrix() @ gear2.orientation


def calc_bevel_gear_placement_vector(
    target_dir_norm: np.ndarray,
    cone_data_1: ConicData,
    cone_data_2: ConicData,
    inside_ring_1=False,
    inside_ring_2=False,
    offset: float = 0,
):
    """
    Calculate the center position for gear 1 to mesh with gear 2 when both gears are
    bevel gears.

    Parameters
    ----------
    target_dir_norm : np.ndarray
        Normalized target direction vector for mesh alignment.
    cone_data_1 : ConicData
        Conic data for gear 1.
    cone_data_2 : ConicData
        Conic data for gear 2.
    inside_ring_1 : bool, optional
        Whether gear 1 is an internal ring gear, by default False
    inside_ring_2 : bool, optional
        Whether gear 2 is an internal ring gear, by default False
    offset : float, optional
        Additional angular offset for the mesh in radians, by default 0

    Returns
    -------
    np.ndarray
        Center position for gear 1 to properly mesh with gear 2

    Notes
    -----
    Use calc_mesh_orientation() to get the orientation matrix for gear 1.
    """
    gamma_1 = cone_data_1.gamma
    gamma_2 = cone_data_2.gamma
    R = cone_data_1.R

    if inside_ring_2:
        angle_ref = gamma_2 - gamma_1 + offset / R
    elif inside_ring_1:
        angle_ref = gamma_2 - gamma_1 - offset / R
    else:
        angle_ref = gamma_1 + gamma_2 + offset / R
    rot_ax = normalize_vector(np.cross(target_dir_norm, cone_data_2.transform.z_axis))
    rot1 = scp_Rotation.from_rotvec(rot_ax * angle_ref)
    center_h_1 = R * np.cos(gamma_1)
    center_h_2 = R * np.cos(gamma_2)
    # center_sph = gear2_transform.center + gear2_transform.z_axis * center_h_2
    diff_vector = rot1.apply(-center_h_1 * cone_data_2.transform.z_axis)
    return cone_data_2.center + diff_vector


if __name__ == "__main__":
    print(cone_angle_from_teeth(20, 40) * 180 / np.pi)
