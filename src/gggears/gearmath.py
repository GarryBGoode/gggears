import numpy as np
from gggears import *
from scipy.optimize import root


def cone_angle_from_teeth(
    num_teeth_1: int, num_teeth_2: int, axis_angle: float = np.pi / 2
):
    cosfi = np.cos(axis_angle)
    a = float(num_teeth_1)
    b = float(num_teeth_2)
    e = np.sqrt((-(b**2) - 2 * a * b * cosfi - a**2 * cosfi) / (cosfi - 1))
    gamma2 = np.arctan(e / a)
    gamma1 = np.pi - axis_angle - gamma2
    return np.array((gamma1 * 2, gamma2 * 2))


def calc_involute_mesh_distance(
    gear1: "InvoluteGear", gear2: "InvoluteGear", backlash: float = 0.0
):
    """
    Calculate the shortest axial distance for 2 gears meshing without backlash.
    When one of the gears has profile shift, the simple axial distance calculated from
    pitch radius and profile shift results in backlash.
    :param gear2: Other gear for calculating minimum distance
    :return: Axial distance
    """
    if gear2.inside_teeth or gear1.inside_teeth:
        d1 = gear1.r_base * (
            -gear1.gearcore.shape_recipe(0).tooth_generator.get_base_angle()
        )
        d2 = gear2.r_base * (
            -gear2.gearcore.shape_recipe(0).tooth_generator.get_base_angle()
        )
        sol = root(
            lambda a: np.tan(a)
            - a
            + (d2 - d1 + backlash) / (gear2.r_base - gear1.r_base),
            0.01,
        )
        Dist = (gear2.r_base - gear1.r_base) / np.cos(sol.x[0])
    else:

        d1 = gear1.r_base * (
            -gear1.gearcore.shape_recipe(0).tooth_generator.get_base_angle()
        )
        d2 = gear2.r_base * (
            gear2.pitch_angle / 2
            + gear2.gearcore.shape_recipe(0).tooth_generator.get_base_angle()
        )
        sol = root(
            lambda a: np.tan(a)
            - a
            + (d2 - d1 - backlash) / (gear1.r_base + gear2.r_base),
            0.1,
        )
        Dist = (gear1.r_base + gear2.r_base) / np.cos(sol.x[0])

    return Dist


def calc_nominal_mesh_distance(
    gear1: "InvoluteGear", gear2: "InvoluteGear", backlash: float = 0.0
):
    # gear 1 being inside-ring means profile shift of the other
    # gear will reduce axial distance

    # gear 1 being inside-ring and having profile shift
    # still increases the axial distance
    if gear1.inside_teeth:
        ps_mult_2 = -1
    else:
        ps_mult_2 = 1
    if gear2.inside_teeth:
        ps_mult_1 = -1
    else:
        ps_mult_1 = 1

    Dist = (
        gear1.rp
        + gear2.rp
        + gear1.inputparam.profile_shift * ps_mult_1 * gear1.module
        + gear2.inputparam.profile_shift * ps_mult_2 * gear2.module
    )
    return Dist


if __name__ == "__main__":
    # print(cone_angle_from_teeth(20, 40) * 180 / np.pi)

    gear1 = InvoluteGear(number_of_teeth=12, profile_shift=1)
    gear2 = InvoluteGear(number_of_teeth=24, profile_shift=-1)

    guess = (
        gear1.rp
        + gear1.inputparam.profile_shift
        + gear2.rp
        + gear2.inputparam.profile_shift
    )

    print(
        f"guess: {guess}; " + f"calculated: {calc_involute_mesh_distance(gear1, gear2)}"
    )
