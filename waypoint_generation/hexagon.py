import math
from matplotlib import pyplot as plt
from main_utilities import my_round


def hexagon_pattern(points, pars, flat_top=True):
    scanning_rad = pars.get('scanning_radius', 300)
    waypoints = _generate_pattern(
        scanning_rad=scanning_rad, flat_top=flat_top, points=points
    )
    # waypoints2 = _generate_pattern(
    #     scanning_rad=scanning_rad, flat_top=False, points=points
    # )
    # x, y = zip(*waypoints)
    # plt.scatter(x, y)
    # plt.show()
    # x, y = zip(*waypoints2)
    # plt.scatter(x, y)
    # plt.show()
    return waypoints


def make_point(i, j, R, r):
    i_new = my_round(i, 0, 2 * R)
    if is_even(i_new / (2 * R)):
        j_new = my_round(j, 0, 2 * r)
    else:
        j_new = round(my_round(j, 0, 2 * r) + r)
    return i_new, j_new


def _generate_pattern(scanning_rad, points, flat_top):
    R, r = scanning_rad, math.sqrt(3) * scanning_rad / 2
    waypoints = set()
    if flat_top:
        print('\tGenerating Flat Top Hex Pattern')
        for e, n in points:
            waypoints.add(make_point(e, n, R, r))
    else:
        print('\tGenerating Pointy Top Hex Pattern')
        for e, n in points:
            waypoints.add(make_point(n, e, R, r)[::-1])  # need to reverse the tuple
    # x, y = zip(*waypoints)
    # plt.scatter(x, y)
    # plt.show()
    return waypoints


def is_even(num):
    if (num % 2) == 0:
        return True
    else:
        return False
