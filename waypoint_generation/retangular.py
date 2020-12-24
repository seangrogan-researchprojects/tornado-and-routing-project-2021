import math

from tqdm import tqdm

from main_utilities import my_round


def rectangular(points, scanning_radius):
    waypoints = set(round_point(pt, scanning_radius)
                    for pt in tqdm(points, desc="Rectangular"))
    return waypoints


def round_point(point, scanning_radius):
    sr = int(scanning_radius * math.sqrt(2))
    x, y = point
    _x, _y = my_round(x, precision=0, base=sr), my_round(y, precision=0, base=sr)
    new_point = (_x, _y)
    return new_point
