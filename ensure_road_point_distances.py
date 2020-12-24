"""
There are a descent number of functions here that i'm group together
"""
import concurrent.futures

from main_utilities import euclidean


def ensure_road_point_distance(roads, scanning_radius, mp=False):
    if mp:
        return _ensure_road_point_distance_mp(roads, scanning_radius)
    else:
        return _ensure_road_point_distance_seq(roads, scanning_radius)


def _ensure_road_point_distance_seq(roads, scanning_radius):
    repaired_roads = []
    scanning_diameter = scanning_radius * 2
    for road in roads:
        repaired_road = __ensure_road_point_distance(road, scanning_diameter)
        repaired_roads.append(repaired_road)
    return repaired_roads


def _ensure_road_point_distance_mp(roads, scanning_radius):
    def _helper_ensure_road_point_distance(helper_args):
        return __ensure_road_point_distance(*helper_args)

    scanning_diameter = scanning_radius * 2
    arguments = [dict(road=road, scanning_diameter=scanning_diameter) for road in roads]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_helper_ensure_road_point_distance, arguments)
    return list(results)


def __ensure_road_point_distance(road, scanning_diameter):
    repaired_road = list()
    for p1, p2 in zip(road[:-1], road[1:]):
        repaired_road.append(p1)
        if euclidean(p1, p2) > scanning_diameter:
            distance = euclidean(p1, p2)
            div, rem = divmod(distance, scanning_diameter)
            frac, increase = 1.0 / (1.0 + div), 1.0 / (1.0 + div)
            d = euclidean(p1, p2)
            x1, y1 = p1
            x2, y2 = p2
            x_hat, y_hat = (x2 - x1) / d, (y2 - y1) / d
            while frac < 1:
                p3 = (d * frac * x_hat, d * frac * y_hat)
                repaired_road.append(p3)
                frac += increase
    repaired_road.append(road[-1])
    return repaired_road
