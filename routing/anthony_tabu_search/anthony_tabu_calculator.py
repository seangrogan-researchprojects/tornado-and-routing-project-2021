# from data_structure.distance_matrix import DistMatrix
from routing.routing_toolbox.distance_calc import DistMatrix


def _arc_calc_a1(name, t_s, i, j, WP, dist_matrix):
    a1_old = (i, WP, j)  # old arc
    a1_new = (i, j)  # removing node
    a1_old_dist = __calculate_distance_of_route(a1_old, dist_matrix)  # old arc dist
    a1_new_dist = __calculate_distance_of_route(a1_new, dist_matrix)  # new arc dist
    return (t_s[name] - a1_old_dist) + a1_new_dist


def _arc_calc_a2(name, t_s, p, q, WP, dist_matrix):
    a2_old = (p, q)  # removing node
    a2_new = (p, WP, q)  # old arc
    a2_old_dist = __calculate_distance_of_route(a2_old, dist_matrix)  # old arc dist
    a2_new_dist = __calculate_distance_of_route(a2_new, dist_matrix)  # new arc dist
    return (t_s[name] - a2_old_dist) + a2_new_dist


def __calculate_distance_of_route(route, dist_matrix=None):
    if dist_matrix is None:
        dist_matrix = DistMatrix()
    d = 0
    for i, j in zip(route, route[1:]):
        if j is not None:
            d += dist_matrix.get_dist(i, j)
    return d


def _calc_solution_value(solution, dist_matrix=None, t_bar=True, with_key=False, all=True):
    if dist_matrix is None:
        dist_matrix = DistMatrix()
    dists = dict()
    for name, route in solution.items():
        d = __calculate_distance_of_route(route, dist_matrix)
        dists[name] = d
    if all:
        v = max(dists.values())
        k = tuple(x for x, y in dists.items() if y == v)
        return (max(dists.values())), dists, (k, v)
    elif not t_bar and not with_key:
        return dists
    elif with_key:
        v = max(dists.values())
        k = tuple(x for x, y in dists.items() if y == v)
        return k, v
    return max(dists.values())