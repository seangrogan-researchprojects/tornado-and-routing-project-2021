import json
import logging
from collections import namedtuple, defaultdict
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

# from file_readers.specific.read_fire_stations import fire_station_file_reader
# from geospatial_toolbox.convert_to_utm import convert_fire_stations_to_utm
# from geospatial_toolbox.distance_calcualtor import euclidean_dist
from main_utilities import euclidean
from plotting_toolbox.map_plotter_utm import map_plotter_utm
from routing.routing_toolbox.distance_calc import DistMatrix, calc_dist_of_tour


def cluster_first_route_second(fire_stations, waypoints, n_depots, scanning_radius, args=None):
    """ Heuristic to cluster the waypoints first and generate a route second.
    :param fire_stations: dict of fire stations
    :param waypoints: set or np.array of waypoints
    :return: the solution in dict form.
    """
    print('Cluster First Route Second')
    logging.info('Cluster First Route Second')
    if not isinstance(waypoints, np.ndarray):
        waypoints = np.array(list(waypoints))
    clusters = _cluster_first(waypoints, n_depots)
    # fs = {f.get("name"): f.get("coordinate") for f in fire_stations}
    solution, fs = _better_route_second(clusters, scanning_radius, fire_stations)
    solution = _improve_solution_loop(solution, fs)
    return solution


def _improve_solution_loop(solution, fire_stations):
    init_len, k = len(solution), 0
    while True:
        k += 1
        print(f"== WORKING ON ITER {k} ==")
        # map_plotter_utm(f"Solution len {len(solution)}", solution=solution)
        new_solution = _improve_solution(solution, fire_stations)
        if len(new_solution) >= len(solution):
            # map_plotter_utm(f"Solution len final {len(solution)}", solution=solution)
            return solution
        solution = {k: v.copy() for k, v in new_solution.items()}


def _improve_solution(solution, fire_stations):
    print("Attempting to improve solution")
    print(f"len solution {len(solution)}")
    dists = {station: get_tour_dist(tour) for station, tour in solution.items()}
    dists = list(dists.items())
    dists.sort(key=lambda x: x[1], reverse=False)
    current_t_bar = get_max_tour_dist(solution)
    new_solutions = []
    for idx, (tour1, dist1) in enumerate(dists[:-1], 1):
        print(f"== Working on tour {idx} ==")
        for tour2, dist2 in dists[idx:-1]:
            if dist1+dist2 > 1.1 * current_t_bar:
                continue
            waypoint_pool = set()
            waypoint_pool.update(set(solution[tour1][1:]))
            waypoint_pool.update(set(solution[tour2][1:]))
            new_fs = fire_stations.copy()
            new_fs.update({tour1: solution[tour1][0], tour2: solution[tour2][0]})
            new_route, _ = _better_route_second({'TEST': waypoint_pool}, None, new_fs)
            new_solution = {k: v.copy() for k, v in solution.items() if k not in {tour1, tour2}}
            new_solution.update(new_route)
            if current_t_bar < get_max_tour_dist(new_solution):
                pass
                # new_solution = {k: v for k, v in solution.items()}
            else:
                print(f"**New Best Possible Solution Found**")
                new_solutions.append(new_solution)
    if len(new_solutions) <= 0:
        logging.info("No New Solutions")
        return solution
    else:
        new_solutions.sort(key=lambda x: get_min_tour_dist(x), reverse=False)
        return new_solutions[0]


def _better_route_second(clusters, scanning_radius, fs):
    solution = defaultdict(list)
    print('Routing Clusters')
    # fs = {f.get("name"): f.get("coordinate") for f in fire_stations}
    dm = DistMatrix()
    for cluster_id, waypoints in clusters.items():
        print(f"Routing Cluster {cluster_id}")
        wp_set = {tuple(wp) for wp in waypoints}
        dm_dict = {tuple(i): {tuple(j): dm.get_dist(i, j) for j in wp_set if i != j} for i in wp_set}
        dm_df = pd.DataFrame().from_dict(dm_dict)
        i = dm_df.max(axis=0).idxmax()
        j = dm_df[i].idxmax()
        tour = [i, j]
        p_bar = tqdm(total=len(wp_set), desc=f"Routing Cluster {cluster_id}", position=0, leave=True)
        while len(tour) < len(wp_set):
            # print(f"\rTour Len {len(tour)} \t {len(wp_set)} \t {int(len(tour) * 100 / len(wp_set))} %", end="")
            dm_explore = dm_df.filter(items=tour, axis=0).filter(items=set(wp_set).difference(set(tour)), axis=1)
            i = dm_explore.min(axis=0).idxmin()
            j = dm_explore[i].idxmin()
            if i in tour:
                loc = find_min_insertion(wp=j, tour=tour)
                tour.insert(loc, tuple(j))
            elif j in tour:
                loc = find_min_insertion(wp=i, tour=tour)
                tour.insert(loc, tuple(i))
            else:
                input("Huston We have a problem")
            p_bar.update()
        print(f"\rTour Len {len(tour)} \t {len(wp_set)} \t {int(len(tour) * 100 / len(wp_set))} %")
        print("cleaning up intersections")
        tour = clean_up_intersections(tour)
        print("Finding best station")
        tour, station = find_best_firestation(dm, fs, tour, tour)
        tour.insert(0, fs[station])
        fs.pop(station)
        print("cleaning up intersections")
        tour = clean_up_intersections(tour)
        print("Applying to solution")
        solution[station] = tour
    return solution, fs




def get_tour_dist(tour):
    return sum(euclidean(i, j) for i, j in zip(tour, tour[1:]))


def get_max_tour_dist(solution):
    return max(get_tour_dist(tour) for tour in solution.values())


def get_min_tour_dist(solution):
    return min(get_tour_dist(tour) for tour in solution.values())


def find_min_insertion(tour, wp, dm=DistMatrix()):
    min_loc, min_d = 0, float("inf")
    t = [wp, tour[0]]
    d1 = calc_dist_of_tour(t, dm)
    if d1 < min_d:
        min_loc, min_d = 0, d1
    t = [tour[-1], wp]
    d1 = calc_dist_of_tour(t, dm)
    if d1 < min_d:
        min_loc, min_d = -1, d1
    idx = 0
    if len(tour) > 1:
        for i, j in zip(tour, tour[1:]):
            idx += 1
            d0 = calc_dist_of_tour([i, j], dm)
            d1 = calc_dist_of_tour([i, wp, j], dm)
            if (d1 - d0) < min_d:
                min_loc, min_d = idx, (d1 - d0)
    return min_loc


def _cluster_first(waypoints, n_depots):
    print('Cluster First')
    logging.info('k means')
    kmeans = KMeans(n_clusters=n_depots, init='k-means++', max_iter=300, random_state=12345, algorithm='auto')
    kmeans.fit(waypoints)
    clusters = dict()
    for point, cluster in zip(waypoints, kmeans.labels_):
        if cluster not in clusters:
            clusters[cluster] = list()
        clusters[cluster].append(point)
    for k, v in clusters.items():
        a = np.array(v)
        ind = np.lexsort((a[:, 1], a[:, 0]))
        clusters[k] = np.array(a[ind])
    return clusters


def find_nearest_insertions(tours):
    d = DistMatrix()
    Dists = namedtuple("Dists", ['tour', 'dist'])
    dists = []
    for s, tour in tours.items():
        dists.append(Dists(tour=tour, dist=calc_dist_of_tour(tour, d)))
    sorted(dists, key=attrgetter('dist'), reverse=True)
    for t, dist in dists:
        tour = tours[t].copy()
        for point in tour[1:]:
            min_idx, min_station, min_val = None, None, float('inf')
            tours[t].remove(point)
            for station, tour in tours.items():
                for i, _ in enumerate(tour[1:], start=1):
                    test = tour.copy()
                    test.insert(i, point)
                    distance = calc_dist_of_tour(test, d)
                    pass
    return tours


def clean_up_intersections(tour):
    done = False
    cleaned = tour.copy()
    while not done:
        cleaned, done = _clean_up_intersections(cleaned)
    return cleaned


def ccw(A, B, C):
    Ax, Ay = A[0], A[1]
    Bx, By = B[0], B[1]
    Cx, Cy = C[0], C[1]
    return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    # Return true if line segments AB and CD intersect


def intersect(line1, line2):
    A, B = line1[0], line1[1]
    C, D = line2[0], line2[1]
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def _clean_up_intersections(tour):
    clean = tour.copy()
    for p1, p2 in zip(tour, tour[1:]):
        for p3, p4 in zip(tour, tour[1:]):
            if (p1 == p3 and p2 == p4) or (p2 == p3) or (p1 == p4):
                pass
            else:
                line1, line2 = (p1, p2), (p3, p4)
                assert line1 != line2
                assert p2 != p3 and p1 != p4
                if intersect(line1, line2):
                    i, j = clean.index(p1), clean.index(p4)
                    clean = clean[:i + 1] + clean[j - 1:i:-1] + clean[j:]
                    return clean, False
    return clean, True


def test_if_line_intersects(line1, line2):
    p1i, p1j = line1[0], line1[1]
    p2i, p2j = line2[0], line2[1]
    le = max(min(p1i[0], p1j[0]), min(p2i[0], p2j[0]))
    ri = min(max(p1i[0], p1j[0]), max(p2i[0], p2j[0]))
    to = max(min(p1i[1], p1j[1]), min(p2i[1], p2j[1]))
    bo = min(max(p1i[1], p1j[1]), max(p2i[1], p2j[1]))
    if to > bo or le > ri:
        return False
    if (to, le) == (bo, ri):
        return False
    x_1, y_1 = zip(*line1)
    x_2, y_2 = zip(*line2)
    plt.plot(x_1, y_1)
    plt.plot(x_2, y_2)
    plt.show()
    plt.close()
    return True


def line_intersection(line1, line2):
    _x, _y = 0, 1
    d_x = (line1[0][_x] - line1[1][_x], line2[0][_x] - line2[1][_x])
    d_y = (line1[0][_y] - line1[1][_y], line2[0][_y] - line2[1][_y])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    x_1, y_1 = zip(*line1)
    x_2, y_2 = zip(*line2)
    plt.plot(x_1, y_1)
    plt.plot(x_2, y_2)
    plt.show()
    plt.close()

    div = det(d_x, d_y)
    if div == 0:
        return False
    d = (det(*line1), det(*line2))
    x = det(d, d_x) / div
    y = det(d, d_y) / div
    return x, y


def find_best_firestation(d, fs, tour_n, tour_s):
    t_n_d, t_s_d = 0, 0
    for i, j in zip(tour_n, tour_n[1:]):
        t_n_d += d.get_dist(i, j)
    for i, j in zip(tour_s, tour_s[1:]):
        t_s_d += d.get_dist(i, j)
    # find north forward
    north_forward, north_forward_station = _find_best_firestation(d, fs, tour_n)
    # find north reverse
    tour_n.reverse()
    north_reverse, north_reverse_station = _find_best_firestation(d, fs, tour_n)
    tour_n.reverse()
    # find south forward
    south_forward, south_forward_station = _find_best_firestation(d, fs, tour_s)
    # find south reverse
    tour_s.reverse()
    south_reverse, south_reverse_station = _find_best_firestation(d, fs, tour_s)
    tour_s.reverse()
    north_forward += t_n_d
    north_reverse += t_n_d
    south_forward += t_s_d
    south_reverse += t_s_d
    min_direction_station = {
        "north_forward": north_forward_station,
        "north_reverse": north_reverse_station,
        "south_forward": south_forward_station,
        "south_reverse": south_reverse_station
    }
    min_direction_value = {
        "north_forward": north_forward,
        "north_reverse": north_reverse,
        "south_forward": south_forward,
        "south_reverse": south_reverse
    }
    min_val = min(min_direction_value.values())
    direction = [k for k, v in min_direction_value.items() if v == min_val][0]
    if direction in {'north_forward'}:
        tour = tour_n
        return tour_n, min_direction_station[direction]
    elif direction in {'north_reverse'}:
        tour_n.reverse()
        tour = tour_n
        return tour_n, min_direction_station[direction]
    elif direction in {'south_forward'}:
        tour = tour_s
        return tour_s, min_direction_station[direction]
    elif direction in {'south_reverse'}:
        tour_s.reverse()
        tour = tour_s
        return tour_s, min_direction_station[direction]
    else:
        tour = tour_n
        return tour_n, min_direction_station[north_forward]


def _find_best_firestation(d, fs, tour):
    min_d, min_f = float('inf'), None
    for f in fs:
        dist = d.get_dist(fs[f], tour[0])
        if dist < min_d:
            min_d, min_f = dist, f
    return min_d, min_f


if __name__ == '__main__':
    pass
    # def read_solution_file(solution_file):
    #     with open(solution_file) as f:
    #         solution = json.load(f)
    #     waypoints = set(tuple(wp) for tour in solution.values() for wp in tour[1:])
    #     solution = {name: [tuple(wp) for wp in tour] for name, tour in solution.items()}
    #     return solution, waypoints
    #
    #
    # file = "D:/phd_projects/project_anthony/output/solution/" \
    #        "solution_2013 Moore-Newcastle OK Tornado No Welzl_steiner_zone_20191107_123501.json"
    #
    # pars = {"fire_stations": "D:/gis_database/usa/oklahoma_fire_stations/fire_stations/ok_firestations.csv"}
    # fire_stations = fire_station_file_reader(None, pars, as_dict=False)
    # fire_stations = convert_fire_stations_to_utm(fire_stations, common_number=14, common_letter="N")
    # solution, waypoints = read_solution_file(file)
    # fs = {f.get("name"): f.get("coordinate") for f in fire_stations}
    # [fs.pop(key) for key in solution.keys()]
    # _improve_solution(solution, fs)
