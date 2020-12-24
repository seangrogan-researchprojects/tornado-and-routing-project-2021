import logging
import time
from collections import defaultdict

import math
from tqdm import tqdm

# from create_waypoints.waypoint_creators.steiner_zone import can_add_to_group
# from geospatial_toolbox.distance_calcualtor import euclidean_dist
# from geospatial_toolbox.smallest_bounding_circle import find_smallest_bounding_circle
from main_utilities import euclidean
from smallest_bounding_circle import find_smallest_bounding_circle
from waypoint_generation.steiner_inspired.steiner_zone import can_add_to_group


def random_steiner_zone(points, pars, improve_please):
    logging.info(f"WELCOME TO THE (RANDOM) STIENERZONE")
    waypoints, scanning_r = set(), pars.get("scanning_radius", 300)
    clusters = build_clusters(points, scanning_r)
    if improve_please:
        old_len = len(clusters)
        clusters = try_to_improve_clusters(clusters, scanning_r, iter_no_better=500)
        while old_len < len(clusters):
            old_len = len(clusters)
            clusters = try_to_improve_clusters(clusters, scanning_r, max_time=60, iter_no_better=200)
    for c, pts in tqdm(clusters.items(), desc="Finding centers of groups", position=0, leave=True):
        waypoints.add(center_of_cluster(pts))
    logging.info(f"Created {len(waypoints)} waypoints")
    return waypoints


def build_clusters(points, scanning_r):
    clusters = defaultdict(set)
    unvisited_pts = list(points.copy())
    unvisited_pts.sort(key=lambda element: element[0])
    unvisited_pts.sort(key=lambda element: element[1])
    p1 = unvisited_pts.pop()
    k = 0
    clusters[k].add(p1)
    p_bar = tqdm(total=len(points), desc="Creating Random STIENERZONE Groups", position=0)
    p_bar.update(1)
    while len(unvisited_pts) > 0:
        p1 = unvisited_pts.pop()
        added = False
        for c in range(k, -1, -1):
            pts = clusters[c]
            if can_add_to_group(p1, pts, scanning_r * 2):  # test_if_can_add(p1, pts, scanning_r):
                clusters[c].add(p1)
                added = True
                break
        if not added:
            k += 1
            clusters[k].add(p1)
        p_bar.update(1)
        p_bar.set_postfix_str(f"NGroups {k}")
    print("Finished Creating Initial Groups")
    logging.info("Finished Creating Initial Groups")
    logging.info(f"n clusters {len(clusters)}")
    return clusters


def try_to_improve_clusters(clusters, scanning_r, max_time=None, max_iter=10_000, iter_no_better=2000):
    def find_max_key(data):
        return max(data, key=lambda x: len(data[x]))

    def find_min_key(data):
        return min(data, key=lambda x: len(data[x]))

    def get_key(data, tabu):
        key = find_max_key(data)
        while key in tabu:
            test_data = {k: v for k, v in data.items() if key not in tabu}
            if len(test_data) == 0:
                tabu = []
                return key
            key = find_max_key(test_data)
        return key

    clusters = {k: v for k, v in clusters.items()}
    print("Heuristically improving clusters")
    logging.info("Heuristically improving clusters")
    init_c = len(clusters)
    max_time = float('inf') if max_time is None else max_time
    max_iter = float('inf') if max_iter is None else max_iter
    iter_no_better = float('inf') if iter_no_better is None else iter_no_better
    if max_iter is None and max_time is None and iter_no_better is None:
        max_iter=5000
    logging.debug(f"init_c={init_c}, scanning_r={scanning_r}, "
                  f"max_time={max_time}, max_iter={max_iter}, max_w/o_imprv={iter_no_better}")
    t_init, global_iter, local_iter, num_moves = time.time(), 0, 0, 0
    searched_clusters = []
    p_bar = tqdm(desc="Improving clusters", position=0)
    while (time.time() - t_init) < max_time and global_iter < max_iter and local_iter < iter_no_better:
        # attempt to improve
        searched_clusters = searched_clusters[:len(clusters) - 2]
        source_key = get_key(clusters, searched_clusters)
        source_cluster = clusters.pop(source_key, None)
        searched_clusters.append(source_key)
        if source_cluster is None:
            logging.warning("We had an error popping the key from the clusters")
            continue
        moved_points = set()
        if iter_no_better >= float("inf"):
            limiter = int(len(searched_clusters) * 0.5)
        else:
            limiter = int(len(searched_clusters) * (iter_no_better - local_iter) / iter_no_better)
        for point in source_cluster:
            for destination_key, destination_cluster in clusters.items():
                if destination_key not in searched_clusters[:limiter] \
                        and can_add_to_group(point, destination_cluster, scanning_r * 2):
                    clusters[destination_key].add(point)
                    moved_points.add(point)
                    num_moves += 1
                    break
        source_cluster.difference_update(moved_points)
        if len(source_cluster) > 0:
            clusters[source_key] = source_cluster
        else:
            local_iter = 0
        local_iter += 1
        global_iter += 1
        p_bar.update()
        p_bar.set_postfix_str(
            f": Time={time.time() - t_init:.2f}/{max_time} "
            f": localIter={local_iter}/{iter_no_better} "
            f": nMoves={num_moves} : nClusters={len(clusters)}")
    print(f'reduced problem from {init_c} to final {len(clusters)}')
    logging.info(f'reduced problem from {init_c} to final {len(clusters)}')
    return clusters


# def try_to_improve_clusters(clusters, scanning_r, max_time=360, max_iter=1000):
#     print("Heuristically improving clusters")
#     logging.info("Heuristically improving clusters")
#     t0, i = time.time(), 0
#     print()
#     searched_clusters = []
#     n_moves = 0
#     init_c = len(clusters)
#     logging.debug(f"init_c: {init_c} scanning_r: {scanning_r} max_time={max_time}, max_iter={max_iter}")
#     if max_iter is None and max_time is None:
#         max_iter = 100_000
#         p_bar = tqdm(total=max_iter, desc="Trying to Repair Groups", position=0)
#     elif max_time is None:
#         max_time = float('inf')
#         p_bar = tqdm(total=max_iter, desc="Trying to Repair Groups", position=0)
#     elif max_iter is None:
#         max_iter = float('inf')
#         p_bar = tqdm(desc="Trying to Repair Groups", position=0)
#     else:
#         p_bar = tqdm(total=max_iter, desc="Trying to Repair Groups", position=0)
#     while (time.time() - t0) < max_time and i < max_iter:
#         i += 1
#         sorted_info = list(clusters.items())
#         sorted_info.sort(key=lambda x: len(x[1]), reverse=True)
#         key1, group1 = sorted_info.pop(0)
#         searched_clusters = searched_clusters[:len(clusters) - 1]
#         skip = False
#         while key1 in searched_clusters:
#             try:
#                 key1, group1 = sorted_info.pop(0)
#             except:
#                 skip = True
#                 break
#         if skip:
#             searched_clusters = []
#         else:
#             searched_clusters.append(key1)
#             for point1 in list(group1).copy():
#                 __clusters = {_k: _v.copy() for _k, _v in clusters.items()}
#                 for key, group in __clusters.items():
#                     if len(group) > 0 and key != key1 and can_add_to_group(point1, group,
#                                                                            scanning_r * 2):  # test_if_can_add(point1, group, scanning_r):
#                         clusters[key].add(point1)
#                         clusters[key1].discard(point1)
#                         n_moves += 1
#                         break
#         p_bar.update(1)
#         p_bar.set_postfix_str(f"N Moves = {n_moves}")
#     keys = list(clusters.keys())
#     logging.debug(f"N Moves = {n_moves}")
#     for key in keys:
#         if len(clusters[key]) == 0:
#             clusters.pop(key, None)
#     print(f'reduced problem from {init_c} to final {len(clusters)}')
#     logging.info(f'reduced problem from {init_c} to final {len(clusters)}')
#     return clusters


def center_of_cluster(pts):
    """Wrapper for geometric_center for legacy"""
    # if len(pts) <= 3:
    #     return mean_center(pts)
    r, center = find_smallest_bounding_circle(pts)
    return center


def test_if_can_add(new_points, current_group, scanning_r):
    """(probably) a quicker way to check if the new points can be added to the group"""
    # Spoiler.... it probably isn't faster
    if isinstance(new_points, tuple) and len(new_points) == 2:
        return all(euclidean(new_points, p2) < scanning_r for p2 in current_group)
    if isinstance(current_group, tuple) and len(current_group) == 2:
        current_group = {current_group}
    return all(euclidean(p1, p2) < scanning_r * math.sqrt(2) for p1 in new_points for p2 in current_group)


def get_bounding_circle_reduced_scope(new_points, old_points):
    """Just tests new points against old points"""
    if isinstance(new_points, tuple) and len(new_points) == 2:
        new_points = {new_points}
    if isinstance(old_points, tuple) and len(old_points) == 2:
        old_points = {old_points}
    bounding_circle = max(euclidean(p1, p2) for p1 in new_points for p2 in old_points)
    return bounding_circle


def get_bounding_circle(*points):
    """gets the maxim distance between any two points"""
    collection = set()
    for arg in points:
        if isinstance(arg, tuple) and len(arg) == 2:
            collection.update({arg})
        else:
            collection.update(set(arg))
    bounding_circle = max(euclidean(p1, p2) for p1 in collection for p2 in collection)
    return bounding_circle
