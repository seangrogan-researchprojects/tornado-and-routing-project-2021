import concurrent.futures
import logging
from collections import defaultdict

import pandas as pd
from math import sqrt
from scipy.spatial import distance_matrix
from tqdm import tqdm

from main_utilities import euclidean
from smallest_bounding_circle import find_smallest_bounding_circle
from waypoint_generation.steiner_inspired.steiner_zone_set_covering import find_minimum_covering


def steiner_zone(points, pars, args=None):
    print(f"WELCOME TO THE STEINER ZONE")
    logging.info(f"WELCOME TO THE STEINER ZONE")
    scanning_d = pars.get("scanning_radius", 300) * 2
    logging.debug(f"reducing problem size")
    logging.debug(f"Initial Points {len(points)}")
    points = reduce_problem_size(points)
    logging.debug(f"Initial Points {len(points)}")
    dist_matrix = create_dist_matrix(points)
    if args is not None and (args.mp is True or pars.get('mp', False) is True):
        groups = create_initial_groups(points, dist_matrix, scanning_d)
    else:
        groups = create_initial_groups_ProcessPoolExecutor(points, scanning_d, dist_matrix)
    print(len(groups))
    waypoints = find_minimum_covering(groups, points)
    return waypoints


def create_initial_groups_ProcessPoolExecutor(points, scanning_d, dist_matrix=None):
    def build_new_dm(_pt1, _points, _scanning_d):
        x, y = _pt1
        dm = {
            (x2, y2): euclidean(_pt1, (x2, y2))
            for (x2, y2) in _points
            if abs(x - x2) <= _scanning_d and abs(y - y2) <= _scanning_d
               and euclidean(_pt1, (x2, y2)) <= _scanning_d
        }
        return pd.Series(dm)

    groups = set()
    points = list(points)
    points.sort(key=lambda element: element[0])
    points.sort(key=lambda element: element[1])
    print(f"Creating initial groups with concurrent.futures")
    args = [
        dict(pt1=pt, neighborhood_points=build_new_dm(pt, points, scanning_d), scanning_d=scanning_d,
             chunk_num=idx)
        for idx, pt in enumerate(tqdm(points, desc="Arging", position=0, leave=True))
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(create_temp_group_threading_wrapper, arg)
                   for arg in tqdm(args, desc='executing', position=0, leave=True)]
        p_bar = tqdm(desc="Creating Groups", position=0, leave=True, total=len(points))
        for result in concurrent.futures.as_completed(results):
            groups.update(result.result())
            p_bar.update()
            p_bar.set_postfix_str(f"nGroups = {len(groups)}")
    # groups = filter_duplicates_tanja_help_me(groups)
    return groups


def create_temp_group_threading_wrapper(kwargs):
    return create_temp_group_threading(**kwargs)


def create_temp_group_threading(pt1, neighborhood_points, scanning_d, chunk_num=None):
    if isinstance(neighborhood_points, pd.DataFrame):
        neighborhood_points = get_close_pts(pt1, neighborhood_points, scanning_d)
    temp_groups = defaultdict(set)
    past_pts = {pt1}
    k = 0
    temp_groups[k].add(pt1)
    try:
        neighborhood_points.pop(pt1)
    except KeyError:
        pass
    for pt2, distance in neighborhood_points.iteritems():
        past_pts.add(pt2)
        add = False
        for idx, group in temp_groups.items():
            if can_add_to_group(pt2, group, scanning_d):
                temp_groups[idx].add(pt2)
                add = True
        if not add:
            k += 1
            temp_groups[k].add(pt2)
            for pt3 in past_pts:
                if can_add_to_group(pt2, temp_groups[k], scanning_d):
                    temp_groups[k].add(pt3)
    new_groups = set(frozenset(group) for group in temp_groups.values())
    return new_groups


def create_temp_group_mp(points, dist_matrix, scanning_d, chunk_num=None):
    print(f"    Working on chunk {chunk_num}")
    logging.debug(f"Working on chunk {chunk_num}")
    groups = set()
    for pt1 in points:
        temp_groups = defaultdict(set)
        past_pts = {pt1}
        k = 0
        temp_groups[k].add(pt1)
        neighborhood_points = get_close_pts(pt1, dist_matrix, scanning_d)
        try:
            neighborhood_points.pop(pt1)
        except KeyError:
            pass
        for pt2, distance in neighborhood_points.iteritems():
            past_pts.add(pt2)
            add = False
            for idx, group in temp_groups.items():
                if can_add_to_group(pt2, group, scanning_d):
                    temp_groups[idx].add(pt2)
                    add = True
            if not add:
                k += 1
                temp_groups[k].add(pt2)
                for pt3 in past_pts:
                    if can_add_to_group(pt2, temp_groups[k], scanning_d):
                        temp_groups[k].add(pt3)
        new_groups = set(frozenset(group) for group in temp_groups.values())
        groups.update(new_groups)
    return groups


def create_initial_groups(points, dist_matrix, scanning_d):
    groups = set()
    points = list(points)
    points.sort(key=lambda element: element[0])
    points.sort(key=lambda element: element[1])
    p_bar = tqdm(desc="Creating Groups", position=0, leave=True, total=len(points))
    for pt1 in points:
        temp_groups = defaultdict(set)
        past_pts = {pt1}
        k = 0
        temp_groups[k].add(pt1)
        neighborhood_points = get_close_pts(pt1, dist_matrix, scanning_d)
        try:
            neighborhood_points.pop(pt1)
        except KeyError:
            pass
        for pt2, distance in neighborhood_points.iteritems():
            past_pts.add(pt2)
            add = False
            for idx, group in temp_groups.items():
                if can_add_to_group(pt2, group, scanning_d):
                    temp_groups[idx].add(pt2)
                    add = True
            if not add:
                k += 1
                temp_groups[k].add(pt2)
                for pt3 in past_pts:
                    if can_add_to_group(pt2, temp_groups[k], scanning_d):
                        temp_groups[k].add(pt3)
        new_groups = set(frozenset(group) for group in temp_groups.values())
        groups.update(new_groups)
        p_bar.set_postfix_str(f"N Groups {len(groups)}")
        p_bar.update(1)
    groups = filter_duplicates_tanja_help_me(groups)
    return groups


def _intermediate_filter_duplicates_tanja_help_me(groups, new_groups):
    unique_sets = {ng for ng in new_groups if not any(ng < g for g in groups)}
    unique_sets.update(groups)
    return unique_sets


def filter_duplicates_tanja_help_me(groups):
    print("Finding Unique Sets")
    logging.info("Finding Unique Sets")
    unique_sets = {e for e in tqdm(groups, desc="Finding Unique Sets", position=0, leave=True)
                   if not any(e < s for s in groups)}
    return unique_sets


def can_add_to_group(point, group, scanning_d):
    if len(group) == 0:
        return True
    if any(euclidean(point, pt2) >= scanning_d for pt2 in group):
        # if any two points are further than the scanning d from each other, can never work
        return False
    if all(euclidean(point, pt2) <= (sqrt(2) * scanning_d) / 2 for pt2 in group):
        # if all points are within root2/2 * scanning d from each other, it will always fit
        return True
    # Update here to test if Welzl works???
    # return False
    # calculate minimum bounding circle
    radius, center = find_smallest_bounding_circle(set(group).union({point}))
    # return the result
    return radius * 2 < scanning_d


def get_close_pts(pt, dist_matrix, dist):
    return dist_matrix[pt][dist_matrix[pt] <= dist]


def create_dist_matrix(points):
    print(f"Creating Distance Matrix")
    df = pd.DataFrame(points, columns=['xcord', 'ycord'], index=points)
    matrix = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
    print(f"Created Distance Matrix")
    return matrix


def reduce_problem_size(points, factor=0):
    if factor <= 0:
        new_points = set(tuple(int(round(element, factor)) for element in point) for point in points)
    else:
        new_points = set(tuple(round(element, factor) for element in point) for point in points)
    return new_points
