import json
import logging
import time
from collections import defaultdict
from copy import deepcopy

from routing.anthony_cluster_first_route_second.cluster_first_route_second import cluster_first_route_second
from routing.anthony_tabu_search.anthony_tabu_calculator import _calc_solution_value
from routing.routing_toolbox.distance_calc import DistMatrix


def anthony_tabu(fire_stations, waypoints, n_depots, scanning_r, init_solution=None, name=None, pars=None):
    logging.info("Working on Tabu Search")
    print("Working on Tabu Search")
    if not init_solution:
        logging.debug("No Solution passed, running CFRS")
        print("No Solution passed, running CFRS")
        init_solution = cluster_first_route_second(fire_stations, waypoints, n_depots, scanning_r)
    if not name:
        name = f"ProjectAnthony_{time.strftime('%Y%m%d_%H%M%S')}"
    if not pars:
        pars = "D:/phd_projects/project_anthony/routing/anthony_tabu/anthony_tabu_search_pars.json"
    with open(pars) as json_file:
        pars = json.load(json_file)
    print("Completed Setup")
    logging.info("Completed Setup")
    best_solution, current_solution = deepcopy(init_solution), deepcopy(init_solution)
    t_bar_best_solution, t_bar_current_solution = _calc_solution_value(best_solution), _calc_solution_value(current_solution)
    vicinity = define_vicinity(waypoints, DistMatrix(), scanning_r)
    time_0 = time.time()
    while True:
        pass
        if not pars.get("time_lim", False) and abs(time_0-time.time()) > pars.get("time_lim", False):
            break

def define_vicinity(waypoints, dist_matrix, scanning_r):
    vicinity = defaultdict(set)
    for wp1 in waypoints:
        for wp2 in waypoints:
            if wp1 != wp2:
                if dist_matrix.get_dist(wp1, wp2) < scanning_r * 10:
                    vicinity[wp1].add(wp2)
    return vicinity


