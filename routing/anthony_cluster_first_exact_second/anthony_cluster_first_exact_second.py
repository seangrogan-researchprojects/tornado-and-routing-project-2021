import logging

import numpy as np
import pulp as plp
from tqdm import tqdm

# from create_waypoints.waypoint_creators.steiner_zone_set_covering import Config
# from geospatial_toolbox.distance_calcualtor import euclidean_dist
from main_utilities import euclidean
from routing.anthony_cluster_first_route_second.cluster_first_route_second import _cluster_first
from waypoint_generation.steiner_inspired.steiner_zone_set_covering import Config


def cluster_first_route_exact(fire_stations, waypoints, n_depots, scanning_radius):
    """ Heuristic to cluster the waypoints first and generate a route second.
    :param fire_stations: dict of fire stations
    :param waypoints: set or np.array of waypoints
    :return: the solution in dict form.
    """
    print('Cluster First Route Exactly')
    logging.info('Cluster First Route Exactly')
    if not isinstance(waypoints, np.ndarray):
        waypoints = np.array(list(waypoints))
    clusters = _cluster_first(waypoints, n_depots)
    solution = _route_exactly(clusters, fire_stations)
    return solution


def _route_exactly(clusters, fire_stations):
    print("Routing exactly")
    tours = dict()
    for idx, cluster in clusters.items():
        print(f"Routing exactly cluster {idx}")
        cluster = [tuple(ele) for ele in cluster]
        tour = _pulp_tsp(cluster, fire_stations, tour_id=idx)
        station = tour.pop(0)
        tour.pop(-1)
        coordinate = [s.get("coordinate") for s in fire_stations].pop()
        tours[station] = [coordinate] + tour + [coordinate]
    return tours


def _pulp_tsp(waypoints, fire_stations, tour_id=None):
    def _build_dist_matrix(wpts, f_stations):
        matrix = dict()
        wpts = list(wpts)
        for idx, i in enumerate(wpts):
            for j in wpts[idx:]:
                matrix[(i, j)] = matrix[(j, i)] = euclidean(i, j)
            for fire_station in f_stations:
                j = fire_station.get("name")
                matrix[(i, j)] = 0
                location = fire_station.get("coordinate")
                matrix[(j, i)] = euclidean(i, location)
        return matrix

    waypoints = list(waypoints)
    matrix = _build_dist_matrix(waypoints, fire_stations)
    arcs = matrix.keys()
    station_keys = list({fire_station.get("name") for fire_station in fire_stations})
    all_keys = waypoints + station_keys
    X = {arc: plp.LpVariable(cat=plp.LpBinary, name=f"x_{arc}") for arc in arcs}
    D = {key: plp.LpVariable(cat=plp.LpContinuous, name=f"d_{key}") for key in all_keys}
    model = plp.LpProblem(name=f"TSP Model {tour_id}", sense=plp.LpMinimize)
    objective = plp.lpSum(
        X[arc] * matrix[arc] for arc in tqdm(arcs, desc='ObjFn', total=len(arcs), position=0, leave=True)
    )
    print("Flow out constraint")
    for i in waypoints:
        model.add(
            plp.LpConstraint(
                plp.lpSum(
                    X[(i, j)] for j in waypoints
                ), rhs=1,
                sense=plp.LpConstraintEQ, name=f'out_{i}'
            )
        )
    for i in station_keys:
        model.add(
            plp.LpConstraint(
                plp.lpSum(
                    X[(i, j)] for j in waypoints
                ), rhs=1,
                sense=plp.LpConstraintLE, name=f'station_out_{i}'
            )
        )
    print("Flow in constraint")
    for j in waypoints:
        model.add(
            plp.LpConstraint(
                plp.lpSum(
                    X[(i, j)] for i in waypoints
                ), rhs=1,
                sense=plp.LpConstraintEQ, name=f'in_{j}'
            )
        )
    for j in station_keys:
        model.add(
            plp.LpConstraint(
                plp.lpSum(
                    X[(i, j)] for i in waypoints
                ), rhs=1,
                sense=plp.LpConstraintLE, name=f'in_{j}'
            )
        )
    print("Choose One Station")
    model.add(
        plp.LpConstraint(
            plp.lpSum(
                X[(i, j)] for i in station_keys for j in waypoints
            ), rhs=1,
            sense=plp.LpConstraintGE, name=f'station'
        )
    )
    print("Choose one arc")
    for i, j in matrix:
        if i != j:
            model.add(
                plp.LpConstraint(
                    X[(i, j)] + X[(j, i)], rhs=1, sense=plp.LpConstraintLE,
                    name=f'identity_{i}_{j}'
                )
            )
    print("Logical constraint")
    M = len(matrix) + 1
    for i, j in matrix:
        model.add(
            plp.LpConstraint(
                D[i] + 1 - (1 - X[(i, j)]) * M - D[j], sense=plp.LpConstraintGE, rhs=0,
                name=f'order_GE_({i},_{j})'
            )
        )
        model.add(
            plp.LpConstraint(
                D[i] + 1 + (1 - X[(i, j)]) * M - D[j], sense=plp.LpConstraintLE, rhs=0,
                name=f'order_LE_({i},_{j})'
            )
        )
    model.setObjective(objective)
    # model.writeLP(f"{tour_id}_tour_lp.lp")
    config = Config(options=dict(timeLimit=1200, gap=0.1))
    try:
        print(f"Trying to solve with GUROBI_CMD")
        model.solve(solver=plp.GUROBI_CMD(msg=1, options=config.config_gurobi()))
    except:
        print(f"Could not solve with GUROBI_CMD")
        try:
            print(f"Trying to solve with CPLEX_CMD")
            model.solve(solver=plp.CPLEX_CMD(msg=1, options=config.config_cplex()))
        except:
            print(f"Could not solve with CPLEX_CMD")
            try:
                print(f"Trying to solve with default CBC")
                model.solve(solver=plp.PULP_CBC_CMD(msg=1, options=config.config_cbc()))
            except:
                print(f"Could not solve with CBC")
                assert False
    print("fin")
    route_elements = [k for k, v in X.items() if v.varValue > 0.9]
    fire_station = [i for i, j in route_elements if i in station_keys].pop(0)
    route_elements = {i: j for i, j in route_elements}
    tour = [fire_station, route_elements[fire_station]]
    i = route_elements[fire_station]
    j = route_elements[i]
    while j != fire_station:
        tour.append(j)
        i = j
        j = route_elements[i]
    return tour
