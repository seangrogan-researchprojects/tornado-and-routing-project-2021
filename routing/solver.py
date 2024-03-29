import copy
import logging

from routing.anthony_cluster_first_exact_second.anthony_cluster_first_exact_second import cluster_first_route_exact
from routing.anthony_cluster_first_route_second.cluster_first_route_second import cluster_first_route_second
from routing.anthony_pulp.anthony_pulp import anthony_pulp
from routing.anthony_tabu_search.anthony_tabu_search import anthony_tabu


def anthony_solver(pars, args, fire_stations, waypoints, solver=None, *, warm_start=False, init_solution=None):
    print(f"Routing with {solver}")
    fire_stations = copy.deepcopy(fire_stations)
    solver = pars.get('routing', solver)
    n_depots, scanning_radius = pars.get('n_depots', 4), pars.get('scanning_radius', 300)
    if solver is None or solver.lower() == 'none':
        logging.info(f'\'None\' method passed, generating waypoints with {solver}')
    if solver.lower() in {'cfrs', 'cluster first', 'cluster first route second'}:
        return cluster_first_route_second(fire_stations, waypoints, n_depots, scanning_radius)
    elif solver.lower() in {'cfrexact', 'cluster first', 'cluster first route exact', 'cfes'}:
        return cluster_first_route_exact(fire_stations, waypoints, n_depots, scanning_radius)
    elif solver.lower() in {'tabu'}:
        return anthony_tabu(fire_stations, waypoints, n_depots, scanning_radius)
    elif solver.lower() in {'exact'}:
        return anthony_pulp(fire_stations, waypoints, n_depots, scanning_radius,
                            warm_start=warm_start, init_solution=init_solution)
