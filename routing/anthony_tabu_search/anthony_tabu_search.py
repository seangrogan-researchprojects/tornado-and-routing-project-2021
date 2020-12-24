import csv
import logging
import time
# import solver.cluster_first_route_second.cluster_first_route_second as cfrs
# from data_structure.distance_matrix import DistMatrix
# from solver.anthony_tabu_search.anthony_tabu_calculator import _calc_solution_value
# from solver.anthony_tabu_search.anthony_tabu_neighbor_generators import generate_waypoint_move_neighbor_list, \
#     two_opt_name
# from solver.anthony_tabu_search.anthony_tabu_subfunctions import _status_update, __solution_copy, __plot_soln_step, \
#     get_move, apply_move_waypoint, define_vicinity
# from solver.anthony_tabu_search.tabu_datastructure import TabuList
from collections import namedtuple

from routing.anthony_cluster_first_route_second.cluster_first_route_second import clean_up_intersections, \
    cluster_first_route_second
from routing.anthony_tabu_search.anthony_tabu_calculator import _calc_solution_value
from routing.anthony_tabu_search.anthony_tabu_neighbor_generators import generate_waypoint_move_neighbor_list, \
    two_opt_name
from routing.anthony_tabu_search.anthony_tabu_subfunctions import _status_update, __solution_copy, define_vicinity, \
    __plot_soln_step, get_move, apply_move_waypoint
from routing.anthony_tabu_search.tabu_datastructure import TabuList
from routing.routing_toolbox.distance_calc import DistMatrix


def anthony_tabu(fire_stations, waypoints, n_depots, scanning_r, init_solution=None, name=None, pars=None):
    logging.info("Working on Tabu Search")
    print("Working on Tabu Search")
    pars = dict() if pars is None else pars
    dist_matrix = DistMatrix()
    init_solution, name = __setup(init_solution, name, fire_stations, waypoints, n_depots, scanning_r)
    t_bar, t_s, t_bar_name = _calc_solution_value(init_solution, dist_matrix, all=True)
    _status_update(t_bar, t_s, t_bar_name)

    # Apply and store best solutions
    t_bar_best, t_s_best = t_bar, t_s
    best_solution = __solution_copy(init_solution)
    current_solution = __solution_copy(init_solution)

    # Vicinity calculations
    vicinity = define_vicinity(waypoints, dist_matrix, scanning_r)

    # Tabu Parameters:
    max_moves_without_improvement = int(len(waypoints) / 100)
    tabu_max_len = int(len(waypoints) / 10)
    glb_iter, loc_iter = 0, 0
    tabu_list = TabuList(tabu_max_len)
    __plot_soln_step(glb_iter, loc_iter, t_bar, current_solution)

    neighbors = generate_waypoint_move_neighbor_list(current_solution, t_s, t_bar_name, dist_matrix, max_tour_only=True,
                                                     mp=False, vicinity=vicinity)
    tabu_log = []
    TabuLog = namedtuple("TabuLog", ["time","kounter", "local_iter", "t_bar_best", "t_s"])
    print(neighbors)
    time_lim = pars.get('tabu_time_lim', (24 * 60 * 60))
    time_current = 0
    time_init = time.time()
    kounter = 0
    while loc_iter < max_moves_without_improvement and time_current < time_lim:
        glb_iter += 1
        loc_iter += 1
        move = get_move(neighbors, tabu_list)
        current_solution = apply_move(move, waypoints, current_solution)
        t_bar, t_s, t_bar_name = _calc_solution_value(current_solution, dist_matrix, all=True)
        _status_update(t_bar, t_s, t_bar_name)
        __plot_soln_step(glb_iter, loc_iter, t_bar, current_solution)
        tabu_list.add(move)
        if t_bar < t_bar_best:
            print("** \tNew Best!")
            loc_iter = 0
            t_bar_best, t_s_best = t_bar, t_s
            best_solution = __solution_copy(current_solution)
            m_i = 0
            while m_i < tabu_max_len + 1:
                m_i += 1
                neighbors = remove_invalid_moves(neighbors, move)
                if len(neighbors) < 1:
                    break
                print(f"Exploring Generated Neighborhood iter {m_i}")
                move = get_move(neighbors, tabu_list)
                tabu_list.add(move)
                if move["del_objfn"] < 0:
                    kounter += 1
                    current_solution = apply_move(move, waypoints, current_solution)
                    t_bar, t_s, t_bar_name = _calc_solution_value(current_solution, dist_matrix, all=True)
                    time_current = time.time() - time_init
                    tabu_log.append(TabuLog(time_current,kounter, loc_iter, t_bar_best, t_s_best))
                    _status_update(t_bar, t_s, t_bar_name)
                    __plot_soln_step(f'{glb_iter}_{m_i}', loc_iter, t_bar, current_solution)
                else:
                    kounter += 1
                    t_bar, t_s, t_bar_name = _calc_solution_value(current_solution, dist_matrix, all=True)
                    time_current = time.time() - time_init
                    tabu_log.append(TabuLog(time_current, kounter, loc_iter, t_bar_best, t_s_best))
                    _status_update(t_bar, t_s, t_bar_name)
                    break
            if t_bar < t_bar_best:
                best_solution = __solution_copy(current_solution)
                t_bar_best, t_s_best = t_bar, t_s
        __plot_soln_step(f'{glb_iter}cont', loc_iter, t_bar, current_solution)
        time_current = time.time() - time_init
        tabu_log.append(TabuLog(time_current,kounter, loc_iter, t_bar_best, t_s_best))
        if time_current > time_lim:
            break
        if loc_iter > int(max_moves_without_improvement * 0.80):
            neighbors = generate_waypoint_move_neighbor_list(current_solution, t_s, t_bar_name, dist_matrix,
                                                             max_tour_only=False,
                                                             mp=False, vicinity=vicinity, tabu_list=tabu_list)
        else:
            neighbors = generate_waypoint_move_neighbor_list(current_solution, t_s, t_bar_name, dist_matrix,
                                                             max_tour_only=True,
                                                             mp=False, vicinity=vicinity, tabu_list=tabu_list)
        time_current = time.time() - time_init
        print(f"Current Time {time_current} of {time_lim} seconds")

    __plot_soln_step('Fin', 'Fin', t_bar, best_solution)
    write_tabu_log(tabu_log, name)
    return best_solution


def write_tabu_log(tabu_log, name):
    outdata = [entry._asdict() for entry in tabu_log]
    fields = list(outdata[0].keys())
    with open(f"./output/tabu_log_{name}.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outdata)


def apply_move(move, waypoints, current_solution):
    if move['WP'] in waypoints:
        current_solution = apply_move_waypoint(move, current_solution)
    elif move['WP'] in {two_opt_name(name) for name in current_solution}:
        current_solution = apply_move_two_opt(move, current_solution)
    return current_solution


def remove_invalid_moves(neighbors, last_move):
    kys = {last_move["WP"], last_move["i"], last_move["j"], last_move["p"], last_move["q"]}
    neighbors = neighbors[~neighbors['WP'].isin(kys)]
    neighbors = neighbors[~neighbors['i'].isin({last_move["WP"], last_move["i"], last_move["p"]})]
    neighbors = neighbors[~neighbors['j'].isin({last_move["WP"], last_move["j"], last_move["q"]})]
    neighbors = neighbors[~neighbors['p'].isin({last_move["WP"], last_move["i"], last_move["p"]})]
    neighbors = neighbors[~neighbors['q'].isin({last_move["WP"], last_move["j"], last_move["q"]})]
    return neighbors


def apply_move_two_opt(move, current_solution):
    name = move["tour_A1"]
    current_solution[name] = clean_up_intersections(current_solution[name])
    return current_solution


def __setup(initial_solution, name, fire_stations, waypoints, n_depots, scanning_radius):
    if initial_solution is None:
        logging.debug("No Solution passed, running CFRS")
        print("No Solution passed, running CFRS")
        initial_solution = cluster_first_route_second(fire_stations, waypoints, n_depots, scanning_radius)
    if name is None:
        name = "ProjectAnthony_{}".format(time.strftime("%Y%m%d_%H%M%S"))
    return initial_solution, name
