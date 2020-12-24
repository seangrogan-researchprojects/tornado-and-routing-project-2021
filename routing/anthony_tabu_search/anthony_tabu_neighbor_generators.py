from multiprocessing import Pool

import pandas as pd

# from anthony_tabu_search.anthony_subfunctions import columns, _arc_calc_a1, __solution_copy, _arc_calc_a2, pass_to_dict
# from data_structure.distance_matrix import DistMatrix
# from solver.tabu.anthony_tabu_calculator import _arc_calc_a1, _arc_calc_a2, __calculate_distance_of_route
# from solver.anthony_tabu_search.anthony_tabu_subfunctions import columns, __solution_copy, pass_to_dict
# from solver.cluster_first_route_second.cluster_first_route_second import clean_up_intersections
from routing.anthony_cluster_first_route_second.cluster_first_route_second import clean_up_intersections
from routing.anthony_tabu_search.anthony_tabu_calculator import _arc_calc_a1, _arc_calc_a2, \
    __calculate_distance_of_route
from routing.anthony_tabu_search.anthony_tabu_subfunctions import columns, __solution_copy, pass_to_dict
from routing.routing_toolbox.distance_calc import DistMatrix


def generate_waypoint_move_neighbor_list(solution, t_s, t_bar_name, dist_matrix=None, tabu_list=None,
                                         max_tour_only=False, mp=False, vicinity=None):
    print("Generating Neighbor List of Single Waypoint Moves")
    if max_tour_only:
        print("\tLooking at only the longest tour for candidate moves")
    moves = pd.DataFrame(columns=columns())
    if dist_matrix is None:
        dist_matrix = DistMatrix()
    _solution = {k: solution[k] for k in set(t_bar_name[0])} if max_tour_only else solution
    if mp:
        moves = _generate_waypoint_move_neighbor_list_multiprocess(_solution, t_s, solution, dist_matrix)
    else:
        moves = _generate_waypoint_move_neighbor_list_serial(_solution, t_s, solution, tabu_list, dist_matrix, vicinity)
    two_opts = two_opt(solution, tabu_list, t_s)
    moves = pd.concat([moves, two_opts])
    return moves


def _generate_waypoint_move_neighbor_list_multiprocess(_solution, t_s, solution, dist_matrix):
    print("\tWorking as multiprocessing...")
    moves = pd.DataFrame(columns=columns())
    arg_iter = []
    print("\tCreating arguments...")
    for name, tour in _solution.items():
        _tour = tour.copy()
        _tour.append(None)
        for idx, i, WP, j in zip(range(len(_tour)), _tour[0:], _tour[1:], _tour[2:]):
            print(f"\r\t\t{name} {idx}", end='')
            arg_iter.append((name, i, j, WP, t_s, _tour, solution, dist_matrix))
    with Pool() as pool:
        print("\n\t\tSending Workers")
        # results = pool.apply_async(func=_process_wp_arg_wrapper, args=arg_iter)
        results = list(pool.map(func=_process_wp_arg_wrapper, iterable=arg_iter))
    print("\t\tGetting Results")
    for result in results.get():
        moves = moves.append(result)
    print("\t\t\tDONE MP!")
    return moves


def _generate_waypoint_move_neighbor_list_serial(_solution, t_s, solution, tabu_list, dist_matrix, vicinity=None):
    moves = pd.DataFrame(columns=columns())
    for name, tour in _solution.items():
        print(f"\n\tWorking on tour {name}")
        _tour = tour.copy()
        _tour.append(None)
        for idx, i, WP, j in zip(range(len(_tour)), _tour[0:], _tour[1:], _tour[2:]):
            if tabu_list is None or not tabu_list.is_tabu(WP):
                print(f"\r\t\tWP Number {idx} of {len(_tour) - 2}", end="")
                moves = moves.append(process_wp(name, i, j, WP, t_s, _tour, solution, dist_matrix, vicinity))
    print()
    return moves


def _process_wp_arg_wrapper(args):
    cnt = args["_tour"].index(args["WP"])
    print(f"\t\tMultiprocessing, beginning {cnt}")
    return process_wp(*args)


def process_wp(name, i, j, WP, t_s, _tour, solution, dist_matrix, vicinity=None):
    moves = pd.DataFrame(columns=columns())
    a1_calculation = _arc_calc_a1(name, t_s, i, j, WP, dist_matrix)  # updating dist info
    solution_prime = __solution_copy(solution)  # copy to remove WP
    solution_prime[name].remove(WP)  # removing WP
    for name_prime, tour_prime in solution_prime.items():
        if vicinity is None or len(vicinity[WP].intersection(tour_prime)) > int(len(tour_prime) * 0.01):
            _tour_prime = tour_prime.copy()
            _tour_prime.append(None)
            for p, q in zip(_tour_prime[0:], _tour_prime[1:]):
                if vicinity is None or (p in vicinity[WP] or q in vicinity[WP]):
                    _t_s_new = t_s.copy()
                    _t_s_new[name] = a1_calculation
                    _t_s_new[name_prime] = _arc_calc_a2(name_prime, _t_s_new, p, q, WP, dist_matrix)  # updating dist
                    old_objfn, new_objfn = max(t_s.values()), max(_t_s_new.values())
                    move = pass_to_dict(i, j, p, q, WP,
                                        name, name_prime,
                                        old_objfn, new_objfn,
                                        t_s[name], _t_s_new[name],
                                        t_s[name_prime], _t_s_new[name_prime])
                    moves = moves.append(move, ignore_index=True)
    return moves


def two_opt(solution, tabu_list, t_s, dist_matrix=None):
    moves = pd.DataFrame(columns=columns())
    if dist_matrix is None:
        dist_matrix = DistMatrix()
    for name, tour in solution.items():
        n = two_opt_name(name)
        if tabu_list is None or not tabu_list.is_tabu(n):
            new_tour = clean_up_intersections(tour)
            _t_s_new = t_s.copy()
            _t_s_new[name] = __calculate_distance_of_route(new_tour, dist_matrix)
            old_objfn, new_objfn = max(t_s.values()), max(_t_s_new.values())
            move = pass_to_dict(n, n, n, n, n,
                                name, name,
                                old_objfn, new_objfn,
                                t_s[name], _t_s_new[name],
                                t_s[name], _t_s_new[name])
            moves = moves.append(move, ignore_index=True)
    return moves


def two_opt_name(name):
    return f"two_opt_{name}"
