import logging
from collections import defaultdict

# from plotter.map_plotter import map_plotter
from plotting_toolbox.map_plotter_utm import map_plotter_utm


def _status_update(t_bar, t_s, t_bar_name):
    print(f"\tt_bar : {t_bar}")
    logging.info(f"t_bar : {t_bar}")
    logging.debug("Dists")
    for t, s in t_s.items():
        best = '*' if t in set(t_bar_name[0]) else ' '
        logging.debug(f"{best} {t} : {s}")
        print(f"\t{best} {t} : {s}")


def __solution_copy(solution):
    """Since the solution is stored as a dict of lists, doing dict.copy() don't work.
    :param solution:
    :return:
    """
    other_solution = {key: val.copy() for key, val in solution.items()}
    return other_solution


def __plot_soln_step(glb_iter, loc_iter, t_bar, current_solution):
    print("\t\tPlotting new solution")
    map_plotter_utm(title=f"Solution after iter {glb_iter} (t_b={t_bar}) (k={loc_iter})",
                         file=f"./output/tabu/i_{glb_iter}.png",
                         outprocess=True, tours=current_solution)


def get_move(neighbors, tabu_list, strict=False):
    neighbors.sort_values(by=['del_A2', 'del_A1', 'del_objfn'],
                          ascending=[True, True, True], inplace=True)
    for index, row in neighbors.iterrows():
        i, j, WP, p, q = row["i"], row["j"], row["WP"], row["p"], row["q"]
        if strict:
            # if any key in the tabu list
            mvs = (i, j, WP, p, q)
            if not tabu_list.keys_are_tabu(mvs):
                return row.to_dict()
        else:
            if not tabu_list.is_tabu(i):
                return row.to_dict()

    for index, row in neighbors.iterrows():
        return row.to_dict()


def apply_move_waypoint(move, solution):
    i, j, WP, p, q = move["i"], move["j"], move["WP"], move["p"], move["q"]
    t_1, t_2 = move["tour_A1"], move["tour_A2"]
    solution[t_1].remove(WP)
    if q is None:
        solution[t_2].append(WP)
    else:
        insert_here = solution[t_2].index(q)
        solution[t_2].insert(insert_here, WP)
    return __solution_copy(solution)


def pass_to_dict(i, j, p, q, WP, tour_A1, tour_A2, old_objfn, new_objfn,
                 old_tour_dist_A1, new_tour_dist_A1,
                 old_tour_dist_A2, new_tour_dist_A2):
    return {
        'i': i,
        'j': j,
        'p': p,
        'q': q,
        'WP': WP,
        'tour_A1': tour_A1,
        'tour_A2': tour_A2,
        'old_objfn': old_objfn,
        'new_objfn': new_objfn,
        'del_objfn': new_objfn - old_objfn,
        'old_tour_dist_A1': old_tour_dist_A1,
        'new_tour_dist_A1': new_tour_dist_A1,
        'del_A1': new_tour_dist_A1 - old_tour_dist_A1,
        'old_tour_dist_A2': old_tour_dist_A2,
        'new_tour_dist_A2': new_tour_dist_A2,
        'del_A2': new_tour_dist_A2 - old_tour_dist_A2
    }


def columns():
    return [
        'i', 'j', 'p', 'q', 'WP', 'tour_A1', 'tour_A2', 'old_objfn', 'new_objfn', 'del_objfn', 'old_tour_dist_A1',
        'new_tour_dist_A1', 'del_A1', 'old_tour_dist_A2', 'new_tour_dist_A2', 'del_A2'
    ]


def define_vicinity(waypoints, dist_matrix, scanning_r):
    vicinity = defaultdict(set)
    for wp1 in waypoints:
        for wp2 in waypoints:
            if wp1 != wp2:
                if dist_matrix.get_dist(wp1, wp2) < scanning_r * 7.5:
                    vicinity[wp1].add(wp2)
    return vicinity
