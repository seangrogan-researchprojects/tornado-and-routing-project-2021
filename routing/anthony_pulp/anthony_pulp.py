import logging
import time
from collections import defaultdict

import pulp
from grogies_toolbox.auto_mkdir import auto_mkdir

from main_utilities import euclidean
from routing.routing_toolbox.distance_calc import DistMatrix
import matplotlib.pyplot as plt


def remove_far_stations(waypoints, fire_stations, endurance=8, K=None):
    endurance = endurance * 60 * 60 * 22 / 2
    x, y = zip(*waypoints)
    ne, nw, se, sw = (max(x), max(y)), (max(x), min(y)), (min(x), max(y)), (min(x), min(y))
    extreme_points = [ne, nw, se, sw]
    fire_stations = {station: pt for station, pt in fire_stations.items() if
                     any(euclidean(pt, ex_pt) <= endurance for ex_pt in extreme_points)}
    if K:
        n_stations = K * 5
        dists = {station: min(euclidean(pt, ex_pt) for ex_pt in extreme_points) for station, pt in
                 fire_stations.items()}
        while len(fire_stations) > n_stations:
            fire_stations.pop(max(dists, key=dists.get))
            dists.pop(max(dists, key=dists.get))
    return waypoints, fire_stations


def anthony_pulp(fire_stations,
                 waypoints,
                 n_depots,
                 scanning_r,
                 init_solution=None,
                 name=None,
                 *, warm_start=False):
    print("Running Exact With Pulp")
    logging.info("Running Exact With Pulp")
    init_solution, name = _setup(init_solution, name, fire_stations, waypoints, n_depots, scanning_r, warm_start)
    waypoints, fire_stations = remove_far_stations(waypoints, fire_stations, K=n_depots)
    W, S, V = _build_sets(waypoints, fire_stations)
    t, E, K, M = _build_parameters(n_depots, fire_stations, W, S)
    x, y, z, t_bar, t_bar_s = _build_variables(W, S, E)
    if bool(warm_start and init_solution):
        # t_bar_start_val = 0
        for station, tour in init_solution.items():
            # running_tally = 0
            x[station][tour[1]][station].setInitialValue(1)
            x[tour[-1]][station][station].setInitialValue(1)
            # y[station].setInitialValue(1)
            # z[station][station].setInitialValue(running_tally)
            # running_tally += t[station][tour[1]]
            for i, j in zip(tour[1:], tour[2:]):
                # running_tally += t[i][j]
                x[i][j][station].setInitialValue(1)
                # z[i][station].setInitialValue(running_tally)
        #     t_bar_start_val = max(running_tally, t_bar_start_val)
        #     t_bar_s[station].setInitialValue(running_tally)
        # t_bar.setInitialValue(t_bar_start_val)
    # log_file = f'D:/project-anthony/output/cplex/cplex_{name}.log'
    problem = pulp.LpProblem(name, pulp.LpMinimize)
    _ObjFn(problem, t_bar)
    _SetVisitedConstraints(problem, V, W, S, x, True)
    _SetFlowConstraints(problem, W, S, x)
    _SetSubTourConstraints(problem, M, t, W, S, z, x)
    _SetEnduranceConstraint(problem, E, t, W, S, x)
    _SetMaximumTourTime(problem, t, V, W, S, x, t_bar)
    _SetNDepots(problem, K, W, S, x)
    _ExportLP(problem, name)
    solver = get_best_available_solver()
    print("Solving...")
    logging.info("Solving...")
    options = [
        ('MIPGap', 0.1),
        ("MIPFocus", 1),
        ('method', 2),
        ('TimeLimit', 2000),
        ('Cuts', 2)
    ]
    problem.solve(solver=solver(msg=1, options=options, warmStart=True))
    print("Solved!")
    logging.info("Solved!")
    solution = _convert_vars_to_solution(x, y, z, fire_stations)
    return solution


def make_route_from_arcs(arcs, fire_station):
    arcs_as_dict = {i: j for i, j in arcs}
    pts = {i for i, j in arcs}.union({j for i, j in arcs})
    route = [fire_station]
    pts.discard(route[-1])
    while len(pts) > 0:
        route.append(arcs_as_dict[route[-1]])
        pts.discard(route[-1])
    return route


def _convert_vars_to_solution(x, y, z, fire_stations):
    solution = defaultdict(list)
    db = defaultdict(dict)
    arcs = defaultdict(list)
    for i in x:
        for j in x[i]:
            for s in x[i][j]:
                if x[i][j][s].varValue > 0.9:
                    db[s][i] = j
                    _i, _j = i, j
                    if isinstance(i, str):
                        _i = (fire_stations[i][0], fire_stations[i][1])
                    if isinstance(j, str):
                        _j = (fire_stations[j][0], fire_stations[j][1])
                    arcs[s].append((_i, _j))
    plt.show()
    for station in arcs.keys():
        solution[station] = make_route_from_arcs(arcs[station], fire_stations[station])
    return solution


def _ExportLP(problem, name):
    print('Exporting lp file')
    fp = f'D:/project-anthony/output/{name}.lp'
    auto_mkdir(fp)
    problem.writeLP(fp, max_length=255)


def _SetNDepots(problem, K, W, S, x):
    print("\tSetting number of dispatches Constraint")
    problem += pulp.lpSum(x[s][j][s] for s in S for j in W) <= K


def _SetMaximumTourTime(problem, t, V, W, S, x, t_bar):
    print("\tSetting Max Tour Time Constraint")
    for s in S:
        W_s = _W_s(W, s)
        W_s.add(s)
        problem += pulp.lpSum(
            t[i][j] * x[i][j][s] for i in W_s for j in W if j != i
        ) <= t_bar


def _SetEnduranceConstraint(problem, E, t, W, S, x):
    print("\tSetting Endurance Constraint")
    for s in S:
        W_s = _W_s(W, s)
        problem += pulp.lpSum(t[i][j] * x[i][j][s] for i in W_s for j in W_s if j != i) <= E


def _SetSubTourConstraints(problem, M, t, W, S, z, x, indicator=False):
    print("\tMaking Subtour Constraints")
    for s1 in S:
        for s2 in S:
            try:
                problem += z[s1][s2] == 0
            except KeyError:
                pass
    for s in S:
        W_s = _W_s(W, s)
        for i in W_s:
            for j in W:
                if i != j:
                    problem += z[i][s] + t[i][j] - z[j][s] <= M * (1 - x[i][j][s])
                    problem += z[i][s] + t[i][j] - z[j][s] >= -1 * M * (1 - x[i][j][s])


def _SetVisitedConstraints(problem, V, W, S, x, allow_multiple_visits=True):
    if allow_multiple_visits:
        print("\tSetting Visited Constraint -- Allow Multiple Visits")
        for j in W:
            problem += pulp.lpSum(
                x[i][j][s] for i in V for s in S if i != j and (i in W or i == s)) >= 1
        for i in W:
            problem += pulp.lpSum(
                x[i][j][s] for j in V for s in S if i != j and (j in W or i == s)) >= 1
    else:
        print("\tSetting Visited Constraint -- Disallow Multiple Visits")
        for j in W:
            problem += pulp.lpSum(
                x[i][j][s] for i in V for s in S if i != j and (i in W or i == s)) == 1
        for i in W:
            problem += pulp.lpSum(
                x[i][j][s] for j in V for s in S if i != j and (j in W or i == s)) == 1
        for s in S:
            problem += pulp.lpSum(x[s][j][s] for j in W) <= 1
            problem += pulp.lpSum(x[j][s][s] for j in W) <= 1


def _SetFlowConstraints(problem, W, S, x):
    print("\tSetting Flow Constraint")
    for s in S:
        W_s = _W_s(W, s)
        for j in W:
            problem += \
                pulp.lpSum(x[i][j][s] for i in W_s if i != j and (i in W or i == s)) - \
                pulp.lpSum(x[j][k][s] for k in W_s if j != k and (k in W or k == s)) == 0


def _ObjFn(problem, t_bar, *, t=None, x=None):
    print('\tObjFn')
    problem += t_bar
    logging.info("Completed Building ObjFn")


def _build_variables(W, S, E):
    t_bar = pulp.LpVariable("t_bar", 0, E)
    t_bar_s = {s: pulp.LpVariable(f"t_bar_{s}", 0, E) for s in S}
    x = defaultdict(lambda: defaultdict(dict))
    for s in S:
        W_s = _W_s(W, s)
        for i in W_s:
            for j in W_s:
                if i != j:
                    x[i][j][s] = pulp.LpVariable(_x_name(i, j, s), 0, 1, cat="Binary")
    # y = {s: pulp.LpVariable(name=_y_name(s), lowBound=0, upBound=E) for s in S}
    y=None
    z = defaultdict(dict)
    for s in S:
        W_s = _W_s(W, s)
        for i in W_s:
            z[i][s] = pulp.LpVariable(lowBound=0, upBound=E, name=_z_name(i, s))
        z[s][s] = pulp.LpVariable(lowBound=0, upBound=0, name=_z_name(s, s))
    return x, y, z, t_bar, t_bar_s


def get_best_available_solver():
    import pulp
    solvers = [
        pulp.GUROBI,
        pulp.GUROBI_CMD,
        pulp.CPLEX_DLL,
        pulp.CPLEX_CMD,
        pulp.CPLEX_PY,
        pulp.COIN_CMD,
        pulp.COINMP_DLL,
        pulp.PULP_CBC_CMD,
        pulp.GLPK_CMD,
        pulp.XPRESS,
        pulp.PYGLPK,
        pulp.YAPOSIB,
        pulp.PULP_CHOCO_CMD
    ]
    for solver in solvers:
        if solver().available():
            print(f"using {solver.name}")
            return solver
    raise Exception(f"No Solver Installed!")


def _build_sets(waypoints, fire_stations):
    print("\tBuilding Sets")
    W, S = set(waypoints), set(fire_stations.keys())
    V = W.union(S)
    logging.info(f"Completed Building Sets len(W):{len(W)} len(S):{len(S)} len(V):{len(V)}")
    return W, S, V


def _build_parameters(n_depots, fire_stations, W, S):
    print("\tBuilding Parameters")
    E = 8 * 60 * 60 * 22  # 8h * 60min/h * 60s/min * 22m/s # because I dropped it somewhere....
    K = n_depots
    M = 2 * E
    # d_m = DistMatrix()
    t = defaultdict(dict)
    for i in W:
        for j in W:
            if i != j:
                t[i][j] = euclidean(i, j)
        for s in S:
            s_pt = (fire_stations[s][0], fire_stations[s][1])
            t[s][i] = t[i][s] = euclidean(s_pt, i)
    logging.info(f"Completed Building Parameters E:{E} K:{K} M:{M} len(t):{len(t)}")
    return t, E, K, M


def _build_sets(waypoints, fire_stations):
    print("\tBuilding Sets")
    W, S = set(waypoints), set(fire_stations.keys())
    V = W.union(S)
    logging.info(f"Completed Building Sets len(W):{len(W)} len(S):{len(S)} len(V):{len(V)}")
    return W, S, V


def _setup(initial_solution, name, fire_stations, waypoints, n_depots, scanning_radius, warm_start=True):
    if initial_solution is None and warm_start:
        logging.debug("No Solution passed, running CFRS")
        # initial_solution = cfrs.cluster_first_route_second(fire_stations, waypoints, n_depots, scanning_radius)
    if name is None:
        name = "ProjectAnthony_{}".format(time.strftime("%Y%m%d_%H%M%S"))
    return initial_solution, name


def replace_list_of_chars(string, chars_to_replace, char_to_replace_with):
    for c in chars_to_replace:
        string = string.replace(c, char_to_replace_with)
    return string


def simplify_text_for_var_name(element):
    if isinstance(element, str):
        element = element.title()
        element = replace_list_of_chars(element, [' ', '-', '_'], '')
        while len(element) > (100 / 4):
            idx = len(element) // 2  # //2 in python 3
            element = element[:idx] + element[idx + 1:]
    if isinstance(element, tuple):
        element = tuple(round(e) for e in element)
        element = f"e{element[0]}_n{element[1]}"
    return element


def _x_name(i, j, s):
    n = f'x_({simplify_text_for_var_name(i)})_({simplify_text_for_var_name(j)})_({simplify_text_for_var_name(s)})'
    assert len(n) < 255
    return n


def _y_name(s):
    n = simplify_text_for_var_name(s)
    n = f'y_({n})'
    assert len(n) < 255
    return n


def _z_name(i, s):
    n = f'z_({simplify_text_for_var_name(i)})_({simplify_text_for_var_name(s)})'
    assert len(n) < 255
    return n


def _W_s(W, s):
    W_s = W.copy()
    W_s.add(s)
    return W_s
