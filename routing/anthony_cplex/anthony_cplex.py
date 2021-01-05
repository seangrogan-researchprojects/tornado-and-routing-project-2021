import logging
import time
from collections import defaultdict

import matplotlib.pyplot as plt
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from docplex.util.environment import get_environment

# from operator import attrgetter
import solver.cluster_first_route_second.cluster_first_route_second as cfrs
from data_structure.distance_matrix import DistMatrix


def anthony_cplex(fire_stations, waypoints, n_depots, scanning_r, init_solution=None, name=None, *, warm_start=True):
    print("Running CPLEX")
    logging.info("Running CPLEX")
    init_solution, name = _setup(init_solution, name, fire_stations, waypoints, n_depots, scanning_r)
    W, S, V = _build_sets(waypoints, fire_stations)
    t, E, K, M = _build_parameters(n_depots, fire_stations, W, S)
    log_file = f'D:/project-anthony/output/cplex/cplex_{name}.log'
    mdl = Model(name, log_output=log_file)
    x, y, z, t_bar, t_bar_s = _build_variables(mdl, W, S, E)
    # _ObjFn(mdl, t_bar,t=t,x=x)
    _ObjFn(mdl, t_bar)
    _SetVisitedConstraints(mdl, V, W, S, x)
    _SetFlowConstraints(mdl, W, S, x)
    _SetSubTourConstraints(mdl, M, t, W, S, z, x)
    _SetEnduranceConstraint(mdl, E, t, W, S, x)
    _SetMaximumTourTime(mdl, t, V, W, S, x, t_bar)
    _SetNDepots(mdl, K, W, S, x)
    _ExportLP(mdl, name)
    if warm_start:
        _ApplyWarmStart(mdl, init_solution, y, x)
    print('Solving')
    mdl.context.cplex_parameters.mip.tolerances.absmipgap = 0.99
    mdl.context.cplex_parameters.mip.tolerances.mipgap = 0.99
    mdl.context.cplex_parameters.emphasis.mip = 1  # CPX_MIPEMPHASIS_FEASIBILITY
    mdl.context.cplex_parameters.mip.strategy.file = 3
    mdl.context.cplex_parameters.mip.strategy.nodeselect = 2
    solution = mdl.solve(log_output=True)
    with get_environment().get_output_stream("D:/project-anthony/output/cplex/solution_{}.json".format(name)) as fp:
        mdl.solution.export(fp, "json")
    parsed_soln = _convert_vars_to_solution(x, y, z, fire_stations)
    return init_solution


def _ApplyWarmStart(mdl, initial_solution, y, x):
    if initial_solution is not None:
        print("Warm Starting")
        mdl_solution = SolveSolution(mdl)
        for cluster, route in initial_solution.items():
            mdl_solution.add_var_value(y[cluster], 1)
            for i, j in zip(route[1:], route[2:]):
                mdl_solution.add_var_value(x[i][j][cluster], 1)
            f, l = route[1], route[-1]
            mdl_solution.add_var_value(x[cluster][f][cluster], 1)
            mdl_solution.add_var_value(x[l][cluster][cluster], 1)
        print(mdl_solution.display_attributes())
        print(mdl_solution.check_as_mip_start())
        mdl.add_mip_start(mdl_solution)


def _ExportLP(mdl, name):
    print('Exporting lp file')
    mdl.export_as_lp(f'D:/project-anthony/output/cplex/{name}.lp')


def _convert_vars_to_solution(x, y, z, fire_stations):
    solution = defaultdict(list)
    db = defaultdict(dict)
    arcs = defaultdict(list)
    for i in x:
        for j in x[i]:
            for s in x[i][j]:
                if x[i][j][s].solution_value > 0.9:
                    db[s][i] = j
                    _i, _j = i, j
                    if isinstance(i, str):
                        _i = (fire_stations[i]['x'], fire_stations[i]['y'])
                    if isinstance(j, str):
                        _j = (fire_stations[j]['x'], fire_stations[j]['y'])
                    arcs[s].append((_i, _j))
    clusters = set(list(db.keys()))
    color, colors = 0, ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for c in clusters:
        color += 1
        clr = colors[color % len(colors)]
        for arc in arcs[c]:
            # print(arc)
            _x, _y = zip(*arc)
            plt.plot(_x, _y, c=clr)
    print(len(clusters))
    plt.show()
    return solution


def _SetNDepots(mdl, K, W, S, x):
    print("\tSetting number of dispatches Constraint")
    mdl.add_constraint(
        mdl.sum(x[s][j][s] for s in S for j in W) <= K
    )


def _SetMaximumTourTime(mdl, t, V, W, S, x, t_bar):
    print("\tSetting Max Tour Time Constraint")
    for s in S:
        W_s = _W_s(W, s)
        W_s.add(s)
        mdl.add_constraint(
            mdl.sum(
                t[i][j] * x[i][j][s] for i in W_s for j in W if j != i
            ) <= t_bar
        )


def _SetEnduranceConstraint(mdl, E, t, W, S, x):
    print("\tSetting Endurance Constraint")
    for s in S:
        W_s = _W_s(W, s)
        mdl.add_constraint(
            mdl.sum(
                t[i][j] * x[i][j][s] for i in W_s for j in W_s if j != i
            ) <= E
        )


def _SetSubTourConstraints(mdl, M, t, W, S, z, x, *, indicator=False):
    print("\tMaking Subtour Constraints")
    # c7
    for s in S:
        for _s in S:
            try:
                mdl.add_constraint(
                    z[s][_s] == 0
                )
            except KeyError:
                pass
    # c8 c9
    if indicator:
        for s in S:
            W_s = _W_s(W, s)
            mdl.add_indicator_constraints(
                mdl.indicator_constraint(
                    x[i][j][s],
                    z[i][s] + t[i][j] == z[j][s]
                )
                for i in W_s for j in W if i != j
            )
    else:
        # c8
        for s in S:
            W_s = _W_s(W, s)
            mdl.add_constraints(
                z[i][s] + t[i][j] - z[j][s]
                <=
                M * (1 - x[i][j][s])
                for i in W_s for j in W if i != j
            )
            mdl.add_constraints(
                z[i][s] + t[i][j] - z[j][s]
                >=
                -1 * M * (1 - x[i][j][s])
                for i in W_s for j in W if i != j
            )


def _SetFlowConstraints(mdl, W, S, x):
    print("\tSetting Flow Constraint")
    for s in S:
        W_s = _W_s(W, s)
        for j in W:
            mdl.add_constraint(
                (
                        mdl.sum(
                            x[i][j][s] for i in W_s if i != j and (i in W or i == s)
                        ) -
                        mdl.sum(
                            x[j][k][s] for k in W_s if j != k and (k in W or k == s)
                        )
                ) == 0
            )


def _SetVisitedConstraints(mdl, V, W, S, x, *, allow_multiple_visits=True):
    print("\tSetting the visit all waypoints constraint")
    if allow_multiple_visits:
        mdl.add_constraints(
            mdl.sum(
                x[i][j][s] for i in V for s in S if i != j and (i in W or i == s)
            ) >= 1
            for j in W
        )
        mdl.add_constraints(
            mdl.sum(
                x[i][j][s] for j in V for s in S if i != j and (j in W or j == s)
            ) >= 1
            for i in W
        )
    else:
        mdl.add_constraints(
            mdl.sum(
                x[i][j][s] for i in V for s in S if i != j and (i in W or i == s)
            ) == 1
            for j in W
        )
        mdl.add_constraints(
            mdl.sum(
                x[i][j][s] for j in V for s in S if i != j and (j in W or j == s)
            ) == 1
            for i in W
        )
    mdl.add_constraints(
        mdl.sum(
            x[s][j][s] for j in W
        ) <= 1
        for s in S
    )
    mdl.add_constraints(
        mdl.sum(
            x[j][s][s] for j in W
        ) <= 1
        for s in S
    )


def _ObjFn(mdl, t_bar, *, t=None, x=None):
    print('\tObjFn')
    if t is not None and x is not None:
        mdl.minimize(
            mdl.sum(
                t[i][j] * x[i][j][s] for i in x for j in x[i] for s in x[i][j]
            )
        )
    else:
        mdl.minimize(t_bar)
    logging.info("Completed Building ObjFn")


def _build_variables(mdl, W, S, E):
    print('\t\tx vars')
    x = defaultdict(lambda: defaultdict(dict))
    for s in S:
        W_s = _W_s(W, s)
        for i in W_s:
            for j in W_s:
                if i != j:
                    x[i][j][s] = mdl.binary_var(name=_x_name(i, j, s))
    print('\t\ty vars')
    y = {s: mdl.binary_var(name=_y_name(s)) for s in S}
    print('\t\tz vars')
    z = defaultdict(dict)
    for s in S:
        W_s = _W_s(W, s)
        for i in W_s:
            z[i][s] = mdl.continuous_var(lb=0, ub=E, name=_z_name(i, s))
    print('\t\tt_bar vars')
    t_bar = mdl.continuous_var(lb=0, ub=E, name='t_bar')
    t_bar_s = mdl.continuous_var_dict(keys=S, lb=0, ub=E)
    logging.info(f"Completed Building Variables x:{len(x)} y:{len(y)} z:{len(z)}")
    return x, y, z, t_bar, t_bar_s


def _x_name(i, j, s):
    try:
        _i = tuple(round(e, 5) for e in i)
    except TypeError:
        if len(i) > 12:
            _i = i[:6] + i[-6:]
        else:
            _j = i
        _i = i[:6] + i[-6:]
    try:
        _j = tuple(round(e, 5) for e in j)
    except TypeError:
        if len(j) > 12:
            _j = j[:6] + j[-6:]
        else:
            _j = j
    n = 'x_{}_{}_{}'.format(_i, _j, s)
    n = n.replace('(-', 'w')  # Need to have a generic way of writing east and west ....
    n = n.replace('(', 'e')  # Need to have a generic way of writing east and west ....
    n = n.replace(', -', '_s')  # Need to have a generic way of writing east and west ....
    n = n.replace(', ', '_n')  # Need to have a generic way of writing east and west ....
    n = n.replace(' ', '_')
    n = n.replace('-', '_')
    n = n.replace(')', '')
    n = n.replace('.', '_')
    return n


def _y_name(s):
    _s = s[:6] + s[-6:]
    n = 'y_{}'.format(s)
    n = n.replace(' ', '_')
    n = n.replace('-', '_')
    n = n.replace('.', '_')
    return n


def _z_name(i, s):
    try:
        _i = round(i, 4)
    except TypeError:
        _i = i[:6] + i[-6:]
    _s = s[:6] + s[-6:]
    n = 'z_{}_{}'.format(_i, _s)
    n = n.replace('(-', 'w')  # Need to have a generic way of writing east and west ....
    n = n.replace('(', 'e')  # Need to have a generic way of writing east and west ....
    n = n.replace(', -', '_s')  # Need to have a generic way of writing east and west ....
    n = n.replace(', ', '_n')  # Need to have a generic way of writing east and west ....
    n = n.replace(' ', '_')
    n = n.replace('-', '_')
    n = n.replace(')', '')
    n = n.replace('.', '_')
    return n


def _W_s(W, s):
    W_s = W.copy()
    W_s.add(s)
    return W_s


def _build_parameters(n_depots, fire_stations, W, S):
    print("\tBuilding Parameters")
    E = 8 * 60 * 60 * 22  # 8h * 60m * 60s * 22m/s # because I dropped it somewhere....
    K = n_depots
    M = E + 1
    d_m = DistMatrix()
    t = defaultdict(dict)
    for i in W:
        for j in W:
            if i != j:
                t[i][j] = d_m.get_dist(i, j)
        for s in S:
            s_pt = (fire_stations[s]['x'], fire_stations[s]['y'])
            t[s][i] = t[i][s] = d_m.get_dist(s_pt, i)
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
        initial_solution = cfrs.cluster_first_route_second(fire_stations, waypoints, n_depots, scanning_radius)
    if name is None:
        name = "ProjectAnthony_{}".format(time.strftime("%Y%m%d_%H%M%S"))
    return initial_solution, name
