import os
from collections import defaultdict

import pulp as plp
from tqdm import tqdm

# from geospatial_toolbox.smallest_bounding_circle import find_smallest_bounding_circle
from smallest_bounding_circle import find_smallest_bounding_circle


def find_minimum_covering(groups, points, sbws=None, options=None):
    print("Building Mathematical Model")
    groups = list(groups)
    groups.sort(key=len, reverse=True)
    GroupsDict = {idx: group for idx, group in enumerate(groups)}
    PointsMembership = defaultdict(set)
    for i, group in enumerate(groups):
        for pt in group:
            PointsMembership[pt].add(i)
    model = plp.LpProblem(name="MIP Model", sense=plp.LpMinimize)
    X = {
        j: plp.LpVariable(cat=plp.LpBinary, name=f"x_{j}")
        for j in tqdm(GroupsDict.keys(), desc='X', total=len(groups), position=0)
    }
    objective = plp.lpSum(
        X[j] for j in tqdm(GroupsDict.keys(), desc='ObjFn', total=len(groups), position=0, leave=True)
    )
    for point, membership in tqdm(PointsMembership.items(), desc='Assignment Constraint', total=len(points), position=0, leave=True):
        model.add(
            plp.LpConstraint(
                plp.lpSum(X[j] * 1 for j in membership),
                sense=plp.LpConstraintGE, rhs=1
            ), point)
    model.setObjective(objective)
    print("Done Building Model")
    print("Running!")
    config = Config()
    try:
        print(f"Trying to solve with GUROBI_CMD")
        model.solve(solver=plp.GUROBI_CMD(msg=1, options=[('TimeLimit', 600)]))
    except:
        print(f"Could not solve with GUROBI_CMD")
        try:
            print(f"Trying to solve with CPLEX_CMD")
            model.solve(solver=plp.CPLEX_CMD(msg=1, options=f'set timelimit {600}'))
        except:
            print(f"Could not solve with CPLEX_CMD")
            try:
                print(f"Trying to solve with default CBC")
                model.solve(solver=plp.PULP_CBC_CMD(msg=1))
            except:
                print(f"Could not solve with CBC")
                assert False
    print("fin")
    covering = [GroupsDict[k] for k, v in X.items() if v.varValue > 0.9]
    stieners = {find_smallest_bounding_circle(pts)[1] for pts in tqdm(covering, desc="Finding Points in covering")}
    print(len(stieners))
    return stieners


class Config(object):
    solver = "GUROBI"

    def __init__(self, options=None):
        if options is None:
            options = {}

        default_options = {
            'timeLimit': 600
            , 'gap': 0.01
            , 'solver': "GUROBI"
        }

        # the following merges the two configurations (replace into):
        options = {**default_options, **options}

        self.gap = options['gap']
        self.path = options.get('path', './output/pulp')
        self.timeLimit = options['timeLimit']
        self.solver = options['solver']

    def config_gurobi(self):
        # GUROBI parameters: http://www.gurobi.com/documentation/7.5/refman/parameters.html#sec:Parameters
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        result_path = self.path + 'results.sol'
        log_path = self.path + 'results.log'
        return [('TimeLimit', self.timeLimit),
                ('ResultFile', result_path),
                ('LogFile', log_path),
                ('MIPGap', self.gap)]

    def config_cplex(self):
        # CPLEX parameters: https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.6.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/settingParams.html
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        log_path = self.path + 'results.log'
        return ['set logfile {}'.format(log_path),
                'set timelimit {}'.format(self.timeLimit),
                'set mip tolerances mipgap {}'.format(self.gap),
                # 'set mip limits treememory {}'.format(self.memory),
                ]

    def config_cbc(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        log_path = self.path + 'results.log'
        return \
            ["presolve on",
             "gomory on",
             "knapsack on",
             "probing on",
             "ratio {}".format(self.gap),
             "sec {}".format(self.timeLimit)]

    def get(self):
        return 0
        # if self.solver == "GUROBI":
        #     return model.solve(pl.GUROBI_CMD(options=self.config_gurobi(), keepFiles=1))
        # if self.solver == "CPLEX":
        #     return model.solve(pl.CPLEX_CMD(options=self.config_cplex(), keepFiles=1))
        # if self.solver == "CHOCO":
        #     return model.solve(pl.PULP_CHOCO_CMD(options=self.config_choco(), keepFiles=1, msg=0))
