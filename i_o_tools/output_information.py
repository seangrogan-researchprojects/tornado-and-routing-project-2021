import json
import logging
import os
import time

from grogies_toolbox.auto_mkdir import auto_mkdir

from routing.routing_toolbox.distance_calc import DistMatrix


def write_solution_to_json(solution, outfile, incl_time=True):
    t_now = time.strftime("%Y%m%d_%H%M%S")
    if incl_time:
        filename, file_extension = os.path.splitext(outfile)
        outfile = '{}_{}{}'.format(filename, t_now, file_extension)
    logging.info("Writing Solution File {}".format(outfile))
    auto_mkdir(outfile)
    with open(outfile, 'w') as fp:
        json.dump(solution, fp, indent=4)


def calc_distances(solution, outfile=None, incl_time=True):
    d = DistMatrix()
    dists = {}
    for t in solution:
        dist = 0
        for p1, p2 in zip(solution[t], solution[t][1:]):
            dist += d.get_dist(p1, p2)
        dists[t] = dist
    _m = max(dists.values())
    for t in dists:
        f = '*' if dists[t] == _m else ' '
        dists[t] = {'t': t, 'd': dists[t], 'f': f}
        _l = max(len(_t) for _t in solution)
        print("{f}TOUR: {t: <{_l}}\tDIST: {d}".format(f=f, t=t, d=dists[t]['d'], _l=_l))
    if outfile is not None:
        auto_mkdir(outfile)
        t_now = time.strftime("%Y%m%d_%H%M%S")
        if incl_time:
            filename, file_extension = os.path.splitext(outfile)
            outfile = '{}_{}{}'.format(filename, t_now, file_extension)
        logging.info("Writing Distance File {}".format(outfile))
        with open(outfile, 'w') as fp:
            json.dump(dists, fp, indent=4)
