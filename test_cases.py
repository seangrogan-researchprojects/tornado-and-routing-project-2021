import csv
import logging
import random
from time import time

import utm
from grogies_toolbox.file_readers.generic_geospatial_reader.generic_geospatial_reader import generic_geospatial_reader
from grogies_toolbox.my_logging_setup import setup_logging
from grogies_toolbox.time_for_filename import time_for_files
from shapely.geometry import Polygon, Point
from tqdm import tqdm

from i_o_tools.output_information import calc_distances
from i_o_tools.read_list_of_geospatial_files import read_list_of_geospatial_files
from i_o_tools.write_point_file import write_point_file
from main import get_points_from_roads, GrogiesTimer, convert_fire_stations_to_utm, create_waypoints
from main_utilities import get_args
from parameters.parameter_reader import parameter_reader
from plotting_toolbox.map_plotter_utm import map_plotter_utm
from routing.routing_toolbox.distance_calc import calc_dist_of_solution
from routing.solver import anthony_solver
from time_tools import add_datetime_to_pars


def test_cases(arguments, sbw_size=1000, test_cases=10, random_seed=8675349, state='OK'):
    print(f'Getting test cases')
    random.seed(a=random_seed, version=2)
    pars = parameter_reader(arguments.par)
    pars = add_datetime_to_pars(pars)
    sbw_size = pars.get('sbw_size', sbw_size)
    points, zone_number, zone_letter = get_points_from_roads(pars, arguments)
    state_bounds = get_state_bounds(zone_number, zone_letter)
    fire_stations = read_list_of_geospatial_files(pars.get("fire_stations"))
    fire_stations = convert_fire_stations_to_utm(fire_stations, zone_number, zone_letter)
    name = pars['name']
    heuristic_routes = dict()
    exact_routes = dict()
    for counter in range(test_cases):
        try:
            print(f"TESTING CASE {counter}")
            heuristic_routes[counter], exact_routes[counter] = dict(), dict()
            # generating random SBW
            sbw, pts = random_sbw(points, sbw_size, state_bounds)
            grogies_timer = GrogiesTimer()
            grogies_timer.apply("Start", pars.get("name"))
            # Getting Waypoints
            waypoints = dict()
            waypoint_methods = ['rectangular', 'hex', 'meanshift', 'random_steiner_zone',
                                'steiner']
            for method in waypoint_methods:
                logging.debug(f'Waypoint Test {method}')
                grogies_timer.apply("Generating Waypoints Prep", method)
                waypoints[method] = create_waypoints(
                    points=pts,
                    wpt_method=method,
                    scanning_radius=pars.get("scanning_radius"), sbws=sbw,
                    pars=pars,
                    args=args
                )
                map_plotter_utm(f"Waypoints Method : {method}",
                                file=f"./output/img/case{counter}-{name}-{method}.png",
                                waypoints=waypoints[method], sbws=sbw)
            for method, waypoints in waypoints.items():
                grogies_timer.apply("BeginRoutingHeuristic", method)
                solution_heuristic = anthony_solver(pars, args, fire_stations, waypoints,
                                                    solver="cluster first route second")
                grogies_timer.apply("EndRoutingHeuristic", method)
                output_soln(solution_heuristic, method, name, sbw, 'Heuristic')
                grogies_timer.apply("BeginRoutingExact", method)
                solution_exact = anthony_solver(pars, args, fire_stations, waypoints,
                                                solver="exact",
                                                warm_start=True, init_solution=solution_heuristic)
                grogies_timer.apply("EndRoutingExact", method)
                comparison_csv(counter, method, waypoints, solution_exact, solution_heuristic)
                output_soln(solution_exact, method, name, sbw, 'Exact')
        except:
            print("there was an exception, skipping...")
            logging.exception("there was an exception, skipping...")


def comparison_csv(case, method, waypoints, solution_exact, solution_heuristic, addl_text=time_for_files()):
    line = [case, method, len(waypoints), calc_dist_of_solution(solution_exact), calc_dist_of_solution(solution_heuristic)]
    with open(f"comparison{addl_text}.csv", newline='', mode='a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(line)




def output_soln(solution, method, name, sbw, ext_heur):
    print(f"=======================================")
    print(f"ObjFn:{calc_dist_of_solution(solution)}")
    print(f"=======================================")
    time_hr = calc_dist_of_solution(solution) / (22 * 60 * 60)
    time_hh, time_m = int(time_hr), int((time_hr * 60) % 60)
    map_plotter_utm(f"Routes {ext_heur} : {method} (t={time_hh}h{time_m:0=2d}m)",
                    file=f"./output/img/tours_{name}_{method}.png",
                    solution=solution, sbws=sbw)
    return 0


def waypoint_generation_tests(pars, points, sbws, tests=None, args=None, fire_stations=None, timer=None, solver='cfrs'):
    print('Executing Waypoint Tests')
    logging.info('Executing Waypoint Tests')
    name = pars.get('name', "anthony")
    if tests is None:
        waypoint_methods = ['rectangular', 'hex', 'meanshift', 'random_steiner_zone',
                            'steiner'] if tests is None else tests
    waypoints, tme, num_wpts = dict(), dict(), dict()
    for method in waypoint_methods:
        logging.debug(f'Waypoint Test {method}')
        timer.apply("Generating Waypoints Prep", method)
        t1 = time()
        waypoints[method] = create_waypoints(
            points=points,
            wpt_method=method,
            scanning_radius=pars.get("scanning_radius"), sbws=sbws,
            pars=pars,
            args=args
        )
        for k, v in waypoints.items():
            map_plotter_utm(k, file=f"./output/img/{name}_{k}.png", waypoints=v, sbws=sbws)
        num_wpts[method] = len(waypoints[method])
        tme[method] = time() - t1
        timer.apply("Generating Waypoints", method)
        write_point_file(method, f"./output/points/{name}_{method}.json", waypoints[method])
        timer.apply("Writing Waypoints", method)

    print("===========================")
    print("==== Waypoint Results =====")
    print("===========================")
    for k, v in waypoints.items():
        print(f"{k:{max(len(_) for _ in waypoints.keys())}} | NumWpts:{len(v):8} | Time: {tme[k]:.2f}")
        logging.info(f"{k:>} :: NumWpts : {len(v)} :: Time : {tme[k]}")
    for k, v in waypoints.items():
        map_plotter_utm(k, file=f"./output/img/{name}_{k}.png", waypoints=v, sbws=sbws)
    for method in waypoint_methods:
        wpts = waypoints[method]
        timer.apply("BeginRouting", method)
        solution_exact = anthony_solver(pars, args, fire_stations, wpts, solver=solver)
        solution_heuristic = anthony_solver(pars, args, fire_stations, wpts, solver="cluster first route second")
        timer.apply("Routing", method)
        calc_distances(solution_exact, outfile=f"./output/distances/distances_{name}_{method}_exact.json",
                       incl_time=True)
        calc_distances(solution_heuristic, outfile=f"./output/distances/distances_{name}_{method}_heuristic.json",
                       incl_time=True)
        data = dict()
        for k, v in solution.items():
            data[k] = [list(int(ele) for ele in wp) for wp in v]
        write_solution_to_json(data, outfile=f"./output/solution/solution_{name}_{method}.json", incl_time=True)
        map_plotter_utm(f"{method} (t={calc_dist_of_solution(data):.6})",
                        file=f"./output/img/tours_{name}_{method}.png",
                        solution=data, sbws=sbws)


def random_sbw(points, size, bounds):
    print("Generating SBW")
    x0, y0 = -9_999_999, -9_999_999
    pts = []
    pbar = tqdm(desc="Generating SBW")
    while len(pts) <= 5:
        pbar.update()
        x0, y0 = random.choice(list(points))
        x1, y1 = x0 + size, y0 + size
        sbw = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        pts = [pt for pt in points if sbw.contains(Point(pt))]
    sbw = [dict(shape_utm=dict(points=[(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]), phenom="TO")]
    return sbw, pts


def get_state_bounds(zone_number, zone_letter):
    state_bounds = generic_geospatial_reader("D:/Maps/Data/UnitedStates/Oklahoma/StateBoundsOK/StateBoundsOK.shp").pop(
        0)
    state_bounds = [utm.from_latlon(lat, lon, zone_number, zone_letter) for lon, lat in state_bounds['shape'].points]
    state_bounds = [(x, y) for x, y, _, __ in state_bounds]
    state_bounds = Polygon(state_bounds)
    return state_bounds


if __name__ == '__main__':
    args = get_args()
    setup_logging()
    try:
        test_cases(args)
    except:
        logging.exception("EXCEPTION!")
