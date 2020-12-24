import json
import logging
from collections import defaultdict
from time import time
import pandas as pd
import utm
from numpy import average

from ensure_road_point_distances import ensure_road_point_distance
from i_o_tools.output_information import calc_distances, write_solution_to_json
from i_o_tools.read_list_of_geospatial_files import read_list_of_geospatial_files
from i_o_tools.write_point_file import write_point_file
from main_utilities import get_args, strip_roads, convert_roads_to_utm, simplify_points, strip_utm_data, \
    read_storm_data, get_overlapping_data, get_relevant_storm_data, convert_storm_data_to_utm, get_relevant_points
from parameters.parameter_reader import parameter_reader
from plotting_toolbox.map_plotter_utm import map_plotter_utm
from routing.routing_toolbox.distance_calc import calc_dist_of_solution
from routing.solver import anthony_solver
from time_tools import add_datetime_to_pars
from waypoint_generation.hexagon import hexagon_pattern
from waypoint_generation.retangular import rectangular
from waypoint_generation.steiner_inspired.steiner_zone import steiner_zone
from waypoint_generation.steiner_inspired.random_steiner_zone import random_steiner_zone
from waypoint_generation.wpts_mean_shift import generate_waypoints_mean_shift_utm


def convert_fire_stations_to_utm(fire_stations, zone_number=None, zone_letter=None):
    if zone_letter and zone_number:
        fire_stations = {
            station['name']:
                utm.from_latlon(station['y'], station['x'], zone_number, zone_letter) for station in fire_stations
        }
    else:
        lats, lons = [station['y'] for station in fire_stations], [station['x'] for station in fire_stations]
        lat, lon = average(lats), average(lons)
        east, nord, zn, zl = utm.from_latlon(lat, lon)
    fire_stations = {name: (round(x), round(y)) for name, (x, y, _i, _j) in fire_stations.items()}
    return fire_stations


def get_points_from_roads(pars, arguments):
    roads = read_list_of_geospatial_files(pars.get("road_files"))
    roads = strip_roads(roads)
    roads, zone_number, zone_letter = convert_roads_to_utm(roads, pars.get("quick_utm_convert", True))
    roads = strip_utm_data(roads)
    roads = ensure_road_point_distance(roads, pars.get("scanning_radius"), arguments.utm_mp)
    points = simplify_points(roads, pars.get("round_pts_to", 1))
    return points, zone_number, zone_letter


def get_storm_data(pars, zone_number, zone_letter):
    sbws, lsrs = read_storm_data(pars.get("sbws"), pars.get("lsrs"))
    sbws, lsrs = get_relevant_storm_data(sbws, lsrs, pars)
    sbws, lsrs = convert_storm_data_to_utm(sbws, lsrs, zone_number, zone_letter)
    sbws, lsrs = get_overlapping_data(sbws, lsrs)
    return sbws, lsrs


def main(parameters_file=None, arguments=None):
    pars = parameter_reader(parameters_file)
    pars = add_datetime_to_pars(pars)
    points, zone_number, zone_letter = get_points_from_roads(pars, arguments)
    sbws, lsrs = get_storm_data(pars, zone_number, zone_letter)
    fire_stations = read_list_of_geospatial_files(pars.get("fire_stations"))
    fire_stations = convert_fire_stations_to_utm(fire_stations, zone_number, zone_letter)
    wpt_method = pars.get("wpt_method", "rectangular")
    scanning_radius = pars.get("scanning_radius", 300)
    points = get_relevant_points(points, sbws)
    grogies_timer = GrogiesTimer()
    grogies_timer.apply("Start", pars.get("name"))
    waypoint_generation_tests(pars, points, sbws, tests=None,
                              args=args, fire_stations=fire_stations, timer=grogies_timer)
    # waypoints = create_waypoints(points, wpt_method, scanning_radius, sbws, pars, args)
    # for key, wpts in waypoints.items():
    #     map_plotter_utm(key, file=None, display=True, mp=True, waypoints=wpts)

    return 0


def create_waypoints(points, wpt_method, scanning_radius, sbws, pars, args):
    if not isinstance(wpt_method, str):
        results = dict()
        for method in wpt_method:
            results[method] = create_waypoints(points, method, scanning_radius, sbws, pars, args)
        return results
    else:
        if wpt_method.lower() in {"rectangular", "rec", "square"}:
            return rectangular(points, scanning_radius)
        elif wpt_method.lower() in {'hex', 'hexagon'}:
            return hexagon_pattern(points, pars, pars.get('flat_top', True))
        elif wpt_method.lower() in {'meanshift', 'mean_shift', 'mean shift', 'mean'}:
            return generate_waypoints_mean_shift_utm(points, scanning_radius)
        elif wpt_method.lower() in {'stiener', 'steiner_zone', 'steiner', "steiner zone"}:
            return steiner_zone(points, pars)
        elif wpt_method.lower() in {'random stiener', 'random_steiner_zone', 'random steiner', "random steiner zone"}:
            return random_steiner_zone(points, pars, improve_please=True)
        elif wpt_method.lower() in {'random stiener no improve', 'random_steiner_zone_no_improve'}:
            return random_steiner_zone(points, pars, improve_please=False)
        print(f"{wpt_method} not implemented")
        return set()
    return (None, None),


def waypoint_generation_tests(pars, points, sbws, tests=None, args=None, fire_stations=None, timer=None):
    print('Executing Waypoint Tests')
    logging.info('Executing Waypoint Tests')
    name = pars.get('name', "anthony")
    if tests is None:
        waypoint_methods = ['rectangular', 'hex', 'meanshift', 'random_steiner_zone', 'steiner']
        waypoint_methods.reverse()
    else:
        waypoint_methods = tests
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
        num_wpts[method] = len(waypoints[method])
        tme[method] = time() - t1
        timer.apply("Generating Waypoints", method)
        write_point_file(method, f"./output/points/{name}_{method}.json", waypoints[method])
        timer.apply("Writing Waypoints", method)
        # print_waypoint_check(points, waypoints[method], pars.get('scanning_r', 300), sbws, name=method)
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
        solution = anthony_solver(pars, args, fire_stations, wpts, solver='cfrs')
        timer.apply("Routing", method)
        calc_distances(solution, outfile=f"./output/distances/distances_{name}_{method}.json", incl_time=True)
        data = dict()
        for k, v in solution.items():
            data[k] = [list(int(ele) for ele in wp) for wp in v]
        write_solution_to_json(data, outfile=f"./output/solution/solution_{name}_{method}.json", incl_time=True)
        map_plotter_utm(f"{method} (t={calc_dist_of_solution(data):.6})",
                        file=f"./output/img/tours_{name}_{method}.png",
                        solution=data, sbws=sbws)


class GrogiesTimer:
    last, cases, steps, time_stamps = None, None, None, None

    def __init__(self, init_time=None):
        self.last = time() if init_time is None else init_time
        # self.last = self.start_time
        self.cases = ['rectangular', 'hex', 'meanshift', 'random_steiner_zone_no_improve', 'random_steiner_zone',
                      'steiner_zone']
        self.steps = ["Debut", "Reading Files", "Processing Files", "Generating Waypoints", "Routing"]
        self.time_stamps = defaultdict(dict)
        for case in self.cases:
            self.time_stamps[case]["Debut"] = 0

    def apply(self, step, case=None):
        now = time()
        time_step = now - self.last
        if case is None:
            for case in self.cases:
                self.time_stamps[case][step] = time_step
        else:
            self.time_stamps[case][step] = time_step
        self.last = now

    def write_info(self, path, file_name):
        with open(f"./{path}/{file_name}.json", "w") as jsonfile:
            json.dump(self.time_stamps, fp=jsonfile, indent=4)
        df = pd.DataFrame(self.time_stamps)
        df.to_csv(f"./{path}/{file_name}.csv")


if __name__ == '__main__':
    args = get_args()
    main(args.par, args)
