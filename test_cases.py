import random

import utm
from grogies_toolbox.file_readers.generic_geospatial_reader.generic_geospatial_reader import generic_geospatial_reader
from shapely.geometry import Polygon, Point

from i_o_tools.read_list_of_geospatial_files import read_list_of_geospatial_files
from main import get_points_from_roads, GrogiesTimer, convert_fire_stations_to_utm, waypoint_generation_tests
from main_utilities import get_args
from parameters.parameter_reader import parameter_reader
from time_tools import add_datetime_to_pars


def test_cases(arguments, sbw_size=1000, test_cases=10, random_seed=12345, state='OK'):
    random.seed(a=random_seed, version=2)
    pars = parameter_reader(arguments.par)
    pars = add_datetime_to_pars(pars)
    points, zone_number, zone_letter = get_points_from_roads(pars, arguments)
    state_bounds = get_state_bounds(zone_number, zone_letter)
    fire_stations = read_list_of_geospatial_files(pars.get("fire_stations"))
    fire_stations = convert_fire_stations_to_utm(fire_stations, zone_number, zone_letter)
    for counter in range(test_cases):
        print(f"TESTING CASE {counter}")
        sbw, pts = random_sbw(points, sbw_size, state_bounds)
        grogies_timer = GrogiesTimer()
        grogies_timer.apply("Start", pars.get("name"))
        waypoint_generation_tests(pars, pts, sbw, tests=None,
                                  args=args, fire_stations=fire_stations, timer=grogies_timer)
        pass
    return 0


def random_sbw(points, size, bounds):
    x0, y0 = -9_999_999, -9_999_999
    while not bounds.contains(Point(x0, y0)):
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
    test_cases(args)
