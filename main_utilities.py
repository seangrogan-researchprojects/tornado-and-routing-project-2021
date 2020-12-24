import argparse
import logging
import math
from collections import Counter
from math import sqrt

import utm
from numpy import average
from shapely.geometry import Polygon, Point

from i_o_tools.parse_weather_files import find_applicable_lsrs, find_applicable_sbws
from i_o_tools.read_list_of_geospatial_files import read_list_of_geospatial_files
from time_tools import curate_by_time


def get_args():
    parser = argparse.ArgumentParser(
        description='Project Anthony',
        epilog='Questions?  sean.grogan@polymtl.ca'
    )
    parser.add_argument(
        '-par', '-parameter', '-parameters', '-p',
        action='store', default=None,
        help='Path to parameter file'
    )
    parser.add_argument(
        '-utm_mp',
        action='store_true', default=False,
        help='Utilize a multiprocessing method for computing the UTM coordinates.  '
             'May be faster for large large large number of roads'
    )
    return parser.parse_args()


def euclidean(p1, p2):
    return sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def midpoint(p1, p2):
    return tuple((j+i)/2 for i, j in zip(p1, p2))


def strip_roads(roads):
    print(f"Stripping road files")
    logging.info(f"Stripping road files")
    roads = [path.get('shape').points for path in roads]
    print(f"Stripped road files!")
    logging.info(f"Stripped road files!")
    return roads


def convert_roads_to_utm(roads, quick=True):
    print("Converting Coordinates to UTM")
    logging.info("Converting Coordinates to UTM")
    lons, lats = zip(*[(i, j) for road in roads for i, j in road])
    if quick:
        lat, lon = average(lats), average(lons)
        east, nord, zn, zl = utm.from_latlon(lat, lon)
    else:
        east, nord, zn, zl = zip(*[utm.from_latlon(lat, lon) for lat, lon in zip(lats, lons)])
        zn = Counter(zn).most_common(1).pop()[0]
        zl = Counter(zl).most_common(1).pop()[0]
    roads_utm = [[utm.from_latlon(lat, lon, zn, zl) for lon, lat in road] for road in roads]
    print("Converted Coordinates to UTM")
    logging.info("Converted Coordinates to UTM")
    return roads_utm, zn, zl


def simplify_points(roads, round_pts_to=1):
    n_digits = math.floor(math.log10(round_pts_to))
    print(f"Rounding UTM Data to the nearest 10^{-1 * n_digits}")
    logging.info(f"Rounding UTM Data to the nearest 10^{-1 * n_digits}")
    points = {(round(float(i), -1 * n_digits), round(float(j), -1 * n_digits)) for road in roads for i, j in road}
    print(f"Rounded UTM Data to the nearest 10^{-1 * n_digits}!")
    logging.info(f"Rounded UTM Data to the nearest 10^{-1 * n_digits}")
    return points


def strip_utm_data(roads):
    print("Stripping UTM Data")
    return [[(x, y) for x, y, n, l in road] for road in roads]


def my_round(x, precision=0, base=5):
    """
    :param x: Number to round
    :param precision: rounded to the multiple of 10 to the power minus precision
    :param base: to the nearest 'base'
    :return: rounded value
    """
    return round(base * round(float(x) / base), int(precision))


def count_zeroes(d):
    """Counts the number of zeros in a number.
    """
    return int(math.log10(d))


def read_storm_data(sbw_files, lsr_files):
    print(f"Reading storm data")
    sbws, lsrs = read_list_of_geospatial_files(sbw_files), read_list_of_geospatial_files(lsr_files)
    print(f"Read storm data")
    return sbws, lsrs


def get_overlapping_data(sbws, lsrs):
    new_sbws = []
    for sbw in sbws:
        polygon = Polygon(sbw['shape_utm']['points'])
        for lsr in lsrs:
            point = Point(*lsr['shape_utm']['points'][0])
            if polygon.contains(point):
                new_sbws.append(sbw)
                break
    return new_sbws, lsrs


def get_relevant_storm_data(sbws, lsrs, pars):
    lsrs = find_applicable_lsrs(lsrs, typecode=('TO', 'T'))
    sbws = find_applicable_sbws(sbws, phenom={'TORNADO', 'T', 'TO'}, gtype={'P'})
    sbws, lsrs = curate_by_time(sbws, lsrs, pars)
    return sbws, lsrs


def convert_storm_data_to_utm(sbws, lsrs, zone_number, zone_letter):
    for entry in lsrs + sbws:
        points = []
        for point in entry['shape'].points:
            e, n, zn, zl = utm.from_latlon(
                latitude=point[1],
                longitude=point[0],
                force_zone_number=zone_number,
                force_zone_letter=zone_letter
            )
            points.append((int(e), int(n)))
        entry['shape_utm'] = dict(
            parts=entry['shape'].parts,
            points=points,
            shapeType=entry['shape'].shapeType,
            shapeTypeName=entry['shape'].shapeTypeName
        )
    return sbws, lsrs


def get_relevant_points(points, sbws):
    new_points = set()
    for sbw in sbws:
        polygon = Polygon(sbw['shape_utm']['points'])
        for pt in points:
            if polygon.contains(Point(*pt)):
                new_points.add(pt)
    return new_points
