import logging
from math import sqrt
import numpy as np
from sklearn.cluster import MeanShift


def generate_waypoints_mean_shift_utm(points, scanning_radius):
    bandwidth = sqrt(2) * scanning_radius / 2
    print("Generating Waypoints")
    logging.info("Generating Waypoints using MeanShift")
    if not isinstance(points, np.ndarray):
        points = np.array(list(points))
    print("Applying Mean-shift to create waypoints")
    print('\tBandwith', bandwidth)
    clustering = MeanShift(bandwidth=bandwidth).fit(points)
    centers = clustering.cluster_centers_
    waypoints = set(tuple((x, y)) for x, y in list(centers))
    return waypoints
