# from geospatial_toolbox.distance_calcualtor import euclidean_dist
from main_utilities import euclidean


def calc_dist_of_tour(tour, d=None):
    if d is None:
        d = DistMatrix()
    _temp_d = 0
    for i, j in zip(tour, tour[1:]):
        _temp_d += d.get_dist(i, j)
    return _temp_d


def calc_dist_of_solution(solution, d=None):
    if d is None:
        d = DistMatrix()
    return max(calc_dist_of_tour(tour) for tour in solution.values())


class DistMatrix:
    def __init__(self):
        pass

    @staticmethod
    def get_dist(p1, p2):
        return euclidean(p1, p2)
