import random
import sys


from grogies_toolbox.auto_mkdir import auto_mkdir

from main_utilities import euclidean, midpoint


def find_smallest_bounding_circle(points):
    """This function takes a list of points and converts it into a a cricle of radius and center.
    :param points: iterable of points
    :return tuple, circle radius (as a float) and circle center (as a tuple)"""
    points = list(points)
    random.shuffle(points)
    # Initial Cases
    if len(points) == 0:  # Silly case
        return 0, (None, None)
    if len(points) == 1:  # Also a silly case
        return 0, tuple(points[0])
    if len(points) == 2:  # A less silly case, but trivial
        return euclidean(points[0], points[1]) / 2, midpoint(points[0], points[1])
    # regular cases
    limit = int(sys.getrecursionlimit()*.9)
    if len(points) > limit:
        sub_cases = [points[i * limit:(i + 1) * limit] for i in range((len(points) + limit - 1) // limit)]
        radial_points = set()
        for case in sub_cases:
            _case = case + list(radial_points)
            disk, radial_points = calculate_smallest_enclosing_disk(_case, set())
    else:
        disk, radial_points = calculate_smallest_enclosing_disk(points, set())
    return disk.radius, disk.center

#
# class StackElement:
#     def __init__(self, current_point=None, points=None, radial_points=None, disk=None):
#         self.current_point = current_point
#         self.points = points.copy() if points else []
#         self.radial_points = radial_points.copy() if radial_points else []
#         self.disk = disk.copy() if disk else Disk()
#

# def calculate_smallest_enclosing_disk_stack(points):
#     stack = []
#     radial_points = set()
#     points = points.copy()
#     stack.append(StackElement(points.pop(), points=points))
#     while stack:
#         current_element = stack.pop(0)
#         if len(current_element.points) + len(current_element.radial_points) <= 3:
#         if test_if_all_points_in_circle(points, disk.center, disk.radius):
#             return
#
#
# def test_if_all_points_in_circle(points, center, radius):
#     return all(point_is_in_circle(point, center, radius) for point in points)
#
#
# def calculate_boundary_from_points(points, radial_points):
#     if len(radial_points) == 3:
#         disk = CreateCircle.calculate_3pt_circle(radial_points)
#     elif len(points) == 1 and len(radial_points) == 0:
#         disk = CreateCircle.calculate_1pt_circle(points)
#     elif len(points) == 0 and len(radial_points) == 2:
#         disk = CreateCircle.calculate_2pt_circle(radial_points)
#     elif len(points) == 1 and len(radial_points) == 1:
#         disk = CreateCircle.calculate_2pt_circle(list(radial_points) + list(points))
#     return disk


def calculate_smallest_enclosing_disk(points, radial_points=None):
    radial_points = set() if radial_points is None else radial_points
    points = points.copy()
    radial_points = radial_points.copy()
    if len(radial_points) == 3:
        disk = CreateCircle.calculate_3pt_circle(radial_points)
    elif len(points) == 1 and len(radial_points) == 0:
        disk = CreateCircle.calculate_1pt_circle(points)
    elif len(points) == 0 and len(radial_points) == 2:
        disk = CreateCircle.calculate_2pt_circle(radial_points)
    elif len(points) == 1 and len(radial_points) == 1:
        disk = CreateCircle.calculate_2pt_circle(list(radial_points) + list(points))
    else:
        pt = points.pop()
        disk, _ = calculate_smallest_enclosing_disk(points, radial_points)
        if not point_is_in_circle(pt, disk.center, disk.radius):
            radial_points.add(pt)
            disk, _ = calculate_smallest_enclosing_disk(points, radial_points)
    return disk, radial_points


def point_is_in_circle(point, center, radius):
    if center == (None, None):
        return False
    return euclidean(point, center) <= radius


class Disk:
    radius, center = 0, (None, None)

    def __init__(self, radius=0, center=(None, None)):
        self.radius = radius
        self.center = center

    def __str__(self):
        return f"(R:{self.radius:.4f}|C:{self.center})"

    def __copy__(self):
        return Disk(self.radius, self.center)


class CreateCircle:
    def __init__(self):
        pass

    @staticmethod
    def calculate_0pt_circle(points):
        points = list(points)
        assert len(points) == 0
        return Disk(0, (None, None))

    @staticmethod
    def calculate_1pt_circle(points):
        points = list(points)
        assert len(points) == 1
        return Disk(0, tuple(points[0]))

    @staticmethod
    def calculate_2pt_circle(points):
        assert len(points) == 2
        points = list(points)
        return Disk(euclidean(points[0], points[1]) / 2, midpoint(points[0], points[1]))

    @staticmethod
    def calculate_3pt_circle(points):
        points = list(points)
        assert len(points) == 3
        p1, p2, p3 = list(points)
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        b = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1)
        c = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2)
        x = -b / (2 * a)
        y = -c / (2 * a)
        # if -0.1 < a < 0.1:
        #     logging.warning(f"p1={p1}")
        #     logging.warning(f"p2={p2}")
        #     logging.warning(f"p3={p3}")
        #     logging.warning(f" a={a}")
        #     logging.warning(f" b={b}")
        #     logging.warning(f" c={c}")
        #     logging.warning(f"cn={x} , {y}")
        return Disk(center=(x, y), radius=euclidean((x, y), p1))


def test(id_code, _min=-10, _max=10):
    pts = {(random.randint(_min, _max), random.randint(_min, _max)) for _ in range(30)}
    radius, center = find_smallest_bounding_circle(pts)
    x, y = zip(*pts)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(x, y, zorder=1000, color='orange')
    ax.scatter(center[0], center[1], zorder=1000, color='yellow')
    ax.add_artist(plt.Circle(center, radius=radius, zorder=10))
    ax.axis('equal')
    ax.set_xlim(_min - 10, _max + 10)
    ax.set_ylim(_min - 10, _max + 10)
    plt.title(f'Minimum Bounding Circles\nRadius : {radius:.4f} units | Center : {center[0]:.2f}, {center[1]:.2f}')
    fp = f'./test_work_v2/{id_code}.png'
    auto_mkdir(fp)
    plt.savefig(fp)
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    for i in range(100):
        test(i)
