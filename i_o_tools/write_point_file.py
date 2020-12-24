import json
import logging

from grogies_toolbox.auto_mkdir import auto_mkdir
from grogies_toolbox.ensure_extension import ensure_extension


def write_point_file(name, file, points):
    logging.info(f'Writing point file {name}')
    data = {name: list(list(pt) for pt in points)}
    file = ensure_extension(file, 'json')
    auto_mkdir(file)
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
