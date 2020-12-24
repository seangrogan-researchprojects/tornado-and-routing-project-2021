import logging

from grogies_toolbox.file_readers.generic_geospatial_reader.generic_geospatial_reader import generic_geospatial_reader


def read_list_of_geospatial_files(files):
    print(f"Reading file(s)")
    logging.info(f"Reading file(s)")
    if isinstance(files, str):
        data = generic_geospatial_reader(files)
    else:
        data = [generic_geospatial_reader(file) for file in files]
        data = [entry for file in data for entry in file]
    print(f"Read file(s)!")
    logging.info(f"Read file(s)!")
    return data
