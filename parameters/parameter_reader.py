import json
import logging

from grogies_toolbox.ensure_extension import ensure_extension


def parameter_reader(par_file_name=None):
    print(f"Reading parameter file {par_file_name}")
    logging.info(f"Reading parameter file {par_file_name}")
    if par_file_name is None:
        print(f"No parameter file name was passed!")
        logging.critical(f"No parameter file name was passed!")
        exit(-99)
    try:
        par_file_name = ensure_extension(par_file_name, "json")
        with open(par_file_name) as f:
            par_file = json.load(f)
    except:
        print(f"Parameter file {par_file_name} could not be loaded!")
        logging.exception(f"Parameter file {par_file_name} could not be loaded!")
        raise
    print(f"Read parameter file {par_file_name}!")
    logging.info(f"Read parameter file {par_file_name}!")
    [logging.debug(f"Par : {key} \t Value : {val}") for key, val in par_file.items()]
    return par_file
