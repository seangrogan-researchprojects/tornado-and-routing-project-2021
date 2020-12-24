import json
import logging

from grogies_toolbox.auto_mkdir import auto_mkdir


def _json_writer(output_data, file):
    logging.debug('Writing to file \'{file}\''.format(file=file))
    auto_mkdir(file)
    try:
        with open(file, 'w') as json_output:
            json.dump(output_data, file)
    except:
        logging.exception("Exception occurred")
