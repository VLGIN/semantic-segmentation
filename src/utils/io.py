import json
import logging

logger = logging.getLogger(__name__)


def save_json(data, path_file):
    with open(path_file,'w') as file:
        json.dump(data, file, indent=4)
    logger.info(f"Save architecture done in {path_file}")