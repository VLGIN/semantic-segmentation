import json
import logging
import os
import glob

logger = logging.getLogger(__name__)


def save_config_architecture(data, path_file):
    with open(path_file,'w') as file:
        json.dump(data, file, indent=4)
    logger.info(f"Save architecture done in {path_file}")


def load_config_architecture(model_path):
    model_path = os.path.abspath(model_path)
    config_architecture = glob.glob(model_path + '/*model.json')
    if len(config_architecture) == 0:
        raise Exception("Not found config architecture file")
    else:
        config_path = config_architecture[0]
    with open(config_path, 'r') as file:
        config = json.load(file)

    return config
