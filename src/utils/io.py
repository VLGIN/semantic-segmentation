import json
import logging
import os
import glob
from matplotlib import pyplot as plt
from collections import defaultdict

logger = logging.getLogger(__name__)


def save_config_architecture(data, path_file):
    with open(path_file, 'w') as file:
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


def plot_loss(history: defaultdict, path_save: str):
    range_x = [i for i in range(len(history['train_loss']))]
    plt.plot(range_x, history['train_loss'], label='train_loss')
    plt.plot(range_x, history['val_loss'], label='val_loss')
    plt.legend(loc="upper left")
    plt.savefig(f'{path_save}/loss.png')
    logger.info(f"Save loss.png at {path_save}")
    plt.close()


def plot_iou_score(history: defaultdict, path_save: str):
    range_x = [i for i in range(len(history['train_iou']))]
    plt.plot(range_x, history['train_iou'], label='train_iou')
    plt.plot(range_x, history['val_iou'], label='val_iou')
    plt.legend(loc="upper left")
    plt.savefig(f'{path_save}/iou.png')
    logger.info(f"Save iou.png at {path_save}")
    plt.close()
