import os
from src.dataset import CityScapesDataset
import logging

logger = logging.getLogger(__name__)


class DataSource:
    def __init__(self,
                 train_dataset: CityScapesDataset = None,
                 val_dataset: CityScapesDataset = None,
                 test_dataset: CityScapesDataset = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @classmethod
    def create_datasource(cls,
                          path_folder,
                          resize_mode: bool = False,
                          width: int = 256,
                          height: int = 256):
        train_dataset, val_dataset, test_dataset = None, None, None
        for split_mode in os.listdir(os.path.join(path_folder, 'leftImg8bit')):
            if split_mode == 'train':
                train_dataset = CityScapesDataset(
                    path_folder=path_folder,
                    split='train',
                    resize_mode=resize_mode,
                    width=width,
                    height=height
                )
                logger.info("Create train dataset done")

            elif split_mode == 'val':
                val_dataset = CityScapesDataset(
                    path_folder=path_folder,
                    split='val',
                    resize_mode=resize_mode,
                    width=width,
                    height=height
                )
                logger.info("Create val dataset done")

            if split_mode == 'test':
                test_dataset = CityScapesDataset(
                    path_folder=path_folder,
                    split='train',
                    resize_mode=resize_mode,
                    width=width,
                    height=height
                )
                logger.info("Create test dataset done")

        return cls(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)