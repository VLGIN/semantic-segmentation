import os
from src.dataset import CityScapesDataset


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
                          width: int = 256,
                          height: int = 256):
        train_dataset, val_dataset, test_dataset = None, None, None
        for split_mode in os.path.join(path_folder, 'leftImg8bit'):
            if split_mode == 'train':
                train_dataset = CityScapesDataset(
                    path_folder=path_folder,
                    split='train',
                    width=width,
                    height=height
                )
            elif split_mode == 'val':
                val_dataset = CityScapesDataset(
                    path_folder=path_folder,
                    split='val',
                    width=width,
                    height=height
                )
            if split_mode == 'test':
                test_dataset = CityScapesDataset(
                    path_folder=path_folder,
                    split='train',
                    width=width,
                    height=height
                )
        return cls(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)