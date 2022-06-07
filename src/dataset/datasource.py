import numpy as np
import os

from src.dataset.data import ImageDataset
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes

class DataSource():
    def __init__(self, train_dataset: ImageDataset=None,
                valid_dataset: ImageDataset=None,
                test_dataset: ImageDataset=None, **kwargs):
        super(DataSource, self).__init__
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    @classmethod
    def create_datasource_from_path(cls, args=None,
                                    path_to_folder_data: str=None,
                                    ):
        assert os.path.exists(path_to_folder_data), "Invalid folder path"

        data = Cityscapes(path_to_folder_data,split ='train',target_type='semantic')
        train_dataset = ImageDataset(data = data )
        data = Cityscapes(path_to_folder_data,split ='test',target_type='semantic')
        test_dataset = ImageDataset(data = data )
        data = Cityscapes(path_to_folder_data,split ='val',target_type='semantic')
        valid_dataset = ImageDataset(data = data )

        return cls(train_dataset=train_dataset,
                   valid_dataset=valid_dataset,
                   test_dataset=test_dataset)
