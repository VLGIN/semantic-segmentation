from abc import ABC
import os

from segmentation_models_pytorch import Unet
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import optim

from src.dataset import DataSource
from src.trainer import TrainerBase
from src.utils.io import save_json
import logging

logger = logging.getLogger(__name__)


class UnetTrainer(TrainerBase, ABC):
    def __init__(self, arg, model: Unet, data_source: DataSource, **kwargs):
        super().__init__(**kwargs)
        self.config = {}
        self.arg = arg
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.arg.is_train:
            self.train_loader = self.make_loader(data_source.train_dataset)
            self.val_loader = self.make_loader(data_source.val_dataset)
        elif self.arg.is_test:
            self.test_loader = self.make_loader(data_source.test_dataset)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.arg.learning_rate)
        self.path_save = f"{self.arg.path_save}/{self.__class__.__name__}"
        if os.path.exists(self.path_save) is False:
            os.makedirs(self.path_save, exist_ok=True)

        if self.arg.continue_training:
            self.model.load_state_dict(torch.load(os.path.join(self.path_save, 'model.pth')))
            self.optimizer.load_state_dict(torch.load(os.path.join(self.path_save, 'optimizer.pth')))

        self.config['model'] = {
            'architecture': 'unet',
            'path_save': self.path_save
        }
        self.config['hyperparameter'] = {
            'batch_size': self.arg.batch_size,
            'epoch': self.arg.epoch,
            'learning_rate': self.arg.learning_rate,
        }
        save_json(self.config, f"{self.path_save}/config_model.json")


    def make_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.arg.batch_size, shuffle=True, num_workers=2)

    def save_model(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass

    def train_one_epoch(self, epoch: int, **kwargs):
        logger.info(f"Training epoch: {epoch}")
        for idx, batch in enumerate(self.train_loader):
            image = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)
            output = self.model(image)
            loss = self.loss_fn()

    def fit(self, **kwargs):
        for epoch in
