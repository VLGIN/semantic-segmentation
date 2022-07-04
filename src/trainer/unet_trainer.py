from abc import ABC
import os
import sys
from collections import defaultdict

from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.utils.metrics import IoU
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import optim

from src.dataset import DataSource
from src.trainer import TrainerBase
from src.utils.io import *
import logging

logger = logging.getLogger(__name__)


class UnetTrainer(TrainerBase, ABC):
    def __init__(self, arg, model: Unet, data_source: DataSource, **kwargs):
        super().__init__(**kwargs)
        self.config = {}
        self.arg = arg
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history = defaultdict(list)
        self.iou_fn = IoU()

        if self.arg.is_train:
            self.train_loader = self.make_loader(data_source.train_dataset)
            self.val_loader = self.make_loader(data_source.val_dataset)
        elif self.arg.is_test:
            self.test_loader = self.make_loader(data_source.test_dataset)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.arg.learning_rate)
        self.path_save = f"{self.arg.path_save}/{self.__class__.__name__}"
        self.best_val_loss = sys.maxsize

        if os.path.exists(self.path_save) is False:
            os.makedirs(self.path_save, exist_ok=True)

        if self.arg.continue_training:
            self.config = load_config_architecture(self.arg.path_save)
            self.best_val_loss = self.config['weight_information'].get('best_val_loss', self.best_val_loss)
            self.model.load_state_dict(torch.load(os.path.join(self.path_save, 'model.pth')))
            self.optimizer.load_state_dict(torch.load(os.path.join(self.path_save, 'optimizer.pth')))
        else:
            self.config['model'] = {
                'architecture': 'unet',
            }
            self.config['hyperparameter'] = {
                'batch_size': self.arg.batch_size,
                'epoch': self.arg.num_epoch,
                'learning_rate': self.arg.learning_rate,
            }

    def make_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.arg.batch_size, shuffle=True, num_workers=2)

    def save_model(self, epoch, **kwargs):
        self.config['weight_information'] = {
            'best_val_loss': self.best_val_loss,
            'epoch': epoch
        }
        save_config_architecture(self.config, f"{self.path_save}/config_model.json")
        torch.save(self.model.state_dict(), f"{self.path_save}/model.pth")
        torch.save(self.optimizer.state_dict(), f"{self.path_save}/optimizer.pth")
        logger.info(f"Save model done")

    def evaluate(self, **kwargs):
        logger.info("Start evaluate")
        self.model.eval()
        val_loss = 0
        val_iou_score = 0
        for idx, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            image = batch['image'].to(self.device)
            mask = batch['mask'].type(torch.LongTensor).to(self.device)
            out = self.model(image)
            loss = self.loss_fn(out, mask)
            val_loss += loss.item()
            val_iou_score += self.iou_fn(out.transpose(0, 1),
                                         batch['mask'].type(torch.LongTensor).to(self.device))
        return val_loss / len(self.val_loader), val_iou_score / len(self.val_loader)

    def train_one_epoch(self, epoch: int, **kwargs):
        logger.info(f"Training epoch: {epoch}")
        train_loss = 0
        train_iou_score = 0
        self.model.train()
        for idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            image = batch['image'].to(self.device)
            mask = batch['mask'].type(torch.LongTensor).to(self.device)
            output = self.model(image)
            loss = self.loss_fn(output, mask)
            self.optimizer.zero_grad()
            loss.backward()
            if self.arg.debug_mode:
                logger.info(f"Step: {idx} --- Loss: {loss.item()}")

            self.optimizer.step()
            train_loss += loss.item()
            train_iou_score += self.iou_fn(output.transpose(0, 1),
                                           batch['mask'].type(torch.LongTensor).to(self.device))

        return train_loss / len(self.train_loader), train_iou_score / len(self.train_loader)

    def fit(self, **kwargs):
        for epoch in range(self.arg.num_epoch):
            train_loss, train_iou = self.train_one_epoch(epoch)
            val_loss, val_iou = self.evaluate()
            logger.info(f"Epoch: {epoch} --- Train loss: {train_loss} --- Val loss: {val_loss}")
            self.history['train_loss'].append(train_loss)
            self.history['train_iou'].append(train_iou)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_iou)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch)
                plot_loss(self.history, path_save=self.path_save)
                plot_iou_score(self.history, path_save=self.path_save)

