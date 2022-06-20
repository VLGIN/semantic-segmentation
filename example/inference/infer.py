from abc import ABC
from ast import Num
import os
from random import shuffle
import sys

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.utils import functional as F
from torchvision.transforms import functional as torch_f
import torchvision.transforms as T
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import optim
from PIL import Image
from src.dataset import DataSource
import matplotlib.pyplot as plt
import logging
from segmentation_models_pytorch.utils.metrics import IoU
logger = logging.getLogger(__name__)


class Infer():
    def __init__(self, path, model, dataset: DataSource):
        self.path = path
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.dataset = dataset.test_dataset
        self.iou_fn = IoU(threshold=0.5)
        self.mappingrgb = {
            0: (255, 0, 0),  # unlabeled
            1: (255, 0, 0),  # ego vehicle
            2: (255, 0, 0),  # rect border
            3: (255, 0, 0),  # out of roi
            4: (255, 0, 0),  # static
            5: (255, 0, 0),  # dynamic
            6: (255, 0, 0),  # ground
            7: (0, 255, 0),  # road
            8: (0, 255, 0),  # sidewalk
            9: (0, 255, 0),  # parking
            10: (0, 255, 0),  # rail track
            11: (0, 0, 255),  # building
            12: (0, 0, 255),  # wall
            13: (0, 0, 255),  # fence
            14: (0, 0, 255),  # guard rail
            15: (0, 0, 255),  # bridge
            16: (0, 0, 255),  # tunnel
            17: (255, 255, 0),  # pole
            18: (255, 255, 0),  # polegroup
            19: (255, 255, 0),  # traffic light
            20: (255, 255, 0),  # traffic sign
            21: (255, 0, 255),  # vegetation
            22: (255, 0, 255),  # terrain
            23: (0, 255, 255),  # sky
            24: (255, 102, 26),  # person
            25: (255, 102, 26),  # rider
            26: (163, 41, 122),  # car
            27: (163, 41, 122),  # truck
            28: (163, 41, 122),  # bus
            29: (163, 41, 122),  # caravan
            30: (163, 41, 122),  # trailer
            31: (163, 41, 122),  # train
            32: (163, 41, 122),  # motorcycle
            33: (163, 41, 122),  # bicycle
            -1: (163, 41, 122)  # licenseplate
        }
        # self.data_loader = DataLoader(dataset, batch_size=self.arg.batch_size, shuffle=True, num_workers=2)
    def class_to_rgb(self, mask):
        '''
        This function maps the classification index ids into the rgb.
        For example after the argmax from the network, you want to find what class
        a given pixel belongs too. This does that but just changes the color
        so that we can compare it directly to the rgb groundtruth label.
        '''
        mask2class = dict((v, k) for k, v in self.dataset.map_labels.items())
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mask2class:
            rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
            rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
            rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
        return rgbimg

    def infer(self, path_img):
        image = Image.open(path_img).convert("RGB")
        image = torch_f.resize(image, size=[512, 1024], interpolation=torch_f.InterpolationMode.BILINEAR)
        image = torch_f.to_tensor(image)
        image = image.unsqueeze(0)
        print("Input model: ", image.shape)
        out = self.model(image)   
        print("Output model: ", out.shape)
        a = out.argmax(dim=1)
        print("Convert model: ", a.shape)
        b = a.squeeze()
        print(b.shape)
        b = self.class_to_rgb(b)
        transform = T.ToPILImage()
        img = transform(b).convert("RGB")
        img.show()

    # def test_mask(self):
    #     image = Image.open(path_img).convert("RGB")
    #     image = torch_f.resize(image, size=[512, 1024], interpolation=torch_f.InterpolationMode.BILINEAR)
    #     image = torch_f.to_tensor(image)
    #     image = image.unsqueeze(0)
    #     print("Input model: ", image.shape)
    #     out = self.model(image)   
    #     print("Output model: ", out.shape)
    #     a = out.argmax(dim=1)
    #     print("Convert model: ", a.shape)
    #     b = a.squeeze()
    #     print(b.shape)
    #     b = self.class_to_rgb(b)
    #     transform = T.ToPILImage()
    #     img = transform(b).convert("RGB")
    #     img.show()

        

    def make_loader(self, batch_size):
        return DataLoader(self.dataset, batch_size, shuffle=True, num_workers=2)

    def cal_iou(self):
        logger.info("Start calculate IOU metrics")
        self.model.eval()
        test_loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=2)

        print(len(test_loader))
        batch = next(iter(test_loader))
        print(batch["image"].shape)
        print(batch["mask"].shape)
 
        image = batch['image'].to(self.device) # 1 3 512 1024
        mask = batch['mask'].type(torch.LongTensor).to(self.device) #1 512 1024

        out = self.model(image) #1 8 512 1024
        print(self.iou_fn.forward(out, mask))
        # print(out.shape)
        # print(out)
        # print(out)
        # print(max(out))
        # print(min(out))
        # a = out.argmax(dim=1)
        
        # print(a.shape)
        # print(a)
        # out = out.argmax(dim=1)
        # out = out.type(torch.LongTensor).to(self.device)
        # print(F.iou(out, mask, 0.5))
        # res = smp.metrics.get_stats(out, mask, mode="multilabel")
        
        # print(res)

        # iou_score = 0
        # for idx, batch in enumerate(test_loader):
        #     image = batch['image'].to(self.device)
        #     mask = batch['mask'].type(torch.LongTensor).to(self.device)
        #     out = self.model(image)

        #     iou_score += F.iou(out, mask)

        # logger.info("IOU score of test dataset is ", iou_score/len(test_loader))
        
        
"""
Input model:  torch.Size([1, 3, 512, 1024])
Output model:  torch.Size([1, 8, 512, 1024])
"""
