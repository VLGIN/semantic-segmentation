import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as torch_f
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CityScapesDataset(Dataset):
    def __init__(self, path_folder: str,
                 split: str = 'train',
                 mode: str = 'fine',
                 resize_mode: bool = False,
                 width: int = 256,
                 height: int = 256):
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dirs = os.path.join(path_folder, 'leftImg8bit', split)
        self.target_dirs = os.path.join(path_folder, self.mode, split)
        self.resize_mode = resize_mode
        if self.resize_mode:
            logger.info("Turn on resize mode")
        else:
            logger.info("Turn of resize model")
        self.width = width
        self.height = height
        self.image_paths, self.target_paths = [], []
        self.map_labels = {
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 1,  # sidewalk
            9: 1,  # parking
            10: 1,  # rail track
            11: 2,  # building
            12: 2,  # wall
            13: 2,  # fence
            14: 2,  # guard rail
            15: 2,  # bridge
            16: 2,  # tunnel
            17: 3,  # pole
            18: 3,  # polegroup
            19: 3,  # traffic light
            20: 3,  # traffic sign
            21: 4,  # vegetation
            22: 4,  # terrain
            23: 5,  # sky
            24: 6,  # person
            25: 6,  # rider
            26: 7,  # car
            27: 7,  # truck
            28: 7,  # bus
            29: 7,  # caravan
            30: 7,  # trailer
            31: 7,  # train
            32: 7,  # motorcycle
            33: 7,  # bicycle
            -1: 7  # licenseplate
        }
        self.num_class = len(set(self.map_labels.values()))
        logger.info(f"Start loading path for {split}")
        for city in os.listdir(self.images_dirs):
            img_dir = os.path.join(self.images_dirs, city)
            target_dir = os.path.join(self.target_dirs, city)
            for file_name in os.listdir(img_dir):
                self.image_paths.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], '{}_labelIds.png'.format(self.mode))
                self.target_paths.append(os.path.join(target_dir, target_name))
        logger.info(f"Total image: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def mask_to_class(self, mask):
        masking = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for value in self.map_labels:
            masking[mask == value] = self.map_labels[value]
        return masking

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = Image.open(self.target_paths[idx]).convert('L')

        if self.resize_mode:
            image = torch_f.resize(image, size=[self.width, self.height], interpolation=torch_f.InterpolationMode.BILINEAR)
            target = torch_f.resize(target, size=[self.width, self.height], interpolation=torch_f.InterpolationMode.NEAREST)

        target = torch.from_numpy(np.array(target, dtype=np.uint8))
        image = torch_f.to_tensor(image)

        target_mask = self.mask_to_class(target)
        return {'image': image, 'mask': target_mask}



