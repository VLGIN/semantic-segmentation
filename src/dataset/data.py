import os
import torch
import numpy as np


from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, data):
        super(ImageDataset, self).__init__()
        self.data = data


    def __getitem__(self, index):
        img,smnt = self.data[index]

        def transform(image):
            transform_ops = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            return transform_ops(image)
        return_img = transform(img)
        return_smnt = torch.from_numpy(np.array(smnt))

        return {"image": return_img, "smnt": return_smnt}

    def __len__(self):
        return len(self.data)

