from src.dataset import CityScapesDataset
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
import torch
from torch import nn

data_set = CityScapesDataset(path_folder='assert/data')
loader = DataLoader(data_set, batch_size=8, shuffle=True)

sample = next(iter(loader))
# print(sample['image'].size())
# print(sample['mask'].size())

# model = Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=8,                      # model output channels (number of classes in your dataset)
# ).double()
# loss_fn = nn.CrossEntropyLoss()
# out = model(sample['image'].type(torch.DoubleTensor))
# print(out.dtype)
# print(sample['mask'].size())
# print(out.size())
# loss = loss_fn(out, sample['mask'].type(torch.LongTensor))
# print(loss)

model = Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=8,                      # model output channels (number of classes in your dataset)
)
loss_fn = nn.CrossEntropyLoss()
out = model(sample['image'])
print(out.dtype)
print(sample['mask'].size())
print(out.size())
loss = loss_fn(out, sample['mask'].type(torch.LongTensor))
print(loss)

