from segmentation_models_pytorch import Unet
from src.dataset import DataSource
from torch.utils.data import DataLoader
from segmentation_models_pytorch.utils.metrics import IoU
import torch
from tqdm import tqdm

data_source = DataSource.create_datasource(
    path_folder='assets/data',
    resize_mode=True,
    height=1024,
    width=512
)

model = Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=data_source.train_dataset.num_class
)
model.load_state_dict(torch.load('assets/models/model.pth', map_location='cpu'))
model.eval()

iou_fn = IoU()
test_loader = DataLoader(dataset=data_source.test_dataset, batch_size=8, shuffle=False, num_workers=2)
score = 0

for idx, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
    outs = model(sample['image'])
    try:
        score += iou_fn(outs.transpose(0,1), sample['mask'].type(torch.LongTensor)).items()
    except:
        print(outs.size())
        print(sample['mask'].size())

print(f"Mean IOU: {score / len(data_source.test_dataset)}")