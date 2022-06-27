from segmentation_models_pytorch import Unet
from src.trainer import UnetTrainer
from src.dataset import DataSource
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder', type=str, default='assets/data')
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--is_train', default=True, type= lambda x: x.lower() == 'true')
parser.add_argument('--is_test', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--path_save', type=str, default='assets/model')
parser.add_argument('--resize_mode', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--debug_mode', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--continue_training', type=int, default=False)
parser.add_argument('--encoder_name', type=str, default='resnet34')
parser.add_argument('--encoder_weights', type=str, default='imagenet')

arg = parser.parse_args()
data_source = DataSource.create_datasource(
    path_folder=arg.path_folder,
    resize_mode=arg.resize_mode, # True
    height=arg.height, # 1024
    width=arg.width # 512
)


model = Unet(
    encoder_name=arg.encoder_name,
    encoder_weights=arg.encoder_weights,
    in_channels=3,
    classes=data_source.train_dataset.num_class
)

trainer = UnetTrainer(arg=arg,
                      model=model,
                      data_source=data_source
                      )

trainer.fit()
