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
parser.add_argument('--model_pretrain', type=str, default='imagenet')

arg = parser.parse_args()
data_source = DataSource.create_datasource(
    path_folder=arg.path_folder,
    resize_mode=arg.resize_mode,
    height=arg.height,
    width=arg.width
)

model = Unet(
    encoder_name="resnet34",
    encoder_weights=arg.model_pretrain,
    in_channels=3,
    classes=data_source.train_dataset.num_class
)

trainer = UnetTrainer(arg=arg,
                      model=model,
                      data_source=data_source
                      )

trainer.fit()
