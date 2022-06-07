#  import argparse
#  import os
#  import numpy as np
#  import sys
#  import torch

#  from src.data import ImageDataset
#  from torch.utils.data import DataLoader
#  from segmentation_models_pytorch import DeepLabV3Plus
#  from torch.nn import MSELoss, BCELoss
#  from torch.optim import AdamW
#  from loguru import logger
#  from decouple import config

#  def valid(model, dataloader, loss_func, device):
    #  model.eval()
    #  loss_valid = []
    #  for i, batch in enumerate(dataloader):
        #  image = torch.clone(batch["image"]).type(torch.DoubleTensor).to(device)
        #  mask = torch.clone(batch["mask"]).type(torch.DoubleTensor).to(device)

        #  output_mask = model(image)
        #  loss = loss_func( output_mask.squeeze(), mask)
        #  loss_valid.append(loss.detach().cpu().item())
    #  return np.mean(loss_valid)

#  def train():
    #  parser = argparse.ArgumentParser()
    #  # Add something right there
    #  parser.add_argument("--train_image_path", default="segment/train/image",
                        #  help="Path to train image folder")
    #  parser.add_argument("--train_mask_path", default="segment/train/mask",
                        #  help="Path to train mask folder")
    #  parser.add_argument("--valid_image_path", default="segment/valid/image",
                        #  help="Path to valid image folder")
    #  parser.add_argument("--valid_mask_path", default="segment/valid/mask",
                        #  help="Path to valid mask folder")
    #  parser.add_argument("--learning_rate", type=float, default=1e-4,
                        #  help="Learning rate")
    #  parser.add_argument("--batch_size", type=int, default=16,
                        #  help="Batch size")
    #  parser.add_argument("--epoch", type=int, default=50,
                        #  help="Number of epochs")
    #  parser.add_argument("--from_checkpoint", action='store_true')
    #  parser.add_argument("--checkpoint", default="model/13.pth",
                        #  help="Path to checkpoint")

    #  args = parser.parse_args()
    #  model_path = config("MODEL_PATH")
    #  if not os.path.exists(model_path):
        #  os.mkdir(model_path)

    #  if args.from_checkpoint == True:
        #  model = torch.load(args.checkpoint).double()
        #  start_epoch = int(args.checkpoint.split("/")[1].split(".")[0].strip()) + 1
    #  else:
        #  start_epoch = 0
        #  model = DeepLabV3Plus(encoder_weights="imagenet", activation="sigmoid")
        #  model = model.double()

    #  train_dataset = ImageDataset(image_path=args.train_image_path, mask_path=args.train_mask_path)
    #  train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    #  valid_dataset = ImageDataset(image_path=args.valid_image_path, mask_path=args.valid_mask_path)
    #  valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    #  loss_func = BCELoss(reduction="mean")
    #  optimizer = AdamW(lr=args.learning_rate, params = model.parameters())
    #  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #  model.to(device)

    #  for epoch in range(start_epoch, args.epoch):
        #  logger.info(f"Training epoch {epoch}")
        #  model.train()
        #  for i, batch in enumerate(train_dataloader):
            #  optimizer.zero_grad()
            #  images = torch.clone(batch["image"]).type(torch.DoubleTensor).to(device)
            #  masks = torch.clone(batch["mask"]).type(torch.DoubleTensor).to(device)

            #  output = model(images)
            #  loss = loss_func(output.squeeze(), masks)
            #  loss.backward()
            #  optimizer.step()

            #  if i == len(train_dataloader) or i % 10 == 0:
                #  logger.info(f"Batch {i}/{len(train_dataloader)}: Loss {loss.item()}")

        #  valid_loss = valid(model, valid_dataloader, loss_func, device)
        #  logger.info(f"Epoch {epoch}: Valid Loss {valid_loss}")
        #  torch.save(model, os.path.join(model_path, str(epoch)+".pth"))

#  if __name__ == "__main__":
    #  log_file = config("LOG_FILE")
    #  logger.remove()
    #  logger.add(sys.stdout, level='INFO')
    #  logger.add(log_file, level='INFO')
    #  train()
import sys
import os
import argparse


from src.dataset.datasource import DataSource
from src.trainer.trainer_seg import TrainerSeg
from segmentation_models_pytorch import DeepLabV3Plus
from loguru import logger
from decouple import config

if __name__ == "__main__":
    log_file = config("LOG_FILE")
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file, level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_from_checkpoint", action='store_true',
                        help="Train from checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to checkpoint of model.")
    parser.add_argument("--is_train", action='store_true',
                        help="Train model.")
    parser.add_argument("--is_test", action='store_true',
                        help="Test model.")
    parser.add_argument("--model_path", type=str, default="model/",
                        help="Folder to save model checkpoints.")
    parser.add_argument("--report_batch", type=int, default=10,
                        help="Number of batch for logging.")
    parser.add_argument("--epoch", type=int, default=50,
                        help="Number of train epoch.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for optimizer.")
    parser.add_argument("--data_folder", type=str, default="segment/",
                        help="Path to data folder.")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Num workers of dataloader.")

    args = parser.parse_args()

    assert os.path.isdir(args.data_folder), "Invalid data folder"
    datasource = DataSource.create_datasource_from_path(path_to_folder_data=args.data_folder)

    trainer = TrainerSeg(args, model=DeepLabV3Plus(encoder_weights='imagenet', activation='sigmoid'), datasource=datasource)

    trainer.fit()
