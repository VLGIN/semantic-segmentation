from typing import List
import uvicorn
import torch
<<<<<<< HEAD
import cv2

from fastapi import FastAPI, File, UploadFile, Request, Response, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import dotenv_values
from fastapi.responses import FileResponse
from segmentation_models_pytorch import Unet
from PIL import Image
from torchvision.transforms import functional as torch_f
from torchvision import transforms as T
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from typing import List

CONFIG = dotenv_values("./.env")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class InputImage(BaseModel):
    image: List

@app.on_event('startup')
async def setup():
    global segmentor
    encoder_name = CONFIG['ENCODER_NAME']
    encoder_weights = CONFIG['ENCODER_WEIGHTS']
    in_channels = CONFIG['IN_CHANNELS']
    classes = CONFIG['CLASSES']
    state_dict_path = CONFIG['MODEL_PATH']
    segmentor = Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,\
        in_channels=in_channels, classes=classes)
    state_dict = torch.load(state_dict_path, map_location='cpu')
    segmentor.load_state_dict(state_dict)

map_labels = {
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

mappingrgb = {
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

def transform_image(image):
    image = torch_f.resize(image, size=[int(CONFIG['IMG_HEIGHT']), int(CONFIG['IMG_WIDTH'])], interpolation=torch_f.InterpolationMode.BILINEAR)
    image = torch_f.to_tensor(image)
    output = segmentor(image.unsqueeze(0))
    output = output.argmax(dim=1)
    output = output.squeeze()
    output = class_to_rgb(output)
    transform = T.ToPILImage()
    output = transform(output).convert('RGB')
    return output

def class_to_rgb(mask):
    '''
    This function maps the classification index ids into the rgb.
    For example after the argmax from the network, you want to find what class
    a given pixel belongs too. This does that but just changes the color
    so that we can compare it directly to the rgb groundtruth label.
    '''
    mask2class = dict((v, k) for k, v in map_labels.items())
    rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mask2class:
        rgbimg[0][mask == k] = mappingrgb[mask2class[k]][0]
        rgbimg[1][mask == k] = mappingrgb[mask2class[k]][1]
        rgbimg[2][mask == k] = mappingrgb[mask2class[k]][2]
    return rgbimg

def split_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        cv2.imwrite("static/frame%d.jpg" % count, image)     # save frame as JPEG file
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1
    vidcap.release()
    return count

@app.on_event('startup')
async def setup():
    global segmentor
    encoder_name = CONFIG['ENCODER_NAME']
    encoder_weights = CONFIG['ENCODER_WEIGHT']
    in_channels = int(CONFIG['IN_CHANNELS'])
    classes = int(CONFIG['CLASSES'])
    state_dict_path = CONFIG['MODEL_PATH']
    segmentor = Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,\
        in_channels=in_channels, classes=classes)
    state_dict = torch.load(state_dict_path, map_location='cpu')
    segmentor.load_state_dict(state_dict)

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    open_part = file.filename.split(".")[-1]
    content = await file.read()
    with open("static/tmp." + open_part, "wb") as f:
        f.write(content)
    image = Image.open("static/tmp." + open_part).convert("RGB")
    output = transform_image(image)
    output.save("static/output.png")
    rs = {}
    rs["original_image"] = "/get_file/tmp." + open_part
    rs["result"] = "/get_file/output.png"
    return rs

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    open_part = file.filename.split(".")[-1]
    content = await file.read()
    with open("static/video_temp."+open_part, "wb") as f:
        f.write(content)
    num_frames = split_video("static/video_temp."+open_part)
    output = []
    for i in range(num_frames):
        image = Image.open(f"static/frame{i}.jpg").convert("RGB")
        output = transform_image(image)
        output.save(f"static/video_result{i}.png")
    writer = cv2.VideoWriter("static/outputvideo.mp4",cv2.VideoWriter_fourcc(*"MP4V"),30,(512,512))
    for i in range(num_frames):
        image = cv2.imread(f"static/video_result{i}.png", cv2.IMREAD_COLOR)
        writer.write(image)
    writer.release()
    rs = {}
    rs["original_video"] = "/get_file/video_temp."+open_part
    rs["result"] = "/get_file/outputvideo.mp4"
    return rs

<<<<<<< HEAD
@app.get("/get_file/{file_name}")
def get_file(file_name: str):
    file_path = "./static/" + file_name
    return FileResponse(path=file_path)
=======
@app.post("/segment/image")
def segment(inputImage: InputImage):
    # preprocess input
    image = inputImage.image_path
    # read image
    output = segmentor(image)
    
@app.post("/segment/video")
def segment(inputImage: InputImage):
    # preprocess input
    video_path = inputImage.image_path
    # read video and split video
    video = video_path
    result_list = []
    for item in video:
        result_list.append(segmentor(item).detach().cpu().numpy())
    # concate frames to video
    return video

>>>>>>> e630f9b (resolve conflict)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)