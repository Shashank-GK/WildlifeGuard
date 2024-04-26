import math
import argparse
import os
import sys
from pathlib import Path
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import serial
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (check_file, check_img_size, check_imshow,increment_path, non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import telepot
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import time
import serial
import numpy as np

data1 = serial.Serial(
                 'COM5',
                   baudrate = 9600,
                   parity=serial.PARITY_NONE,
                   stopbits=serial.STOPBITS_ONE,
                   bytesize=serial.EIGHTBITS,
                   timeout=1 # must use when using data.readline()
                    )



@torch.no_grad()
def run(weights='fire_smoke.pt',  # model.pt path(s)
        source='1',  # file/dir/URL/glob, 0 for webcam
        data='data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.50,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_conf=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
            
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
       
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if webcam:  
                im0, frame = im0s[i].copy(), dataset.count
                #s += f'{i}: '
            else:
                im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                height, width, _ = im0.shape
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    data1.write(str.encode('F'))
                   
##                    song = MP3("beep.mp3")
##                    pygame.mixer.init()
##                    pygame.mixer.music.load('beep.mp3')
##                    pygame.mixer.music.play()
##                    time.sleep(song.info.length)
##                    pygame.quit()
                    bot = telepot.Bot("7017999737:AAG9v0SJ4MXwUVjSFfo4yOGOcJEgKz3zZcU")
                    bot.sendMessage("1195161922",  str(names[c]+' detected'))
                     
        # Stream results
        im0 = annotator.result()
        #if view_img:
        cv2.imshow('Fire_smoke Detection', im0)
        cv2.imwrite('result.jpg', im0)
        cv2.waitKey(1)

if __name__ == "__main__":
    run()
