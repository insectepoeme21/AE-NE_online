import os
import time
import stats
import main
import torch
import cv2
import numpy as np
import shutil
import dataset
import train
import utils
from tqdm import tqdm

parser = main.create_parser()
args = parser.parse_args()
args.input_path = "/media/emeric/SSD1/emericssd/CDnet2014/dataset/baseline/highway"
args.results_dir_path = "/media/emeric/SSD1/emericssd"

video_paths = {}

# will train and generate backgrounds and masks on the full dataset
video_paths['train_dataset'] = args.input_path
video_paths['test_dataset'] = args.input_path

video_paths['masks'] = "/media/emeric/SSD1/emericssd/CDnet2014/results/baseline/highway"
video_paths['backgrounds'] = "/media/emeric/SSD1/emericssd/CDnet2014/backgrounds/baseline/highway"
video_paths['models'] = "/media/emeric/SSD1/emericssd/CDnet2014/models/baseline/highway"
video_paths['GT'] = os.path.join(args.input_path,'groundtruth')
spatial_roi_path = os.path.join(args.input_path,"ROI.bmp")
temporal_roi_path = os.path.join(args.input_path, 'temporalROI.txt')
video_name = "highway"

batch_size = 16
buffer_size = 64
lr = 5e-4

device = torch.device("cuda", 0)
model_path = os.path.join(video_paths['models'],'trained_model.pth')

train_dataset = dataset.Image_dataset_buffer(video_paths['train_dataset'], buffer_size)


device = torch.device("cuda", 0)

netBE, netBG = utils.setup_background_models(device, train_dataset.image_height, train_dataset.image_width)

optimizer = torch.optim.Adam([{'params': netBG.parameters()}, {'params': netBE.parameters()}], lr=lr, betas=(0.90, 0.999))

traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            num_workers=4,
                                            drop_last=True, pin_memory=True,
                                            shuffle=True, persistent_workers=True)


for i in range(buffer_size, len(train_dataset.image_names)):
    train_dataset.position(i)
    print(f"image numéro {i}")

    netBE.train()
    netBG.train()

    for epoch in range(2):
        for images in traindataloader:

            images = images.to(device)
            optimizer.zero_grad()
            backgrounds_with_error_prediction = netBG(netBE(images))  # range 0-255
            loss = train.background_loss(args, images, backgrounds_with_error_prediction)
            loss.backward()
            optimizer.step()

        print('[epoch %d] loss: %.6f '% (epoch, loss))
        #print(len(train_dataset), len(traindataloader), list(traindataloader.batch_sampler))
    
    netBE.eval()
    netBG.eval()

    with torch.no_grad():
        test_images = train_dataset[i].unsqueeze(0)
        images = main.compute_background_and_mask_using_trained_model(args, train_dataset, netBE, netBG, test_images, device)

    background = images['backgrounds'][0]
    mask = images['masks'][0]
    cv2.imshow('input', train_dataset.getimage(i))
    cv2.imshow('background', background)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
