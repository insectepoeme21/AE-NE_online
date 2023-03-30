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
import natsort

class GetImage:
    def __init__(self, video_paths):
        self.images_names = {}
        for category in ['masks', 'backgrounds', 'GT']:
            self.images_names[category] = self.get_name_list(video_paths[category])

    def get_name_list(self, directory):
         names = [item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))]
         return natsort.natsorted(names)

    def get_image(self, category, idx):
        image_path = os.path.join(video_paths[category], self.images_names[category][idx])
        #opencv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
        #rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # 0-255 HWC RGB
        return cv2.imread(image_path) #rgb_image

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
spatial_roi = cv2.imread(spatial_roi_path)
video_name = "highway"

stored_images = GetImage(video_paths)

batch_size = 16
buffer_size = 128
lr = 5e-4

device = torch.device("cuda", 0)
model_path = os.path.join(video_paths['models'],'trained_model.pth')

train_dataset = dataset.Image_dataset(video_paths['train_dataset'])


device = torch.device("cuda", 0)

netBE, netBG = utils.setup_background_models(device, train_dataset.image_height, train_dataset.image_width)

optimizer = torch.optim.Adam([{'params': netBG.parameters()}, {'params': netBE.parameters()}], lr=lr, betas=(0.90, 0.999))

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                            num_workers=4,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            batch_sampler=dataset.StreamSampler(batch_size, buffer_size))


for i in range(buffer_size, len(train_dataset.image_names)):
    train_dataloader.batch_sampler.position(i)
    print(f"image {i}")

    netBE.train()
    netBG.train()

    for epoch in range(1):
        for images in train_dataloader:

            images = images.to(device)
            optimizer.zero_grad()
            backgrounds_with_error_prediction = netBG(netBE(images))  # range 0-255
            loss = train.background_loss(args, images, backgrounds_with_error_prediction)
            loss.backward()
            optimizer.step()

        print('[epoch %d] loss: %.6f '% (epoch, loss))
        print(train_dataloader.batch_sampler.batch_list)
    
    netBE.eval()
    netBG.eval()

    with torch.no_grad():
        test_images = train_dataset[i].unsqueeze(0)
        images = main.compute_background_and_mask_using_trained_model(args, train_dataset, netBE, netBG, test_images, device)

    background = images['backgrounds'][0]
    mask = images['masks'][0]
    aene_background = stored_images.get_image('backgrounds', i)
    aene_mask = stored_images.get_image('masks', i)
    ground_truth = stored_images.get_image('GT', i)

    #tp, tn, fp, fn = stats.compute_confusion_matrix("CDnet", mask, ground_truth, True, spatial_roi)
    #print(tp, tn, fp, fn)

    cv2.imshow('input', train_dataset.getimage(i))
    cv2.imshow('background', background)
    cv2.imshow('mask', mask)
    cv2.imshow('ae-ne background', aene_background)
    cv2.imshow('ae-ne mask', aene_mask)
    cv2.imshow('ground truth', ground_truth)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

