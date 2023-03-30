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
args.train_model = True
args.input_path = "/media/emeric/SSD1/emericssd/CDnet2014/dataset/baseline/highway"
args.results_dir_path = "/media/emeric/SSD1/emericssd"

args.unsupervised_mode = True
args.n_iterations = 500
args.background_complexity = False

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


batch_size = 32
device = torch.device("cuda", 0)
model_path = os.path.join(video_paths['models'],'trained_model.pth')

if args.train_model:
    print(f"initialization of train dataset {video_paths['train_dataset']}")
    train_dataset = dataset.Image_dataset(video_paths['train_dataset'])

    netBE, netBG = train.train_dynamic_background_model(args, train_dataset,model_path,batch_size)

print(f"initialization of test dataset {video_paths['test_dataset']}")
test_dataset = dataset.Image_dataset(video_paths['test_dataset'])

if not args.train_model:
    print(f'loading saved models from {model_path}')
    checkpoint = torch.load(model_path)
    encoder_state_dict = checkpoint['encoder_state_dict']
    generator_state_dict = checkpoint['generator_state_dict']
    complexity = checkpoint['complexity']
    netBE, netBG = utils.setup_background_models(device, test_dataset.image_height, test_dataset.image_width, complexity)
    netBE.load_state_dict(encoder_state_dict)
    netBG.load_state_dict(generator_state_dict)
    print('models succesfully loaded')

netBE.eval()
netBG.eval()

with torch.no_grad():

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=4,
                                                drop_last=False, pin_memory=True)

    print(f"generating background and masks for {video_paths['test_dataset']}...")
    for i, test_images in enumerate(tqdm(dataloader)):
            images = main.compute_background_and_mask_using_trained_model(args,test_dataset,netBE, netBG, test_images, device)
            for j in range(test_images.shape[0]):
                index = 1+i*batch_size+j
                cv2.imwrite('%s/background_%06d.jpg' % (video_paths['backgrounds'], index), images['backgrounds'][j])
                cv2.imwrite('%s/bin%06d.png' % (video_paths['masks'], index), images['masks'][j])


statistics = stats.compute_statistics('CDnet', video_name, video_paths['masks'], video_paths['GT'], spatial_roi_path, temporal_roi_path)
print(statistics)
    
#main.compute_dynamic_backgrounds_and_masks(args, video_paths)