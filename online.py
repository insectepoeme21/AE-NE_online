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
import pandas as pd
import matplotlib.pyplot as plt

class GetImage:
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.images_names = {}
        for category in ['masks', 'backgrounds', 'GT']:
            self.images_names[category] = self.get_name_list(video_paths[category])

    def get_name_list(self, directory):
         names = [item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))]
         return natsort.natsorted(names)

    def get_image(self, category, idx):
        image_path = os.path.join(self.video_paths[category], self.images_names[category][idx])
        return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def f_mesure(tp, fp, fn):
    return tp / (tp + 0.5 * (fp + fn))

def create_directories(root_path):
    rootpaths = {}
    rootpaths['masks'] = os.path.join(root_path, 'results')
    rootpaths['backgrounds'] = os.path.join(root_path, 'backgrounds')
    rootpaths['models'] = os.path.join(root_path, 'models')

    for (k, path) in rootpaths.items():
        if not os.path.exists(path):
            os.mkdir(path)

def plot_hist(vect):
    vect = (vect + 1)*50
    vect = vect.astype(int)
    img = np.zeros((256,256,3), np.uint8)
    for i in range(16):
        img[256-vect[i]:,i*16:(i+1)*16,:] = 255
    return img

def online_loop(args, video_paths):
    save = True
    show = False

    data = {'image': [],
        'fm_aene': [],
        'fm_online': [],
        'loss': []}

    aene_images = GetImage(video_paths)

    batch_size = 32
    buffer_size = 128
    lr = 5e-4  # 5e-4
    n_epoch = 1

    train_dataset = dataset.Image_dataset(video_paths['train_dataset'])
    device = torch.device("cuda", 0)
    netBE, netBG = utils.setup_background_models(device, train_dataset.image_height, train_dataset.image_width)
    optimizer = torch.optim.Adam([{'params': netBG.parameters()}, {'params': netBE.parameters()}], lr=lr, betas=(0.90, 0.999))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                num_workers=4,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                batch_sampler=dataset.StreamSamplerRandom(batch_size, buffer_size))
    
    temporal_roi_path = video_paths['temporal_roi']
    if temporal_roi_path != None:
        assert os.path.isfile(temporal_roi_path), f"error, no temporal roi at {temporal_roi_path}"
        f = open(temporal_roi_path, "r")
        start_idx, end_idx = f.readline().split()
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        f.close()
    
    if save:
        for key in ['online_masks', 'online_backgrounds', 'online_models']:
            if os.path.exists(video_paths[key]):
                shutil.rmtree(video_paths[key])
            os.mkdir(video_paths[key])

    for idx_image in range(buffer_size-1, len(train_dataset.image_names)):
        train_dataloader.batch_sampler.position(idx_image)

        netBE.train()
        netBG.train()

        for epoch in range(n_epoch):
            for images in train_dataloader:
                images = images.to(device)
                optimizer.zero_grad()
                backgrounds_with_error_prediction = netBG(netBE(images))  # range 0-255
                loss = train.background_loss(args, images, backgrounds_with_error_prediction)
                loss.backward()
                optimizer.step()

            #print('[epoch %d] loss: %.6f '% (epoch, loss))
            #print(train_dataloader.batch_sampler.batch_list)
        
        netBE.eval()
        netBG.eval()

        with torch.no_grad(): 
            test_image = train_dataset[idx_image].unsqueeze(0)
            images = main.compute_background_and_mask_using_trained_model(args, train_dataset, netBE, netBG, test_image, device)

        background = images['backgrounds'][0]
        mask = images['masks'][0]
        encoded = images['encoded'][0]
        aene_background = aene_images.get_image('backgrounds', idx_image)
        aene_mask = aene_images.get_image('masks', idx_image)
        ground_truth = aene_images.get_image('GT', idx_image)
        roi = cv2.imread(video_paths['spatial_roi'], cv2.IMREAD_UNCHANGED)
        roi = np.asarray(roi)
        roi_mask = (roi == 255)
    
        tp, tn, fp, fn = stats.compute_confusion_matrix("CDnet", mask, ground_truth, True, roi_mask)
        fm_online = f_mesure(tp, fp, fn)
        tp, tn, fp, fn = stats.compute_confusion_matrix("CDnet", aene_mask, ground_truth, True, roi_mask)
        fm_aene = f_mesure(tp, fp, fn)
        print(f"image {idx_image+1}, loss {loss.item():.4f}, online {fm_online:.4f}, aene {fm_aene:.4f}, ratio {fm_online / fm_aene:.4f}")

        if save:
            data['image'].append(idx_image+1)
            data['fm_online'].append(fm_online)
            data['fm_aene'].append(fm_aene)
            data['loss'].append(loss.item())

            cv2.imwrite('%s/background_%06d.jpg' % (video_paths['online_backgrounds'], idx_image+1), background)
            cv2.imwrite('%s/bin%06d.png' % (video_paths['online_masks'], idx_image+1), mask)

        if show:
            cv2.imshow('input', train_dataset.getimage(idx_image))
            cv2.imshow('background', background)
            cv2.imshow('mask', mask)
            cv2.imshow('ae-ne background', aene_background)
            cv2.imshow('ae-ne mask', aene_mask)
            cv2.imshow('ground truth', ground_truth)
            cv2.imshow('latent_space', plot_hist(encoded.cpu().detach().numpy()))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if save:
        df = pd.DataFrame.from_dict(data)
        df.to_csv(os.path.join(video_paths['online_models'], 'perf.csv'))

        model_path = os.path.join(video_paths['online_models'],'trained_model.pth')
        torch.save({'encoder_state_dict': netBE.state_dict(),
                    'generator_state_dict': netBG.state_dict()}, model_path)
    

if __name__ == "__main__":

    parser = main.create_parser()
    args = parser.parse_args()
    input_path = "/media/emeric/SSD1/emericssd/CDnet2014/dataset/baseline/highway"

    video_paths = {}

    video_paths['train_dataset'] = input_path
    video_paths['test_dataset'] = input_path

    video_paths['masks'] = input_path.replace("dataset", "results")
    video_paths['backgrounds'] = input_path.replace("dataset", "backgrounds")
    video_paths['models'] = input_path.replace("dataset", "models")
    video_paths['GT'] = os.path.join(input_path,'groundtruth')

    video_paths['spatial_roi'] = os.path.join(input_path,"ROI.bmp")
    video_paths['temporal_roi'] = os.path.join(input_path, 'temporalROI.txt') 
    #spatial_roi = cv2.imread(video_paths['spatial_roi'])

    video_paths['online_masks'] = video_paths['masks'].replace("CDnet2014", "online_results")
    video_paths['online_backgrounds'] = video_paths['backgrounds'].replace("CDnet2014", "online_results")
    video_paths['online_models'] = video_paths['models'].replace("CDnet2014", "online_results")

    online_loop(args, video_paths)

    statistics, lst1 = stats.compute_statistics('CDnet', "video_name", video_paths['masks'], video_paths['GT'], video_paths['spatial_roi'], video_paths['temporal_roi'])
    print(statistics)
    statistics, lst2 = stats.compute_statistics('CDnet', "video_name", video_paths['online_masks'], video_paths['GT'], video_paths['spatial_roi'], video_paths['temporal_roi'])
    print(statistics)

    print(len(lst1), len(lst2))
    print(np.nanmean(lst1), np.nanmean(lst2))

    plt.plot(lst1)
    plt.plot(lst2)
    plt.show()

