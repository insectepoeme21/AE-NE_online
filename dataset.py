
from __future__ import print_function

import os
import torch
import torch.utils.data as data
import natsort
import cv2
import numpy as np


class Image_dataset(data.Dataset):

    def __init__(self, dataset_path):

        if os.path.exists(os.path.join(dataset_path, 'input')):
            self.dir = os.path.join(dataset_path, 'input')
        else:
            self.dir = dataset_path

        image_names = [item for item in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, item))]
        image_names = natsort.natsorted(image_names)

        self.dataset_length = len(image_names)
        self.image_names = image_names

        first_image = self[0]
        nc_input, self.image_height, self.image_width = first_image.shape
        assert nc_input == 3 , f" input image with {nc_input} channels detected, input images should have 3 channels,"

        print(f'dataset initialized  w = {self.image_width},h = {self.image_height} number of frames {self.dataset_length}')

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        image_path = os.path.join(self.dir, self.image_names[idx])
        opencv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # 0-255 HWC RGB
        np_image = np.asarray(rgb_image)
        np_image = np.transpose(np_image, (2, 0, 1))  # 0-255 CHW RGB
        tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
        return tensor_image

class Image_dataset_buffer(data.Dataset):

    def __init__(self,dataset_path, buffer_size):

        if os.path.exists(os.path.join(dataset_path, 'input')):
            self.dir = os.path.join(dataset_path, 'input')
        else:
            self.dir = dataset_path
        
        image_names = [item for item in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, item))]
        image_names = natsort.natsorted(image_names)

        self.image_names = image_names
        self
        self.buffer_size = buffer_size
        self.dataset_length = 1
        self.start_offset = 0
        self.pos = 0

        first_image = self[0]
        nc_input, self.image_height, self.image_width = first_image.shape
        assert nc_input == 3 , f" input image with {nc_input} channels detected, input images should have 3 channels,"

        print(f'dataset initialized  w = {self.image_width},h = {self.image_height} number of frames {self.dataset_length}')

    def __len__(self):
        return self.dataset_length
    
    def getimage(self, idx):
        image_path = os.path.join(self.dir, self.image_names[idx])
        opencv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # 0-255 HWC RGB
        return rgb_image

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        print(image_name, self.pos)
        image_path = os.path.join(self.dir, image_name)
        opencv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # 0-255 HWC RGB
        np_image = np.asarray(rgb_image)
        np_image = np.transpose(np_image, (2, 0, 1))  # 0-255 CHW RGB
        tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
        return tensor_image
    
    def position(self, position):
        print("position ", self.pos)
        self.pos = position
        if self.pos <= self.buffer_size - 1:
            self.dataset_length = self.pos + 1
        else:
            self.dataset_length = self.buffer_size
            self.start_offset = self.pos - self.buffer_size