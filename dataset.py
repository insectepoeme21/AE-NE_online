
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
    
    def getimage(self, idx):
        image_path = os.path.join(self.dir, self.image_names[idx])
        return cv2.imread(image_path)

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
        image_path = os.path.join(self.dir, image_name)
        opencv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # 0-255 HWC RGB
        np_image = np.asarray(rgb_image)
        np_image = np.transpose(np_image, (2, 0, 1))  # 0-255 CHW RGB
        tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
        return tensor_image
    
    def position(self, position):
        self.pos = position
        if self.pos <= self.buffer_size - 1:
            self.dataset_length = self.pos + 1
        else:
            self.dataset_length = self.buffer_size
            self.start_offset = self.pos - self.buffer_size


class StreamSamplerFIFO(data.sampler.BatchSampler):
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_batch = buffer_size // batch_size

        self.batch_list = self.new_batch_list()
    
    def __iter__(self):
        np.random.shuffle(self.batch_list)
        return iter(self.batch_list)
    
    def __len__(self):
        return self.batch_size * self.n_batch
    
    def new_batch_list(self):
        lst = np.arange(len(self))
        np.random.shuffle(lst)
        lst = lst.reshape((self.n_batch, self.batch_size))
        return lst

    def position(self, pos):
        self.batch_list = self.new_batch_list() + pos - len(self) + 1
        #self.batch_list[0] = np.arange(self.batch_size)*10 + 10
        #np.random.shuffle(self.batch_list[0])


class StreamSamplerRandom(data.sampler.BatchSampler):
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_batch = buffer_size // batch_size

        self.pos = buffer_size
        self.batch_list = self.new_batch_list()
    
    def __iter__(self):
        np.random.shuffle(self.batch_list)
        return iter(self.batch_list)
    
    def __len__(self):
        return self.batch_size * self.n_batch
    
    def new_batch_list(self):
        lst = np.random.choice(self.pos + 1, len(self), False)
        return lst.reshape((self.n_batch, self.batch_size))

    def position(self, pos):
        self.pos = pos
        self.batch_list = self.new_batch_list()


class StreamSamplerRandomWeighted(data.sampler.BatchSampler):
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_batch = buffer_size // batch_size

        self.pos = buffer_size
        self.batch_list = self.new_batch_list()
    
    def __iter__(self):
        np.random.shuffle(self.batch_list)
        return iter(self.batch_list)
    
    def __len__(self):
        return self.batch_size * self.n_batch
    
    def new_batch_list(self):
        proba = np.arange(1, self.pos + 2)
        proba = proba / sum(proba)
        lst = np.random.choice(self.pos + 1, len(self), False, proba)
        return lst.reshape((self.n_batch, self.batch_size))

    def position(self, pos):
        self.pos = pos
        self.batch_list = self.new_batch_list()