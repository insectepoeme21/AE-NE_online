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
from torch.utils.data.sampler import BatchSampler
import random
from torchvision import datasets

def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

class StreamSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.len_dataset = len(dataset)
    
    def __iter__(self):
        combined = [batch.tolist() for batch in self.dataset]
        #random.shuffle(combined)
        return iter([list(range(i*self.batch_size, (i+1)*self.batch_size)) for i in range(len(self)-1)])
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

input_path = "/media/emeric/SSD1/emericssd/CDnet2014/dataset/baseline/highway"

batch_size = 3
buffer_size = 12

test_dataset = dataset.Image_dataset(input_path)

traindataloader = torch.utils.data.DataLoader(test_dataset,
                                                num_workers=4,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                batch_sampler=StreamSampler(test_dataset, batch_size))



buffer = dataset.Image_dataset_buffer(input_path, buffer_size)

loader = torch.utils.data.DataLoader(buffer, batch_size=batch_size,
                                    num_workers=1,
                                    drop_last=True, pin_memory=True,
                                    shuffle=True, persistent_workers=True)

for i in range(batch_size, 24): #len(buffer.image_names)
    for epoch in range(1):
        for j, batch in enumerate(loader):
            pass
        print(len(buffer), len(loader), list(loader.batch_sampler))
    buffer.position(i)

#print(list(test_dataset))

