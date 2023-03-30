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
from torch.utils.data import TensorDataset, DataLoader


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


input_path = "/media/emeric/SSD1/emericssd/CDnet2014/dataset/baseline/highway"

batch_size = 3
buffer_size = 9

"""
my_x = np.array([[10*i] for i in range(100)])

tensor_x = torch.Tensor(my_x)

my_dataset = TensorDataset(tensor_x)
my_dataloader = DataLoader(my_dataset, batch_sampler=dataset.StreamSampler(batch_size, buffer_size))


for image in range(buffer_size, buffer_size + 3):
    my_dataloader.batch_sampler.position(image)
    print("image", image)
    for epoch in range(2):
        print("epoch", epoch)
        for batch in my_dataloader:
            print(batch)"""


"""
buffer = dataset.Image_dataset_buffer(input_path, buffer_size)

loader = DataLoader(buffer, batch_size=batch_size,
                    num_workers=1,
                    drop_last=True, pin_memory=True,
                    shuffle=True, persistent_workers=True)
"""



train_dataset = dataset.Image_dataset(input_path)

train_dataloader = DataLoader(train_dataset,
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            batch_sampler=dataset.StreamSampler(batch_size, buffer_size))


for i in range(buffer_size, buffer_size + 3): #len(buffer.image_names)
    print("image", i)
    train_dataloader.batch_sampler.position(i)
    for epoch in range(2):
        print("epoch", epoch)
        for batch in train_dataloader:
            pass
        print(train_dataloader.batch_sampler.batch_list)

#print(list(test_dataset))


