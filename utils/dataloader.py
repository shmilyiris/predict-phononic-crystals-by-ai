import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import glob
import os
import numpy as np

def get_images(dir_path):
    images_list = []
    for img_path in glob.glob(dir_path + "*.jpg"):
        images_list.append(img_path)
    return images_list

class ImageDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.images = get_images(dir_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = torch.from_numpy(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).float()  # 将numpy -> byteTensor -> floatTensor
        return img.reshape(1, img.shape[0], img.shape[1])


# 自定义MLP数据集
class MLPDataset(Dataset):
    def __init__(self, path):
        inputs = []
        outputs = []
        for idx, filename in enumerate(os.listdir(path)):
            if filename.find("input") == 0:
                inputs.append(np.loadtxt(path + '/' + filename))
            else:
                outputs.append(np.loadtxt(path + '/' + filename))
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if idx < len(self.outputs):
            return self.inputs[idx], self.outputs[idx]
        else:
            return self.inputs[idx]