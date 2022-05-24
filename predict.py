import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataloader import ImageDataset
from utils.dataloader import MLPDataset
from utils.dataloader import Dataset
from utils.draw import LossVisualizer
import torch.optim as optim
from PIL import Image
from nets.AE import AE
from nets.MLP import MLP
import numpy as np
import visdom
import copy

def predict(prePath, nowPath, txtPath):
    # 将txtPath文件夹下的带隙文件通过网络转换成重构后的声子晶体
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_path = './results/AE/best.pth'
    mlp_path = './results/MLP/best.pth'

    mlp_model = MLP(input_nums=1000, output_nums=700).to(device)
    mlp_model.load_state_dict(torch.load(mlp_path))
    mlp_model.eval()

    ae_model = AE(features_num=700, is_predict=True).to(device)
    ae_model.load_state_dict(torch.load(ae_path))
    ae_model.eval()

    ae_model2 = AE(features_num=700).to(device)
    ae_model2.load_state_dict(torch.load(ae_path))
    ae_model2.eval()

    mlp_dataset = MLPDataset(txtPath)
    mlp_dataloader = DataLoader(mlp_dataset, batch_size=1, shuffle=False)

    ae_dataset = ImageDataset(prePath)
    ae_dataloader = DataLoader(ae_dataset, batch_size=1, shuffle=False)

    for batchidx, x in enumerate(mlp_dataloader):
        x = x.float().to(device)
        x = mlp_model(x)
        x = ae_model(x)
        # x = ae_model.fc2(x)
        # x = ae_model.make_five_dconv(x)
        x = x.view(128, 128)
        x = x.cpu().detach().numpy()
        x *= 255
        im = Image.fromarray(x)
        im = im.convert('L')
        im.save(nowPath + '/' + str(batchidx) + '.png')

    for batchidx, x in enumerate(ae_dataloader):
        x = x.float().to(device)

        x_ = x.view(128, 128)
        x_ = x_.cpu().detach().numpy()
        # x_ = x_ * 255
        im = Image.fromarray(x_)
        im = im.convert('L')
        im.save(nowPath + '/' + str(batchidx) + '_pre.png')

        x, y = ae_model2(x)
        # x = ae_model.fc2(x)
        # x = ae_model.make_five_dconv(x)
        x = x.view(128, 128)
        x = x.cpu().detach().numpy()
        # x *= 255
        im = Image.fromarray(x)
        im = im.convert('L')
        im.save(nowPath + '/' + str(batchidx) + '_after.png')


if __name__ == '__main__':
    prePath = './data/predict_pre/'
    txtPath = './data/predict_txt'
    nowPath = './data/predict_now'
    predict(prePath, nowPath, txtPath)