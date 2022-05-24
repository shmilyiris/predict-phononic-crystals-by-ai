import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataloader import ImageDataset
from utils.dataloader import MLPDataset
from utils.dataloader import Dataset
from utils.draw import LossVisualizer
import torch.optim as optim
from nets.AE import AE
from nets.MLP import MLP
import numpy as np
import visdom
import copy
import os



def train_ae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Image_path = "data\\figures_simplify\\"

    visualize = False
    epochs = 1000
    lr0 = 1e-3
    train_ratio = 0.8
    batch_size = 16
    features_num = 700
    model = AE(features_num=features_num).to(device)

    image_set = ImageDataset(Image_path)
    train_size = int(len(image_set) * train_ratio)
    test_size = int(len(image_set)) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(image_set, [train_size, test_size])
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr0)

    if visualize:
        viz = visdom.Visdom()

    # 全局训练参数
    best_loss = 1e9
    best_model_weights = copy.deepcopy(model.state_dict())
    train_loss_list = []
    test_loss_list = []
    cnt_no_increasing = 0

    for epoch in range(epochs):
        model.train()
        flag = False    # flag为true则本轮best_loss被更新
        accumulate_train_loss, accumulate_test_loss = 0, 0
        for batchidx, input in enumerate(train):
            input = input.to(device)
            output, features = model(input)
            loss = criterion(output, input)
            accumulate_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batchidx, x in enumerate(test):
                x = x.to(device)
                y, _ = model(x)
                loss = criterion(y, x)
                accumulate_test_loss += loss.item()

        # 记录loss
        avg_train_loss = accumulate_train_loss / len(train)
        avg_test_loss = accumulate_test_loss / len(test)

        print("{} / {} train_loss: {:.6f}".format(epoch, epochs, avg_train_loss))
        print("{} / {} test_loss : {:.6f}".format(epoch, epochs, avg_test_loss))
        train_loss_list.append(avg_train_loss)
        test_loss_list.append(avg_test_loss)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            flag = True

        if flag == False and epoch > 100:
            # 100轮未得到best_loss连续3轮则结束训练
            cnt_no_increasing += 1
            if cnt_no_increasing > 3:
                break
        else:
            cnt_no_increasing = 0

        if epoch % 50 == 0:
            # 每50轮保存权值数据
            torch.save(best_model_weights, './results/AE/former_{}_rounds.pth'.format(int(epoch)))

        # 抽样出一个batch进行可视化
        x = next(iter(test))
        x_hat, _ = model(x.to(device))
        if visualize:
            viz.images(x, nrow=4, win='x', opts=dict(title='x'))
            viz.images(x_hat, nrow=4, win='x', opts=dict(title='x'))

    # 保存文件和权重
    np.savetxt('./log/AE/train_loss_list.txt', np.array(train_loss_list))
    np.savetxt('./log/AE/valid_loss_list.txt', np.array(test_loss_list))

    # loss可视化
    visualizer = LossVisualizer(train_loss_list, test_loss_list, "./results/AE/AE_loss.jpg")
    visualizer.draw()

    # 存储最终权值文件
    torch.save(best_model_weights, './results/AE/best.pth')



def train_mlp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数
    epochs = 1000
    lr0 = 1e-3
    train_ratio = 0.8
    batch_size = 16
    features_num = 700
    # input_nums: dimension of band-gap bitmap
    # output_nums: dimension of feature nums
    model = MLP(input_nums=1000, output_nums=features_num).to(device)

    dataset = MLPDataset('./data/mlp_train')
    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) - train_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr0)

    # 全局训练参数
    best_loss = 1e9
    best_model_weights = copy.deepcopy(model.state_dict())
    train_loss_list = []
    test_loss_list = []
    cnt_no_increasing = 0

    for epoch in range(epochs):
        model.train()
        flag = False    # flag为true则本轮best_loss被更新
        accumulate_train_loss, accumulate_test_loss = 0, 0
        for batchidx, [input, output_] in enumerate(train):
            input = input.float().to(device)
            output_ = output_.float().to(device)
            output = model(input)
            loss = criterion(output, output_)
            accumulate_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batchidx, [x, y_] in enumerate(test):
                x = x.float().to(device)
                y_ = y_.float().to(device)
                y = model(x)
                loss = criterion(y, y_)
                accumulate_test_loss += loss.item()

        # 记录loss
        avg_train_loss = accumulate_train_loss / len(train)
        avg_test_loss = accumulate_test_loss / len(test)

        print("{} / {} train_loss: {:.6f}".format(epoch, epochs, avg_train_loss))
        print("{} / {} test_loss : {:.6f}".format(epoch, epochs, avg_test_loss))
        train_loss_list.append(avg_train_loss)
        test_loss_list.append(avg_test_loss)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            flag = True

        if flag == False and epoch > 200:
            # 100轮未得到best_loss连续3轮则结束训练
            cnt_no_increasing += 1
            if cnt_no_increasing > 10:
                break
        else:
            cnt_no_increasing = 0

        if epoch % 50 == 0:
            # 每50轮保存权值数据
            torch.save(best_model_weights, './results/MLP/former_{}_rounds.pth'.format(int(epoch)))

    # 保存文件和权重
    np.savetxt('./log/MLP/train_loss_list.txt', np.array(train_loss_list))
    np.savetxt('./log/MLP/valid_loss_list.txt', np.array(test_loss_list))

    # loss可视化
    visualizer = LossVisualizer(train_loss_list, test_loss_list, "./results/MLP/MLP_loss.jpg")
    visualizer.draw()

    # 存储最终权值文件
    torch.save(best_model_weights, './results/MLP/best.pth')


if __name__ == '__main__':
    # options的取值代表训练的方式
    # 0：不训练
    # 1：只训练AE
    # 2：只训练MLP
    option = 2

    if option == 1:
        train_ae()
    elif option == 2:
        train_mlp()
