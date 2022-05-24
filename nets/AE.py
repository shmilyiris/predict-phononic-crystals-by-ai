import torch
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2 - 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BasicTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicTConv, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2 - 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class AE(nn.Module):
    def __init__(self, features_num=1000, is_predict=False):
        super(AE, self).__init__()

        # Encoder
        self.make_five_conv = nn.Sequential(
            BasicConv(1, 8, kernel_size=2, stride=2),  # 128,128,1 -> 64,64,8
            BasicConv(8, 16, kernel_size=2, stride=2),  # 64,64,8 -> 32,32,16
            BasicConv(16, 32, kernel_size=2, stride=2),  # 32,32,16 -> 16,16,32
            BasicConv(32, 32, kernel_size=2, stride=2),  # 16,16,32 -> 8,8,32
            BasicConv(32, 64, kernel_size=2, stride=2),  # 8,8,32 -> 4,4,64
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 64, features_num),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(features_num, 4 * 4 * 64),
            nn.ReLU(),
            Reshape(64,4,4),
        )

        # Decoder
        self.make_five_dconv = nn.Sequential(
            BasicTConv(64, 32, kernel_size=2, stride=2),
            BasicTConv(32, 32, kernel_size=2, stride=2),
            BasicTConv(32, 16, kernel_size=2, stride=2),
            BasicTConv(16, 8 , kernel_size=2, stride=2),
            BasicTConv(8 , 1 , kernel_size=2, stride=2),
        )

        self.features_num = features_num
        self.flag = is_predict

    def forward(self, x):
        if not self.flag:
            # 输入x->压缩->解压
            x = self.make_five_conv(x)  # batch,128,128,1 -> batch,4,4,64
            x = nn.Flatten()(x)  # batch,4,4,64 -> batch,1024
            x = self.fc1(x)  # batch,1024 -> batch,700
            features = x  # latent space features
            x = self.fc2(x)  # batch,700 -> batch,1024 -> batch,4,4,64
            x = self.make_five_dconv(x)  # batch,4,4,64 -> batch,128,128,1
            return x, features
        else:
            # 输入x->解压
            x = self.fc2(x)
            x = self.make_five_dconv(x)
            return x
