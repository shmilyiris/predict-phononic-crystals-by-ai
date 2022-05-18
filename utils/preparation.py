import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt


def generate_params():
    max_phi = 0.12
    max_n = 10
    max_c = 0.4
    phi = random.random() * max_phi
    n1 = random.randint(1, max_n)
    n2 = random.randint(1, max_n)
    n3 = random.randint(1, max_n)
    n4 = random.randint(1, max_n)
    c1 = random.random() * max_c
    c2 = random.random() * max_c
    c3 = random.random() * max_c
    c4 = random.random() * max_c
    # print(phi, n1, n2, n3, n4, c1, c2, c3, c4)
    return phi, n1, n2, n3, n4, c1, c2, c3, c4


def generate_formula_eq2(datasize=3, startidx=1):
    second_path = 'figures_simplify'
    L0 = 10
    for i in range(startidx, startidx + datasize):
        phi, n1, n2, n3, n4, c1, c2, c3, c4 = generate_params()
        phi = 0.12
        if i % 500 == 0:
            print("generate round .. " + str(i))

        # 根据参数得到r0大小
        r0 = math.sqrt(L0**2 * phi / (1 + (c1**2) / 2 + (c2 ** 2) / 2) / math.pi)

        # 给定theta, r，得到点集
        x = []
        y = []
        for theta in np.linspace(0, 2*math.pi, num=200):
            r = r0 * (1 + c1 * math.cos(n1 * theta) + c2 * math.cos(n2 * theta))
            x.append(r * math.cos(theta))
            y.append(r * math.sin(theta))
        x = np.array(x)
        y = np.array(y)

        # 画图
        plt.axis('off')
        fig = plt.gcf()
        frame = plt.gca()

        # 设置坐标轴范围
        frame.set_xlim([-L0 / 2, L0 / 2])
        frame.set_ylim([-L0 / 2, L0 / 2])
        # 指定输出图像大小
        fig.set_size_inches(1.28/3, 1.28/3)  # dpi = 300, output = 128*128 pixels
        # 绘图
        plt.plot(x,y, color="white")
        plt.fill(x, y, color="white")
        # frame.xaxis.set_major_locator(plt.NullLocator())
        # frame.yaxis.set_major_locator(plt.NullLocator())

        # 调整边距，保存图片
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        save_path = os.path.join('..', 'data', second_path, f"{i}.jpg")
        plt.savefig(save_path, format='jpg', facecolor='black',dpi=300, pad_inches=0)
        plt.close()

def generate_formula_eq4(datasize=3, startidx=1):
    second_path = 'comsol_figures'
    L0 = 10
    for i in range(startidx, startidx + datasize):
        phi, n1, n2, n3, n4, c1, c2, c3, c4 = generate_params()
        if i % 500 == 0:
            print("generate round .. " + str(i))

        # 根据参数得到r0大小
        r0 = math.sqrt(L0**2 * phi / (1 + (c1**2 + c2**2 + c3**2 + c4**2) / 2) / math.pi)

        # 给定theta, r，得到点集
        x = []
        y = []
        for theta in np.linspace(0, 2*math.pi, num=200):
            r = r0 * (1 + c1 * math.cos(n1 * theta) + c2 * math.cos(n2 * theta) + c3 * math.cos(n3 * theta) + c4 * math.cos(n4 * theta))
            x.append(r * math.cos(theta))
            y.append(r * math.sin(theta))
        x = np.array(x)
        y = np.array(y)

        # 画图
        plt.axis('off')
        fig = plt.gcf()
        frame = plt.gca()

        # 设置坐标轴范围
        frame.set_xlim([-L0 / 2, L0 / 2])
        frame.set_ylim([-L0 / 2, L0 / 2])
        # 指定输出图像大小
        fig.set_size_inches(1.28/3, 1.28/3)  # dpi = 300, output = 128*128 pixels
        # 绘图
        plt.plot(x,y, color="white")
        plt.fill(x, y, color="white")
        # frame.xaxis.set_major_locator(plt.NullLocator())
        # frame.yaxis.set_major_locator(plt.NullLocator())

        # 调整边距，保存图片
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        save_path = os.path.join('..', 'data', second_path, f"{i}.jpg")
        plt.savefig(save_path, format='jpg', facecolor='black',dpi=300, pad_inches=0)
        plt.close()

def generate_random(datasize=100000, startidx=200001):
    L0 = 10
    second_path = 'comsol_figures'
    for i in range(startidx, startidx + datasize):
        # 给定theta, r，得到点集
        x = []
        y = []
        c = 0.5
        if i % 500 == 0:
            print("generate round .. " + str(i))
        for theta in np.linspace(0, 2*math.pi, num=200):
            r = 5 * random.random() * c
            x.append(r * math.cos(theta))
            y.append(r * math.sin(theta))
        x = np.array(x)
        y = np.array(y)

        # 画图
        plt.axis('off')
        fig = plt.gcf()
        frame = plt.gca()

        # 设置坐标轴范围
        frame.set_xlim([-L0 / 2, L0 / 2])
        frame.set_ylim([-L0 / 2, L0 / 2])
        # 指定输出图像大小
        fig.set_size_inches(1.28/3, 1.28/3)  # dpi = 300, output = 128*128 pixels
        # 绘图
        plt.plot(x,y, color="white")
        plt.fill(x, y, color="white")
        # frame.xaxis.set_major_locator(plt.NullLocator())
        # frame.yaxis.set_major_locator(plt.NullLocator())

        # 调整边距，保存图片
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        save_path = os.path.join('..', 'data', second_path, f"{i}.jpg")
        plt.savefig(save_path, format='jpg', facecolor='black',dpi=300, pad_inches=0)
        plt.close()

if __name__ == '__main__':
    size1 = 2500       # 用4阶公式生成的samples
    size2 = 5000        # 用随机公式生成的samples
    size0 = 30000       # 用2阶公式生成samples
    generate_formula_eq2(size0, 1)
    # generate_formula_eq4(size1, 10001)
    # generate_random(size2, size1 + 1)
