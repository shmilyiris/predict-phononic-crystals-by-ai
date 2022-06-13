import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

if __name__ == '__main__':
    path = '../data/predict_txt'
    img_path = '../data/predict_pre'
    output_path = '../results/bandgap'
    freq_num = 20

    img_names = []
    for idx, filename in enumerate(os.listdir(img_path)):
        img_names.append(img_path + '\\' + filename)

    for idx, filename in enumerate(os.listdir(path)):
        obj = open(path + '\\' + filename, 'r')
        lines = obj.readlines()

        freqs = []
        k = []
        for index, line in enumerate(lines):
            # print("Line {}: {}".format(index, line.strip()))
            x, y = "", ""
            flag = False
            i = 0
            while i < len(line):
                c = line[i]
                if c != ' ' and c != '\n':
                    if not flag:
                        x += c
                    else:
                        y += c
                else:
                    flag = True

                i += 1

            k.append(float(x))
            freqs.append(float(y))

        # 获取每个特征频率的最小频率和最大频率，组成区间
        minF, maxF = 1e9, -1
        sections = []
        for i in range(len(freqs)):
            minF = min(minF, freqs[i])
            maxF = max(maxF, freqs[i])
            if (i + 1) % 15 == 0:
                sections.append([minF, maxF])
                minF, maxF = 1e9, -1

        # 区间合并算法
        Gst, Ged = -1e9, -1e9  # Global start, Global end
        sections.sort()
        merged_sections = []
        for st, ed in sections:
            if Ged < st:
                if Gst != -1e9:
                    merged_sections.append([Gst, Ged])
                Gst, Ged = st, ed
            else:
                Ged = max(Ged, ed)

        if Gst != -1e9:
            merged_sections.append([Gst, Ged])

        # 处理边界
        maxFreq = 3e5
        feature_nums = 1000
        merged_sections[0][0] = min(0, merged_sections[0][0])
        merged_sections[-1][1] = max(maxFreq, merged_sections[-1][1])

        x_axis = ['Γ', 'X', 'M', 'Γ']
        colors = ["red", "green", "blue", "orange", "gray", "black", "purple", "lightblue", "lightgreen", "lightpink", "pink"]


        k_list = []
        f_list = []
        x = []
        y = []
        for i in range(len(freqs)):
            x.append(k[i])
            y.append(freqs[i])
            if (i + 1) % freq_num == 0:
                k_list.append(x)
                f_list.append(y)
                x, y = [], []

        # 原图
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.axis('off')
        img = Image.open(img_names[idx])
        img = np.array(img)
        plt.imshow(img)

        # 画色散曲线图
        plt.subplot(1, 2, 2)
        plt.xlim(0, 3)
        plt.ylim(0, 2e5)
        for i in range(len(k_list)):
            color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0]
            plt.plot(k_list[i], f_list[i], color=color)

        plt.xticks([0,1,2,3], x_axis, fontsize=22)
        plt.xlabel("k", fontsize=15)
        plt.yticks([0, 0.5e5, 1e5, 1.5e5, 2e5], [0, 50, 100, 150, 200])
        plt.ylabel("Frequency/kHz", fontsize=15)

        # 画带隙
        for i in range(1, len(merged_sections)):
            starty = merged_sections[i - 1][1]
            height = merged_sections[i][0] - merged_sections[i - 1][1]
            currentAxis = plt.gca()
            rect = patches.Rectangle((0, starty), 3, height, linewidth=1, edgecolor='black', facecolor='lightgray')
            currentAxis.add_patch(rect)
        fig.tight_layout()
        plt.savefig(output_path + '\\' + str(idx + 1) + '.jpg')
        plt.close('all')

