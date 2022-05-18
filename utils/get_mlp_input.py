import os
import numpy as np
# path = "D:\\University-Learning\\graduation_project\\AE+MLP\\results\\comsol_txt"
path = '../results/comsol_txt'
for idx, filename in enumerate(os.listdir(path)):
    obj = open(path + '\\' + filename, 'r')
    lines = obj.readlines()

    freqs = []
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

        freqs.append(float(y))

    # 获取每个特征频率的最小频率和最大频率，组成区间
    k = 0
    minF, maxF = 1e9, -1
    sections = []
    for i in range(len(freqs)):
        minF = min(minF, freqs[i])
        maxF = max(maxF, freqs[i])
        if (i + 1) % 15 == 0:
            sections.append([minF, maxF])
            minF, maxF = 1e9, -1

    # 区间合并算法
    Gst, Ged = -1e9, -1e9 # Global start, Global end
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

    # 区间映射算法
    bitmap = []
    lastEd = 0
    for i in range(len(merged_sections)):
        st, ed = merged_sections[i][0], merged_sections[i][1]
        if i != 0:
            # Stop Band
            tmp = int(feature_nums * (st - lastEd) / maxFreq)
            while tmp:
                tmp -= 1
                bitmap.append(0)

        # Pass Band
        tmp = int(feature_nums * (ed - st) / maxFreq)
        while tmp:
            tmp -= 1
            bitmap.append(1)

        lastEd = ed

    while len(bitmap) < feature_nums:
        bitmap.append(1)

    file_dir = '../data/mlp_train/input_' + str(idx + 1) + '.txt'
    # input_file = open(file_dir, 'w')
    # input_file.write('\n')
    np.savetxt(file_dir, np.array(bitmap))
    obj.close()

