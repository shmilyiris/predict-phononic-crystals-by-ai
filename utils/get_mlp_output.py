from nets.AE import AE
import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np

if __name__ == '__main__':
    ae_path = '../results/AE/best.pth'
    img_path = '../data/comsol_format_figures_simplify_copy'
    ae_model = AE(features_num = 700)
    ae_model.load_state_dict(torch.load(ae_path))
    ae_model.eval()

    for idx, filename in enumerate(os.listdir(img_path)):
        img = cv2.imread(img_path + '/' + filename, cv2.IMREAD_GRAYSCALE) # 灰度图读取图片
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.view(1, 1, 128, 128)
        output, features = ae_model(img_tensor)
        arr = features.detach().numpy()
        file_dir = '../data/mlp_train/output_' + str(idx + 1) + '.txt'
        np.savetxt(file_dir, arr)