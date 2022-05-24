import os
# path = "D:\\University-Learning\\graduation_project\\AE+MLP\\data\\comsol_figures"
# new_path = "D:\\University-Learning\\graduation_project\\AE+MLP\\data\\comsol_format_figures_simplify"
new_path = '../data/predict_pre_comsol_format'

for i in range(1, 1500 + 1):
    oldname = new_path + os.sep + str(i) + ".jpg"
    newname = new_path + os.sep + "image2model_image" + str(i)

    os.rename(oldname, newname)