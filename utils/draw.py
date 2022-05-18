import matplotlib.pyplot as plt
import numpy as np

class LossVisualizer():
    def __init__(self, train_list, test_list, save_path):
        self.train_list = train_list
        self.test_list = test_list
        self.path = save_path

    def draw(self):
        x = [i for i in range(len(self.train_list))]
        x = np.array(x)
        train_y = np.array(self.train_list)
        test_y = np.array(self.test_list)

        plt.xlabel('epoch')
        plt.ylabel('loss')
        l1, = plt.plot(x, train_y, color="red")
        l2, = plt.plot(x, test_y, color="green")
        plt.legend(handles=[l1, l2], labels=['train', 'test'], loc='best')
        plt.savefig(self.path)
        plt.close()
