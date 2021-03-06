import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_nums=1000, output_nums=1000):
        super(MLP, self).__init__()

        self.input_nums = input_nums
        self.output_nums = output_nums
        self.mlp = nn.Sequential(
            # 1000 -> 1200 -> 1000 -> 1000 -> 700
            nn.Linear(input_nums, int(input_nums * 1.2)),
            nn.ReLU(),
            nn.Linear(int(input_nums * 1.2), input_nums),
            nn.ReLU(),
            nn.Linear(input_nums, input_nums),
            nn.ReLU(),
            nn.Linear(input_nums, output_nums),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)
