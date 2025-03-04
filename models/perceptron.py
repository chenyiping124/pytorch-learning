import torch
import torch.nn as nn

class SimplePerceptron(nn.Module):
    def __init__(self, input_size):
        """
        初始化简单感知机模型。

        参数:
        input_size (int): 输入特征的维度。
        """
        super(SimplePerceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入数据。

        返回:
        torch.Tensor: 经过感知机处理后的输出。
        """
        out = self.linear(x)
        out = self.sigmoid(out)
        return out