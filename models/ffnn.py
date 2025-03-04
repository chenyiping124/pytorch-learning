import torch
import torch.nn as nn

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_type):
        """
        初始化前馈神经网络。

        参数:
        input_size (int): 输入特征的维度。
        hidden_size (int): 隐藏层的神经元数量。
        output_size (int): 输出层的神经元数量。
        activation_type (str): 激活函数类型，可选值为 'linear', 'sigmoid', 'softmax'。
        """
        super(FeedForwardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        if activation_type == 'linear':
            self.activation = None
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            raise ValueError("Invalid activation type. Choose from 'linear', 'sigmoid', 'softmax'.")

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入数据。

        返回:
        torch.Tensor: 模型的输出。
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        if self.activation:
            out = self.activation(out)

        return out