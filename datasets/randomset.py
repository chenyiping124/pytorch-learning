import torch
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size, task_type, num_classes=None):
        """
        初始化随机数据集。

        参数:
        num_samples (int): 数据集的样本数量。
        input_size (int): 输入特征的维度。
        task_type (str): 任务类型，可选值为 'linear', 'binary', 'multiclass'。
        num_classes (int, 可选): 多分类任务中的类别数量，仅在 task_type 为 'multiclass' 时需要。
        """
        self.num_samples = num_samples
        self.input_size = input_size
        self.task_type = task_type

        # 生成随机输入特征
        self.inputs = torch.randn(num_samples, input_size)

        if task_type == 'linear':
            # 线性预测任务，生成随机权重和偏置计算目标值
            self.weights = torch.randn(input_size, 1)
            self.bias = torch.randn(1)
            self.targets = torch.mm(self.inputs, self.weights) + self.bias
        elif task_type == 'binary':
            # 二分类任务，根据输入特征计算 logits 并通过 sigmoid 转换为概率，然后进行二分类
            logits = torch.sum(self.inputs, dim=1, keepdim=True)
            probabilities = torch.sigmoid(logits)
            self.targets = (probabilities > 0.5).float()
        elif task_type == 'multiclass':
            if num_classes is None:
                raise ValueError("num_classes must be provided for multiclass task.")
            # 多分类任务，生成随机的类别标签
            self.targets = torch.randint(0, num_classes, (num_samples,))
        else:
            raise ValueError("Invalid task type. Choose from 'linear', 'binary', 'multiclass'.")

    def __len__(self):
        """
        返回数据集的样本数量。

        返回:
        int: 数据集的样本数量。
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        根据索引获取样本。

        参数:
        idx (int): 样本的索引。

        返回:
        tuple: 包含输入特征和对应目标值的元组。
        """
        return self.inputs[idx], self.targets[idx]