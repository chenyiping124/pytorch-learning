import unittest
from models.cnn import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader




# 定义测试用例类
class TestCNN(unittest.TestCase):
    def setUp(self):
        """
        测试用例的初始化方法，在每个测试方法执行前调用。
        """
        # 定义数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # 加载 MNIST 训练集
        train_dataset = datasets.MNIST(root='./data/mnist', train=True,
                                       download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # 加载 MNIST 测试集
        test_dataset = datasets.MNIST(root='./data/mnist', train=False,
                                      download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        # 初始化 CNN 模型
        self.model = CNN()
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

    def test_cnn(self):
        """
        测试 CNN 模型的训练过程。
        """
        # 训练模型
        num_epochs = 1
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                          f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        # 测试模型
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} '
              f'({100. * correct / len(self.test_loader.dataset):.0f}%)\n')
        self.assertTrue(correct > 0)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    # 创建测试套件
    suite.addTest(TestCNN('test_training'))
    suite.addTest(TestCNN('test_evaluation'))
    # 运行测试套件
    runner = unittest.TextTestRunner()
    runner.run(suite)
