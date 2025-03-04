import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from datasets.randomset import RandomDataset
from models.ffnn import FeedForwardNeuralNetwork
from torch.utils.data import DataLoader, random_split


class TestFFNN(unittest.TestCase):
    def test_linear(self):
        '''
        线性回归
        '''
        print('TestFFNN.test_linear')
        # hyper-parameters
        num_epochs = 20000
        num_samples = 1000
        input_size = 10
        hidden_size = 100
        output_size = 1
        batch_size = 32
        learning_rate = 0.01

        # 数据集
        dataset = RandomDataset(num_samples, input_size, task_type='linear')
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 模型及损失函数
        model = FeedForwardNeuralNetwork(input_size, hidden_size, output_size, 'linear')
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # 训练
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                y_pred = model(inputs)
                loss = criterion(y_pred, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: loss: {loss.item():.4f}')

        # 测试
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in test_loader:
                y_pred = model(inputs)
                loss = criterion(y_pred, targets)
                total_loss += loss.item()
            avg_loss = total_loss / len(test_loader)
            print(f'Test loss: {avg_loss:.4f}')
            self.assertTrue(avg_loss < 0.2)

    def test_binary(self):
        '''
        二分类
        '''
        print('TestFFNN.test_binary')
        # hyper-parameters
        num_epochs = 20000
        num_samples = 1000
        input_size = 10
        hidden_size = 100
        output_size = 1
        batch_size = 32
        learning_rate = 0.01

        # 数据集
        dataset = RandomDataset(num_samples, input_size, task_type='binary')
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 模型及损失函数
        model = FeedForwardNeuralNetwork(input_size, hidden_size, output_size, 'sigmoid')
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # 训练
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                y_pred = model(inputs)
                loss = criterion(y_pred, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: loss: {loss.item():.4f}')

        # 测试
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in test_loader:
                y_pred = model(inputs)
                loss = criterion(y_pred, targets)
                total_loss += loss.item()
            avg_loss = total_loss / len(test_loader)
            print(f'Test loss: {avg_loss:.4f}')
            self.assertTrue(avg_loss < 0.1)

    def test_multiclass(self):
        '''
        多分类
        '''
        print('TestFFNN.test_multiclass')
        # hyper-parameters
        num_epochs = 20000
        num_samples = 1000
        input_size = 10
        hidden_size = 100
        output_size = 3
        batch_size = 32
        learning_rate = 0.01

        # 数据集
        dataset = RandomDataset(num_samples, input_size, task_type='multiclass', num_classes=output_size)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 模型及损失函数
        model = FeedForwardNeuralNetwork(input_size, hidden_size, output_size, 'softmax')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # 训练
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                y_pred = model(inputs)
                loss = criterion(y_pred, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: loss: {loss.item():.4f}')

        # 测试
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in test_loader:
                y_pred = model(inputs)
                loss = criterion(y_pred, targets)
                total_loss += loss.item()
            avg_loss = total_loss / len(test_loader)
            print(f'Test loss: {avg_loss:.4f}')
            self.assertTrue(avg_loss < 2)


if __name__ == '__main__':
    unittest.main()
