import unittest
import torch
from models.perceptron import SimplePerceptron

class TestSimplePerceptron(unittest.TestCase):
    def test_forward_pass(self):
        """
        测试感知机的前向传播。
        """
        input_size = 5
        model = SimplePerceptron(input_size)
        # 生成随机输入
        input_tensor = torch.randn(1, input_size)
        output = model(input_tensor)
        # 检查输出的形状
        self.assertEqual(output.shape, torch.Size([1, 1]))
        # 检查输出的值是否在 [0, 1] 范围内，因为使用了 sigmoid 激活函数
        self.assertTrue((output >= 0).all() and (output <= 1).all())


if __name__ == '__main__':
    unittest.main()