import torch
import unittest
from torch.utils.data import Dataset
from datasets.randomset import RandomDataset

class TestRandomDataset(unittest.TestCase):
    def test_linear_dataset(self):
        num_samples = 100
        input_size = 10
        task_type = 'linear'
        dataset = RandomDataset(num_samples, input_size, task_type)
        self.assertEqual(len(dataset), num_samples)
        inputs, targets = dataset[0]
        self.assertEqual(inputs.shape[0], input_size)
        self.assertEqual(targets.shape, torch.Size([1]))

    def test_binary_dataset(self):
        num_samples = 100
        input_size = 10
        task_type = 'binary'
        dataset = RandomDataset(num_samples, input_size, task_type)
        self.assertEqual(len(dataset), num_samples)
        inputs, targets = dataset[0]
        self.assertEqual(inputs.shape[0], input_size)
        self.assertTrue(targets.item() in [0, 1])

    def test_multiclass_dataset(self):
        num_samples = 100
        input_size = 10
        task_type = 'multiclass'
        num_classes = 5
        dataset = RandomDataset(num_samples, input_size, task_type, num_classes)
        self.assertEqual(len(dataset), num_samples)
        inputs, targets = dataset[0]
        self.assertEqual(inputs.shape[0], input_size)
        self.assertTrue(0 <= targets.item() < num_classes)


if __name__ == '__main__':
    unittest.main()