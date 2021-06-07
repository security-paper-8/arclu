import os
import numpy as np
import torch
import torchvision
from torchvision import transforms


class TransferAttack:
    def __init__(self, dataset):
        self.dataset = dataset

    def is_benign_okay(self, x_benign, y_benign):
        raise NotImplementedError()

    @staticmethod
    def scale_examples(examples: np.array, scaled_min: float, scaled_max: float) -> np.array:
        min_val = examples.min()
        max_val = examples.max()
        scaled_01 = (examples - min_val) / (max_val - min_val)
        return scaled_01 * (scaled_max - scaled_min) + scaled_min

    def preprocess_data_shape(self, x_benign):
        return x_benign

    def perturb(self, min_val, max_val):
        relative_path = "/../generate_transfer/adv_data/" + self.dataset + ".npz"
        blackbox_data = np.load(os.path.abspath(
            os.path.dirname(__file__)) + relative_path)

        scaled_adv_examples = TransferAttack.scale_examples(blackbox_data['adv_examples'],
                                                            min_val, max_val)
        scaled_benign_examples = TransferAttack.scale_examples(blackbox_data['benign_examples'],
                                                               min_val, max_val)
        adv_examples = self.preprocess_data_shape(scaled_adv_examples)
        benign_examples = self.preprocess_data_shape(scaled_benign_examples)
        y_true = blackbox_data['y_true']

        okay_filt_list = []
        BATCH_SIZE = 128
        for i in range(len(adv_examples) // BATCH_SIZE + 1):
            rng = np.arange(i * BATCH_SIZE, min((i + 1) *
                                                BATCH_SIZE, len(adv_examples)))
            benign_examples_batch = benign_examples[rng]
            y_true_batch = y_true[rng]
            okay_filt_batch = self.is_benign_okay(
                benign_examples_batch, y_true_batch)
            okay_filt_list.append(okay_filt_batch)

        okay_filt = np.concatenate(okay_filt_list, axis=0).astype(np.bool)

        return benign_examples[okay_filt], y_true[okay_filt], adv_examples[okay_filt]


class SemanticTestAttack:
    def __init__(self, dataset, attack_func, batch_size=128, attack_size=10000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.attack_func = attack_func

        self.source_min = 99999999
        self.source_max = -99999999
        self.load_data()
        self.attack_size = attack_size

    def load_data(self):

        train_loader, test_loader = None, None
        if self.dataset == "mnist":
            transform_test = transforms.Compose([transforms.ToTensor(), ])
            trainset = torchvision.datasets.MNIST(
                root='../data', train=True, download=True, transform=transform_test)
            testset = torchvision.datasets.MNIST(
                root='../data', train=False, download=True, transform=transform_test)

            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=self.batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=self.batch_size, shuffle=False)

        elif self.dataset == "cifar" or self.dataset == "cifar10":
            transform_test = transforms.Compose([transforms.ToTensor(), ])
            trainset = torchvision.datasets.CIFAR10(
                root='../data', train=True, download=True, transform=transform_test)
            testset = torchvision.datasets.CIFAR10(
                root='../data', train=False, download=True, transform=transform_test)

            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=self.batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=self.batch_size, shuffle=False)
        else:
            raise NotImplementedError()

        for i, (x_data, y_data) in enumerate(train_loader):
            if x_data.max() > self.source_max:
                self.source_max = x_data.max().item()
            if x_data.min() < self.source_min:
                self.source_min = x_data.min().item()

        self.train_loader = train_loader
        self.test_loader = test_loader

    def perturb(self, min_val, max_val):
        x_benign_to_save_list = []
        x_adv_to_save_list = []

        x_benign_to_compute_list = []
        x_adv_to_compute_list = []

        y_true_list = []
        y_adv_pred_list = []
        total = 0
        for i, (x_data, y_data) in enumerate(self.test_loader):
            x_data_numpy = x_data.numpy()
            y_data_numpy = y_data.numpy()
            x_data_numpy_scaled = (x_data_numpy - self.source_min) / (
                self.source_max - self.source_min) * (max_val - min_val) + min_val
            x_adv_batch, y_adv_pred_batch = self.attack_func(
                x_data_numpy_scaled, y_data_numpy)

            x_adv_batch_back_scaled = (x_adv_batch - min_val) / (max_val - min_val) * (
                self.source_max - self.source_min) + self.source_min

            x_adv_to_save_list.append(x_adv_batch_back_scaled)
            x_benign_to_save_list.append(x_data_numpy)

            x_adv_to_compute_list.append(x_adv_batch)
            x_benign_to_compute_list.append(x_data_numpy_scaled)

            y_true_list.append(y_data_numpy)
            y_adv_pred_list.append(y_adv_pred_batch)
            total += x_data.size(0)
            if total > self.attack_size:
                break

        return np.concatenate(x_benign_to_compute_list, axis=0), np.concatenate(y_true_list, axis=0), \
            np.concatenate(x_adv_to_compute_list, axis=0), np.concatenate(y_adv_pred_list, axis=0), \
            np.concatenate(x_benign_to_save_list, axis=0), np.concatenate(
                x_adv_to_save_list, axis=0)

    def save_result(self, path, xb, yb, xa, ya):
        np.savez(path, x_benign=xb, y_true=yb,
                 x_adversarial=xa, y_adversarial=ya)
