import os
import torch

import numpy as np

from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from .models import FCN_MT

from . import dense_transforms
import matplotlib.pyplot as plt

class VehicleClassificationDataset(Dataset):
    def __init__(self, dataset_path, transform):
        """
        Your code here
        Hint: load your data from provided dataset (VehicleClassificationDataset) to train your designed model
        """
        # e.g., Bicycle 0, Car 1, Taxi 2, Bus 3, Truck 4, Van 5
        self.dataset_path = dataset_path
        self.data = []
        self.label = []

        # for type in os.listdir('./'+dataset_path+'/train_subset'):
        #     for file in os.listdir('./'+dataset_path+'/train_subset'+type):
        self.transform  = transform
        for path in glob(dataset_path+'/*'):
            for img in glob(path+'/*'):
                self.data.append(img)
                if path[-3:] == "cle":
                    self.label.append(0)
                elif path[-3:] == "Car":
                    self.label.append(1)
                elif path[-3:] == "axi":
                    self.label.append(2)
                elif path[-3:] == "Bus":
                    self.label.append(3)
                elif path[-3:] == "uck":
                    self.label.append(4)
                elif path[-3:] == "Van":
                    self.label.append(5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        Hint: generate samples for training
        Hint: return image, and its image-level class label
        """
        return self.transform(Image.open(self.data[idx])), self.label[idx]


class DenseCityscapesDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        """
        Your code here
        """
        self.dataset_path = dataset_path
        self.data = []
        self.label = []
        self.depth = []

        self.transform = transform

        for path in glob(dataset_path+'/*'):
            for img in glob(path+'/*'):
                if path[-3:] == "pth":
                    value = np.load(img)[:,:,0]
                    disparity = (value * 65535 - 1) / 256
                    depth = (0.222384 * 2273.82) / disparity 
                    depth[depth < 0] = 0
                    self.depth.append(depth)
                elif path[-3:] == "age":
                    self.data.append(Image.fromarray(np.uint8(np.load(img)*255)))
                    # if self.data[-1].shape != (128, 256, 3):
                    #     print(self.data[-1].shape)
                elif path[-3:] == "bel":
                    self.label.append(dense_transforms.label_to_pil_image(np.load(img)))
                    # if self.label[-1].shape != (128, 256, 1):
                    #     print(self.label[-1].shape)
        

    def __len__(self):

        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):

        """
        Hint: generate samples for training
        Hint: return image, semantic_GT, and depth_GT
        """
        if self.transform.__class__.__name__ == "Compose3":
            # print(self.data[idx].shape)
            # print(self.label[idx].shape)
            # print(self.depth[idx].shape)
            image, label, depth = self.transform(self.data[idx], self.label[idx], self.depth[idx])
            return image, label, depth
        else:
            image, label = self.transform(self.data[idx], self.label[idx])
            return image, label, self.depth[idx]


class DenseVisualization():
    def __init__(self, img, depth, segmentation):
        self.img = img
        self.depth = depth
        self.segmentation = segmentation

    def __visualizeitem__(self):
        """
        Your code here
        Hint: you can visualize your model predictions and save them into images. 
        """
        plt.figure()
        f, axarr = plt.subplots(5, 6)
        model = FCN_MT()
        output_ss, output_dp = model(self.img)
        for i in range(5):
            for j in range(6):
                axarr[i, j].imshow(self.img[j].transpose((1,2,0)))
            for j in range(6,12):
                axarr[i, j].imshow(output_dp[j], cmap="plasma") 
            for j in range(12,18):
                axarr[i, j].imshow(self.depth[j], cmap="plasma") 
            for j in range(18,24):
                axarr[i, j].imshow(dense_transforms.label_to_pil_image(output_ss[j])) 
            for j in range(24,30):
                axarr[i, j].imshow(dense_transforms.label_to_pil_image(self.segmentation[j])) 

def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = VehicleClassificationDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseCityscapesDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


class DepthError(object):
    def __init__(self, gt, pred):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.gt = gt
        self.pred = pred

    @property
    def compute_errors(self):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((self.gt / self.pred), (self.pred / self.gt))
        a1 = (thresh < 1.25     ).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        # rmse = (self.gt - self.pred) ** 2
        # rmse = np.sqrt(rmse.mean())

        # rmse_log = (np.log(self.gt) - np.log(self.pred)) ** 2
        # rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean((np.abs(np.array(self.gt) - np.array(self.pred)) / (np.array(self.gt))))

        # sq_rel = np.mean(((self.gt - self.pred) ** 2) / self.gt)

        return abs_rel, a1, a2, a3