import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        """
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=6, bias=True)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.batch_norm = nn.BatchNorm1d(1024)
        # self.pred_score = nn.Softmax(6)

    def forward(self, x):
        """
        Your code here
        """
        print(x.shape)
        x = self.resnet(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
    
        return x


class FCN_ST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Single-Task FCN needs to output segmentation maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.resnet = resnet50(pretrained=True)

        self.relu = nn.ReLU()

        # self.conv1x1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1,1), stride=1)
        
        self.up_conv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv = nn.Conv2d(in_channels=64, out_channels=19, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding CNNClassifier
              convolution
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.relu(x)
        x_skip4 = x.clone()

        x = self.resnet.layer1(x)
        x_skip3 = x.clone()

        x = self.resnet.layer2(x)
        x_skip2 = x.clone()

        x = self.resnet.layer3(x)
        x_skip1 = x.clone()

        x = self.resnet.layer4(x)

        x = torch.cat([x, x_skip1], dim=1)
        x = self.up_conv1(x)
        x = self.bn1(x)

        x = torch.cat([x, x_skip2], dim=1)
        x = self.up_conv2(x)
        x = self.bn2(x)

        x = torch.cat([x, x_skip3], dim=1)
        x = self.up_conv3(x)
        x = self.bn3(x)

        x = torch.cat([x, x_skip4], dim=1)
        x = self.up_conv4(x)
        x = self.bn4(x)

        x = self.conv(x)

        return x


class FCN_MT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Multi-Task FCN needs to output both segmentation and depth maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN_MT.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation
        @return: torch.Tensor((B,1,H,W)), 1 is one channel for depth estimation
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN_MT.forward')


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.weight=weight
        self.size_average=size_average

    def forward(self, inputs, targets):
        # outputs.requires_grad = True
        if inputs.shape[-1] == 6:
            outputs = -1 * torch.log(torch.true_divide(torch.exp(inputs).T , torch.sum(torch.exp(inputs), dim=1)).T)
            loglik = torch.empty(targets.shape, requires_grad=False, dtype=torch.float32)
            for i, val in enumerate(targets):
                loglik[i] = outputs[i,val]

        else:
            CEloss = nn.CrossEntropyLoss(torch.tensor(self.weight), ignore_index=255, size_average=self.size_average)
            outputs = torch.softmax(inputs, 1)
            CEloss(inputs, targets)
        # loglik = torch.gather(loglik.T, 0, targets[])
        # outputs = torch.sum(outputs * targets)

        # softmax = torch.exp(inputs) / torch.exp(inputs).sum(dim=1, keepdim=True)

        # # Get probabilities of true classes
        # true_class_probs = torch.gather(softmax, 1, targets.unsqueeze(1)).squeeze()

        # # Calculate negative log probabilities of true classes
        # neg_log_probs = -torch.log(true_class_probs)

        # # Calculate mean loss over batch
        # loss = neg_log_probs.mean()
        # print("This1, ", loss)
        # print("This2, ", loglik.mean())


        if self.size_average:
            return loglik.mean()
        else:
            return loglik.sum()

model_factory = {
    'cnn': CNNClassifier,
    'fcn_st': FCN_ST,
    'fcn_mt': FCN_MT
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
