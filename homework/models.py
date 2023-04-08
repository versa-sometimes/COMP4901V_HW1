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
        self.fc2 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc3 = nn.Linear(in_features=512, out_features=6, bias=True)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.batch_norm = nn.BatchNorm1d(1024)
        # self.pred_score = nn.Softmax(6)

    def forward(self, x):
        """
        Your code here
        """
        # print(x.shape)
        x = self.resnet(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    
class ST_Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
        self.up_conv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.up_conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv0_0 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.bn6_0 = nn.BatchNorm2d(2048)
        self.conv0_1 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.bn6_1 = nn.BatchNorm2d(2048)

        self.conv1_r = nn.Conv2d(in_channels=1024+512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv1_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7_0 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7_1 = nn.BatchNorm2d(512)

        self.conv2_r = nn.Conv2d(in_channels=512+256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv2_0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn8_0 = nn.BatchNorm2d(256)
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn8_1 = nn.BatchNorm2d(256)

        self.conv3_r = nn.Conv2d(in_channels=256+128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv3_0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9_0 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9_1 = nn.BatchNorm2d(128)

        self.conv4_r = nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(64)
        self.conv4_0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn10_0 = nn.BatchNorm2d(64)
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn10_1 = nn.BatchNorm2d(64)

        self.conv5_0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn11_0 = nn.BatchNorm2d(32)
        self.conv5_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn11_1 = nn.BatchNorm2d(32)
        self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=19, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout2d(0.2)
    
    def forward(self, x, x_skip4, x_skip3, x_skip2, x_skip1):

        x_resid0 = x.clone()
        x = self.conv0_0(x)
        x = self.bn6_0(x)
        x = self.relu(x)
        x = self.conv0_1(x)
        x = self.bn6_1(x)
        x += x_resid0
        x = self.relu(x)
        
        
        x = self.up_conv1(x)
        x = self.bn1(x)
        x = torch.cat([x, x_skip1], dim=1)

        x = self.conv1_r(x)
        x = self.bn7(x)
        x = self.dropout(x)
        x_resid1 = x.clone()
        x = self.conv1_0(x)
        x = self.bn7_0(x)
        x = self.relu(x)
        x = self.conv1_1(x)
        x = self.bn7_1(x)
        x += x_resid1
        x = self.relu(x)

        x = self.up_conv2(x)
        x = self.bn2(x)
        x = torch.cat([x, x_skip2], dim=1)

        x = self.conv2_r(x)
        x = self.bn8(x)
        x = self.dropout(x)
        x_resid2 = x.clone()
        x = self.conv2_0(x)
        x = self.bn8_0(x)
        x = self.relu(x)
        x = self.conv2_1(x)
        x = self.bn8_1(x)
        x += x_resid2
        x = self.relu(x)

        x = self.up_conv3(x)
        x = self.bn3(x)
        x = torch.cat([x, x_skip3], dim=1)

        x = self.conv3_r(x)
        x = self.bn9(x)
        x = self.dropout(x)
        x_resid3 = x.clone()
        x = self.conv3_0(x)
        x = self.bn9_0(x)
        x = self.relu(x)
        x = self.conv3_1(x)
        x = self.bn9_1(x)
        x += x_resid3
        x = self.relu(x)

        x = self.up_conv4(x)
        x = self.bn4(x)
        x = torch.cat([x, x_skip4], dim=1)

        x = self.conv4_r(x)
        x = self.bn10(x)
        x = self.dropout(x)
        x_resid4 = x.clone()
        x = self.conv4_0(x)
        x = self.bn10_0(x)
        x = self.relu(x)
        x = self.conv4_1(x)
        x = self.bn10_1(x)
        x += x_resid4
        x = self.relu(x)

        x = self.up_conv5(x)
        x = self.bn5(x)

        x = self.dropout(x)
        x_resid5 = x.clone()
        x = self.conv5_0(x)
        x = self.bn11_0(x)
        x = self.relu(x)
        x = self.conv5_1(x)
        x = self.bn11_1(x)
        x += x_resid5
        x = self.relu(x)
        x = self.conv5_2(x)

        return x
    
class DP_Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
        self.up_conv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.up_conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv0_0 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.bn6_0 = nn.BatchNorm2d(2048)
        self.conv0_1 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.bn6_1 = nn.BatchNorm2d(2048)

        self.conv1_r = nn.Conv2d(in_channels=1024+512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv1_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7_0 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7_1 = nn.BatchNorm2d(512)

        self.conv2_r = nn.Conv2d(in_channels=512+256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv2_0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn8_0 = nn.BatchNorm2d(256)
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn8_1 = nn.BatchNorm2d(256)

        self.conv3_r = nn.Conv2d(in_channels=256+128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv3_0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9_0 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9_1 = nn.BatchNorm2d(128)

        self.conv4_r = nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(64)
        self.conv4_0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn10_0 = nn.BatchNorm2d(64)
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn10_1 = nn.BatchNorm2d(64)

        self.conv5_0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn11_0 = nn.BatchNorm2d(32)
        self.conv5_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn11_1 = nn.BatchNorm2d(32)
        self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout2d(0.2)
    
    def forward(self, x, x_skip4, x_skip3, x_skip2, x_skip1):

        x_resid0 = x.clone()
        x = self.conv0_0(x)
        x = self.bn6_0(x)
        x = self.relu(x)
        x = self.conv0_1(x)
        x = self.bn6_1(x)
        x += x_resid0
        x = self.relu(x)
        
        
        x = self.up_conv1(x)
        x = self.bn1(x)
        x = torch.cat([x, x_skip1], dim=1)

        x = self.conv1_r(x)
        x = self.bn7(x)
        x = self.dropout(x)
        x_resid1 = x.clone()
        x = self.conv1_0(x)
        x = self.bn7_0(x)
        x = self.relu(x)
        x = self.conv1_1(x)
        x = self.bn7_1(x)
        x += x_resid1
        x = self.relu(x)

        x = self.up_conv2(x)
        x = self.bn2(x)
        x = torch.cat([x, x_skip2], dim=1)

        x = self.conv2_r(x)
        x = self.bn8(x)
        x = self.dropout(x)
        x_resid2 = x.clone()
        x = self.conv2_0(x)
        x = self.bn8_0(x)
        x = self.relu(x)
        x = self.conv2_1(x)
        x = self.bn8_1(x)
        x += x_resid2
        x = self.relu(x)

        x = self.up_conv3(x)
        x = self.bn3(x)
        x = torch.cat([x, x_skip3], dim=1)

        x = self.conv3_r(x)
        x = self.bn9(x)
        x = self.dropout(x)
        x_resid3 = x.clone()
        x = self.conv3_0(x)
        x = self.bn9_0(x)
        x = self.relu(x)
        x = self.conv3_1(x)
        x = self.bn9_1(x)
        x += x_resid3
        x = self.relu(x)

        x = self.up_conv4(x)
        x = self.bn4(x)
        x = torch.cat([x, x_skip4], dim=1)

        x = self.conv4_r(x)
        x = self.bn10(x)
        x = self.dropout(x)
        x_resid4 = x.clone()
        x = self.conv4_0(x)
        x = self.bn10_0(x)
        x = self.relu(x)
        x = self.conv4_1(x)
        x = self.bn10_1(x)
        x += x_resid4
        x = self.relu(x)

        x = self.up_conv5(x)
        x = self.bn5(x)

        x = self.dropout(x)
        x_resid5 = x.clone()
        x = self.conv5_0(x)
        x = self.bn11_0(x)
        x = self.relu(x)
        x = self.conv5_1(x)
        x = self.bn11_1(x)
        x += x_resid5
        x = self.relu(x)
        x = self.conv5_2(x)

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

        self.sm_head = ST_Head()

        # self.conv1x1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1,1), stride=1)

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

        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x_skip3 = x.clone()

        x = self.resnet.layer2(x)
        x_skip2 = x.clone()

        x = self.resnet.layer3(x)
        x_skip1 = x.clone()

        x = self.resnet.layer4(x)

        x = self.sm_head(x, x_skip4, x_skip3, x_skip2, x_skip1)

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
        self.resnet = resnet50(pretrained=True)

        self.relu = nn.ReLU()

        self.ss_head = ST_Head()

        # self.conv1x1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1,1), stride=1)
        
        self.dp_head = DP_Head()

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
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.relu(x)

        x_skip4 = x.clone()

        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x_skip3 = x.clone()

        x = self.resnet.layer2(x)
        x_skip2 = x.clone()

        x = self.resnet.layer3(x)
        x_skip1 = x.clone()

        x = self.resnet.layer4(x)

        sm_pred = self.ss_head(x, x_skip4, x_skip3, x_skip2, x_skip1)
        dp_pred = self.dp_head(x, x_skip4, x_skip3, x_skip2, x_skip1)

        return sm_pred, dp_pred


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.weight=weight
        self.size_average=size_average

    def forward(self, inputs, targets):
        if inputs.shape[-1] == 6:
            outputs = -1 * torch.log(torch.true_divide(torch.exp(inputs).T , torch.sum(torch.exp(inputs), dim=1)).T)
            loglik = torch.empty(targets.shape, requires_grad=False, dtype=torch.float32)
            for i, val in enumerate(targets):
                loglik[i] = outputs[i,val]

            if self.size_average:
                return loglik.mean()
            else:
                return loglik.sum()

        else:
            CEloss = nn.CrossEntropyLoss(torch.tensor(self.weight), ignore_index=255, size_average=self.size_average)
            outputs = torch.softmax(inputs, 1)
            loglik = CEloss(inputs, targets)

            return loglik


    
# class MultiTaskLoss(nn.Module):
#     def __init__(self, weight):
#         super(MultiTaskLoss, self).__init__()
#         self.CELoss = nn.CrossEntropyLoss(weight, ignore_index=255)
#         self.L1Loss = nn.L1Loss()

#     def forward(self, outputs_ss, labels, outputs_dp, depth):
#         ce_loss = self.CELoss(outputs_ss, labels)
#         dp_loss = self.L1Loss(outputs_dp, depth)
#         return ce_loss*0.5 + dp_loss*0.5

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
