import torch.nn as nn
from torchvision.models import resnet18
from .convlstm import ConvLSTM
import matplotlib.pyplot as plt
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.outSize = (288, 512)
        self.upsample = nn.Upsample(size=self.outSize, mode='bicubic')

        self.resnet = resnet18(pretrained=True)
        for child in list(self.resnet.children())[:-2]:
            for param in child.parameters():
                param.requires_grad = False
        self.conv = nn.Sequential(*list(self.resnet.children())[:-2]) # 9, 16

        # self.reduction = nn.Sequential(
        #                 nn.Conv2d(512, 128, 1),
        #                 nn.ReLU(inplace=True)
        #             )

        self.lstm = ConvLSTM(input_size=(9, 16), input_dim=512, hidden_dim=[512], kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.up = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 18, 32
                nn.Conv2d(256, 128, 3, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(128, 64, 3, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(64, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(16, 1, 3, padding=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 288, 512
                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ip):
        # Combine batches and sequences
        f = ip.reshape(-1, *ip.shape[2:])
        f = self.conv(f)
        # f = self.reduction(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = self.lstm(f)[0][0]
        f = f.reshape(-1, *f.shape[2:])
        f = self.up(f)
        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = self.sigmoid(f)

        # Separate batches and sequences
        return f

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class LossFrame(nn.Module):
    def __init__(self):
        super(LossFrame, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def softmax(self, i):
        return i / i.sum(dim=(1, 2, 3),keepdim=True)
    
    def forward(self, expected, predicted):
        expected = expected.view(-1, *expected.shape[2:])
        predicted = predicted.view(-1, *predicted.shape[2:])
        # fig, ax = plt.subplots(1,2)

        # ax[1].imshow(predicted[0, 0, :, :].cpu().detach().numpy(), cmap='bone')#, vmin=0,vmax=.00001)
        # ax[0].imshow(self.softmax(predicted)[0, 0, :, :].cpu().detach().numpy(), cmap='bone')#, vmin=0,vmax=.00001)
        # plt.show()
        # print(predicted[0, :, :, :].sum())
        predicted = (self.softmax(predicted))
        predicted = predicted.log() # see KLDiv        
        expected = self.softmax(expected)

        loss = self.criterion(predicted, expected)
        return loss
        
class LossSequence(nn.Module):
    def __init__(self):
        super(LossSequence, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def softmax(self, i):
        return i.exp() / i.exp().sum(dim=(1, 2, 3, 4),keepdim=True)
        
    def forward(self, expected, predicted):
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(expected[0, 0, 0, :, :].cpu().detach().numpy(), cmap='bone', vmin=0,vmax=1)
        # ax[1].imshow(predicted[0, 0, 0, :, :].cpu().detach().numpy(), cmap='bone', vmin=0,vmax=1)
        # plt.show()
        predicted = (self.softmax(predicted)).log() # see KLDiv
        expected = self.softmax(expected)

        loss = self.criterion(predicted, expected)
        return loss
