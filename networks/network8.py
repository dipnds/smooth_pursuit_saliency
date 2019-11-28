import torch.nn as nn
from torchvision.models import resnet18
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

        self.down3d = nn.Sequential(
                        nn.Conv3d(512, 512, (4,1,1), stride=(4,1,1), padding=0, bias=True), # 9, 16 (6)
                        nn.ReLU(inplace=True),
                        nn.Conv3d(512, 512, 3, stride=1, padding=1, bias=True), # 9, 16 (6)
                        nn.ReLU(inplace=True)
                        )
        
        self.interpol = nn.Upsample(scale_factor=4) # 36, 64 (24)

        self.up3d = nn.Sequential(
                        nn.Conv3d(512, 512, 3, stride=1, padding=1, bias=True), # 36, 64 (24)
                        nn.ReLU(inplace=True)
        )

        # up        
        self.up = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'),
                nn.Conv2d(256, 128, 3, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'),
                nn.Conv2d(128, 64, 3, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(64, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 144, 256
                nn.Conv2d(16, 1, 3, padding=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 288, 512
                )

        self.sigmoid = nn.Sigmoid()
           
    def forward(self, ip):

        # Combine batches and sequences
        f = ip.reshape(-1, *ip.shape[2:])
        f = self.conv(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = f.permute(0,2,1,3,4)

        f = self.down3d(f)
        f = self.interpol(f)
        f = self.up3d(f)

        f = f.permute(0,2,1,3,4)
        f = f.reshape(-1, *f.shape[-3:])
        
        f = self.up(f)
        f = f.reshape(*ip.shape[:2], *f.shape[-3:])

        # Separate batches and sequences
        f = self.sigmoid(f)
        return f

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
