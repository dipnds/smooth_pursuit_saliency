import torch.nn as nn
from torchvision.models import resnet18
from .convlstm import ConvBLSTM

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.outSize = (288, 512)
        self.upsample = nn.Upsample(size=self.outSize, mode='bicubic', align_corners=False)

        self.resnet = resnet18(pretrained=True)
        for child in list(self.resnet.children())[:-5]:
            for param in child.parameters():
                param.requires_grad = False
        self.conv = nn.Sequential(*list(self.resnet.children())[:-2]) # 9, 16

        # self.reduction = nn.Sequential(
        #         nn.Conv2d(512, 128, 1, bias=True),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(inplace=True)
        # )

        self.lstm1 = ConvBLSTM(in_channels=512, hidden_channels=512, kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.up = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'), # 18, 32
                nn.Conv2d(256, 128, 3, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(128, 64, 3, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic', align_corners=False), # 72, 128
                nn.Conv2d(64, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4,mode='bicubic', align_corners=False), # 72, 128
                nn.Conv2d(16, 1, 3, padding=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Upsample(scale_factor=4,mode='bicubic', align_corners=False), # 288, 512
                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ip):
        # Combine batches and sequences
        f = ip.reshape(-1, *ip.shape[2:])
        f = self.conv(f)
        # f = self.reduction(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = self.lstm1(f)
        f = f.reshape(-1, *f.shape[2:])
        f = self.up(f)
        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = self.sigmoid(f)

        # Separate batches and sequences
        return f

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
