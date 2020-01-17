import torch.nn as nn
from torchvision.models import resnet18
from .convlstm import ConvBLSTM

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.outSize = (288, 512)
        # self.upsample = nn.Upsample(size=self.outSize, mode='bicubic')

        # self.resnet = resnet18(pretrained=True)
        # for child in list(self.resnet.children()):
        #     for param in child.parameters():
        #         param.requires_grad = False
        # self.down = nn.Sequential(*list(self.resnet.children())[:-3]) # 9, 16

        
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bicubic'),
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 64, 3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2)
        )

        self.temporal = nn.Sequential(
            ConvBLSTM(in_channels=256, hidden_channels=64, kernel_size=(3, 3), num_layers=1, batch_first=True),
            ConvBLSTM(in_channels=64, hidden_channels=16, kernel_size=(3, 3), num_layers=1, batch_first=True)
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bicubic'),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.BatchNorm2d(1, track_running_stats=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, ip):
        # Combine batches and sequences
        f = ip.reshape(-1, *ip.shape[2:])
        f = self.down(f)
        # print(f.shape)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = self.temporal(f)
        f = f.reshape(-1, *f.shape[2:])
        f = self.up(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = self.sigmoid(f)

        # Separate batches and sequences
        return f

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
