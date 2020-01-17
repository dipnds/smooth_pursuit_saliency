import torch.nn as nn
from torchvision.models import resnet18
from .convlstm import ConvBLSTM

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
        #         nn.Conv2d(512, 128, 1, bias=True),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(inplace=True)
        # )

        self.lstm1 = ConvBLSTM(in_channels=512, hidden_channels=512, kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.lstm2 = ConvBLSTM(in_channels=512, hidden_channels=128, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.att = nn.Sequential(
                nn.Flatten(),
                nn.Linear(18432,2048,bias=True),
                nn.BatchNorm1d(2048, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Linear(2048,256,bias=True),
                nn.BatchNorm1d(256, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Linear(256,32,bias=True),
                nn.BatchNorm1d(32, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Linear(32,4,bias=True),
                nn.BatchNorm1d(4, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Linear(4,1,bias=True),
                nn.BatchNorm1d(1, track_running_stats=True)
                )

        self.up = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1, bias=True),
                nn.BatchNorm2d(256, track_running_stats=True),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'), # 18, 32
                nn.Conv2d(256, 128, 3, padding=1, bias=True),
                nn.BatchNorm2d(128, track_running_stats=True),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(128, 64, 3, padding=1, bias=True),
                nn.BatchNorm2d(64, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(64, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4,mode='bicubic'), # 72, 128
                nn.Conv2d(16, 1, 3, padding=1, bias=True),
                nn.BatchNorm2d(1, track_running_stats=True),
                nn.Upsample(scale_factor=4,mode='bicubic'), # 288, 512
                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ip):
        # Combine batches and sequences
        f = ip.reshape(-1, *ip.shape[2:])
        f = self.conv(f)
        # f = self.reduction(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        
        c = self.lstm2(f)
        c = c.reshape(-1, *c.shape[2:])
        c = self.att(c)
        c = self.sigmoid(c)

        f = self.lstm1(f)
        f = f.reshape(-1, *f.shape[2:])
        f = self.up(f)
        f = self.sigmoid(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        c = c.reshape(*ip.shape[:2], 1, 1, 1)

        f = f * c

        # Separate batches and sequences
        return f, c

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
