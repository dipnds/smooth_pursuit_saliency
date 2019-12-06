import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from .convlstm import BConvLSTM

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.inSize = (25, 112, 112) # time, h, w
        self.downsample = nn.Upsample(size=self.inSize, mode='trilinear')
        self.outSize = (25, 18, 32)
        self.upsample = nn.Upsample(size=self.outSize, mode='trilinear')

        self.resnet3d = r2plus1d_18(pretrained=True)
        for child in list(self.resnet3d.children())[:-2]:
            for param in child.parameters():
                param.requires_grad = False
        self.conv = nn.Sequential(*list(self.resnet3d.children())[:-2]) # 9, 16

        self.lstm = BConvLSTM(input_size=(7,7), input_dim=512, hidden_dim=[512], kernel_size=(3, 3), num_layers=1, batch_first=True)
        
        self.up = nn.Sequential(
                nn.Conv2d(512, 128, 3, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'), # 18, 32
                nn.Conv2d(128, 32, 3, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(32, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16,track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4,mode='bicubic'), # 72, 128
                nn.Conv2d(16, 4, 3, padding=1, bias=True),
                nn.BatchNorm2d(4,track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 72, 128
                nn.Conv2d(4, 1, 3, padding=1, bias=True),
                nn.BatchNorm2d(1,track_running_stats=True),
                nn.Upsample(scale_factor=2,mode='bicubic'), # 288, 512
                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ip):
        # Combine batches and sequences
        f = ip.permute((0,2,1,3,4))
        f = self.downsample(f)
        f = self.conv(f)

        f = f.permute((0,2,1,3,4))
        f = self.lstm(f)
        f = f.permute((0,2,1,3,4))
        f = self.upsample(f)
        f = f.permute((0,2,1,3,4))
        f = f.reshape(-1, *f.shape[-3:])
        f = self.up(f)
        f = f.reshape(*ip.shape[0:2], *f.shape[-3:])
        f = self.sigmoid(f)

        # Separate batches and sequences
        return f

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
