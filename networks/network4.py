import torch.nn as nn
from torchvision.models import resnet18
from .convlstm import BConvLSTM

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.outSize = (288, 512)
        self.upsample = nn.Upsample(size=self.outSize, mode='bicubic', align_corners=False)

        self.resnet = resnet18(pretrained=True)
        for child in list(self.resnet.children())[:-2]:
            for param in child.parameters():
                param.requires_grad = False
        self.down = nn.Sequential(*list(self.resnet.children())[:-2]) # 9, 16

        # self.reduction = nn.Sequential(
        #         nn.Conv2d(512, 128, 1, bias=True),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(inplace=True)
        # )

        self.lstm1 = BConvLSTM(input_size=(9, 16), input_dim=512, hidden_dim=[512], kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.up1 = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2,mode='bicubic'), # 18, 32
                nn.Conv2d(256, 128, 3, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
                )

        self.lstm2a = BConvLSTM(input_size=(9, 16), input_dim=128, hidden_dim=[128], kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.lstm2b = BConvLSTM(input_size=(9, 16), input_dim=128, hidden_dim=[128], kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.up2a = nn.Sequential(
                nn.Conv2d(128, 32, 3, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic', align_corners=False), # 72, 128
                nn.Conv2d(32, 8, 3, padding=1, bias=True),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4,mode='bicubic', align_corners=False), # 72, 128
                nn.Conv2d(8, 1, 3, padding=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Upsample(scale_factor=4,mode='bicubic', align_corners=False), # 288, 512
                )

        self.up2b = nn.Sequential(
                nn.Conv2d(128, 32, 3, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bicubic', align_corners=False), # 72, 128
                nn.Conv2d(32, 8, 3, padding=1, bias=True),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4,mode='bicubic', align_corners=False), # 72, 128
                nn.Conv2d(8, 1, 3, padding=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Upsample(scale_factor=4,mode='bicubic', align_corners=False), # 288, 512
                )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, ip):
        # Combine batches and sequences
        f = ip.reshape(-1, *ip.shape[2:])
        f = self.down(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = self.lstm1(f)
        f = f.reshape(-1, *f.shape[2:])
        f = self.up1(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        fa = self.lstm2a(f)
        fa = fa.reshape(-1, *fa.shape[2:])
        fb = self.lstm2b(f)
        fb = fb.reshape(-1, *fb.shape[2:])

        fa = self.up2a(fa)
        fa = fa.reshape(*ip.shape[:2], *fa.shape[-3:])
        fb = self.up2b(fb)
        fb = fb.reshape(*ip.shape[:2], *fb.shape[-3:])
        fa = self.sigmoid(fa)
        fb = self.sigmoid(fb)

        # Separate batches and sequences
        return fa, fb

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
