import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 6, stride=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 6, stride=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 144, bias=True)
        )                             
        
    def forward(self, ip):
        # Combine batches and sequences
        f = ip.view(-1, *ip.shape[2:])
        conv_samples = f.shape[0]
        f = self.conv(f)
        f = self.fc(f.view(conv_samples, -1))
        # Separate batches and sequences
        return f.view(*ip.shape[:2], -1)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class NaiveLoss(nn.Module):
    def __init__(self, downsample_dims):
        super(NaiveLoss, self).__init__()
        self.dims = downsample_dims
        self.downsample = nn.Upsample(size=downsample_dims, mode='bilinear', align_corners=False)
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, expected, predicted):
        expected = self.downsample(expected.view(-1, *expected.shape[2:]))
        predicted = predicted.view(expected.shape)
        loss = self.criterion(predicted, expected)
        return loss
        