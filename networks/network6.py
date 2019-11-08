import torch.nn as nn
from torchvision.models import resnet18
# import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.outSize = (288, 512)
        self.upsample = nn.Upsample(size=self.outSize, mode='bicubic')

        self.resnet = resnet18(pretrained=True)
        for child in self.resnet.children():
            for param in child.parameters():
                param.requires_grad = False
        self.conv = nn.Sequential(*list(self.resnet.children())[:-2]) # 9, 16

        # up        
        self.up = nn.Sequential(
                        nn.Conv2d(512, 128, 3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2,mode='bicubic'), # 18, 32
                        nn.Conv2d(128, 32, 3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=4,mode='bicubic'), # 72, 128
                        nn.Conv2d(32, 8, 3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=4,mode='bicubic'), # 288, 512
                        )
        self.up3d = nn.Sequential(
                       nn.Conv3d(8, 1, 3, padding=1, bias=True)
                       )
                
    def forward(self, ip):

        # Combine batches and sequences
        f = ip.reshape(-1, *ip.shape[2:])
        f = self.conv(f)
        f = self.up(f)

        f = f.reshape(*ip.shape[:2], *f.shape[-3:])
        f = f.permute(0,2,1,3,4)
        f = self.up3d(f)
        f = f.permute(0,2,1,3,4)
        # Separate batches and sequences
        return f

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class LossFrame(nn.Module):
    def __init__(self):
        super(LossFrame, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        #self.criterion = nn.KLDivLoss(reduction='mean')
        self.softmax = nn.Softmax(dim=0)

    def forward(self, expected, predicted):
        expected = expected.view(-1, *expected.shape[2:])
        predicted = predicted.view(-1, *predicted.shape[2:])  
        #print((expected-predicted).abs().sum())
        expected = self.softmax(expected)
        predicted = (self.softmax(predicted)).log() # see KLDiv
        loss = self.criterion(predicted, expected)
        return loss
        
class LossSequence(nn.Module):
    def __init__(self):
        super(LossSequence, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        #self.criterion = nn.KLDivLoss(reduction='mean')
#        self.softmax = nn.Softmax()


    def softmax(self, i):
        return i.exp() / i.exp().sum(dim=(1, 2, 3, 4),keepdim=True)
        
    def forward(self, expected, predicted):
#        fig, ax = plt.subplots(1,2)
#        ax[0].imshow(expected[0, 0, 0, :, :].cpu().detach().numpy(), cmap='bone')
#        ax[1].imshow(es[0, 0, 0, :, :].cpu().detach().numpy(), cmap='bone')
#        plt.show()
        predicted = (self.softmax(predicted)).log() # see KLDiv
        expected = self.softmax(expected)

        loss = self.criterion(predicted, expected)
        return loss
