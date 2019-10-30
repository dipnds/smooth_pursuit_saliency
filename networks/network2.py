import torch.nn as nn
from torchvision.models import vgg11_bn

import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg = vgg11_bn(pretrained=True)
        print(self.vgg)
        #self.conv = nn.Sequential(*list(self.resnet.children())[:-3])
        self.conv = nn.Sequential(
                       nn.Conv2d(512, 128, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True),
                       nn.Upsample(size=(9, 16), mode='bilinear', align_corners=False),
                       nn.Conv2d(128, 16, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True),
                       nn.Upsample(size=(72, 128), mode='bilinear', align_corners=False),
                       nn.Conv2d(16, 1, 3, padding=1, bias=True)
                       )
        
        #self.flat = nn.Conv2d(256, 16, 1,bias=True)
#        self.up = nn.Upsample(size=(72,128), mode='bilinear', align_corners=False)
#        self.conv2 = nn.Conv2d(16, 1, 1,bias=True)
        
    def forward(self, ip):
        # Combine batches and sequences
        f = ip.view(-1, *ip.shape[2:])
        f = self.vgg.features(f)
#        print(f.shape)
        f = self.conv(f)
        f = f.reshape(ip.shape[0],ip.shape[1],1,f.shape[2],f.shape[3])
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
