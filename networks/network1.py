import torch.nn as nn
from torchvision.models import resnet18

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.conv = nn.Sequential(*list(self.resnet.children())[:-3])
        
        self.lstm = nn.LSTM(40*256,16*9,bias=True,batch_first=True,bidirectional=True)
        self.flat = nn.Conv2d(2,1,1,bias=True)
        self.up = nn.Upsample(size=(72,128), mode='bilinear', align_corners=False)
        
    def forward(self, ip):
        # Combine batches and sequences
        f = ip.view(-1, *ip.shape[2:])
        f = self.conv(f)
        f = f.view(ip.shape[0],ip.shape[1],-1)
        f,_ = self.lstm(f)
        f = f.reshape(ip.shape[0]*ip.shape[1],2,9,16)
        f = self.up(f)
        f = self.flat(f)
        #f = f.squeeze()

        f = f.reshape(ip.shape[0],ip.shape[1],1,f.shape[2],f.shape[3])
        # Separate batches and sequences
        return f

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class LossSequence(nn.Module):
    def __init__(self):
        super(LossSequence, self).__init__()
        #self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.criterion = nn.KLDivLoss(reduction='mean')
        self.softmax = nn.Softmax(dim=0)

    def forward(self, expected, predicted):
        #print((expected-predicted).abs().sum())
        expected = self.softmax(expected); predicted = (self.softmax(predicted)).log() # see KLDiv
        loss = self.criterion(predicted, expected)
        return loss
        
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
