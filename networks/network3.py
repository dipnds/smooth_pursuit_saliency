import torch.nn as nn
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
                       nn.Conv3d(3, 16, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        self.pool1 = nn.MaxPool3d(2,return_indices=True)
        self.conv2 = nn.Sequential(
                       nn.Conv3d(16, 128, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        self.pool2 = nn.MaxPool3d(2, return_indices=True)
        self.conv3 = nn.Sequential(
                       nn.Conv3d(128, 128, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        self.unpool2 = nn.MaxUnpool3d(2)
        self.conv4 = nn.Sequential(
                       nn.Conv3d(128, 16, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        self.unpool1 = nn.MaxUnpool3d(2)
        self.conv5 = nn.Conv3d(16, 1, 3, padding=1, bias=True)
        
    def forward(self, ip):
        # Combine batches and sequences
        f = ip.permute(0,2,1,3,4) # ip.view(-1, *ip.shape[2:])
        f = self.conv1(f); f,ind1 = self.pool1(f)
        f = self.conv2(f); f,ind2 = self.pool2(f)
        f = self.conv3(f); f = self.unpool2(f,ind2)
        f = self.conv4(f); f = self.unpool1(f,ind1)
        f = self.conv5(f)
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
