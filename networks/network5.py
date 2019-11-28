import torch.nn as nn
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.inSize = (144, 256)
        self.outSize = (288, 512)
        self.downsample = nn.Upsample(size=self.inSize, mode='bicubic')
        self.upsample = nn.Upsample(size=self.outSize, mode='bicubic')


        # down
        self.conv1 = nn.Sequential(
                       nn.Conv3d(3, 16, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        self.pool1 = nn.MaxPool3d((2,4,4),return_indices=True) # 12, 36, 64
        self.conv2 = nn.Sequential(
                       nn.Conv3d(16, 64, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        self.pool2 = nn.MaxPool3d((2,4,4), return_indices=True) # 6, 9, 16
        # self.conv3 = nn.Sequential(
        #                nn.Conv3d(32, 128, 3, padding=1, bias=True),
        #                nn.ReLU(inplace=True)
        #                )
        # self.pool3 = nn.MaxPool3d(2, return_indices=True) # 2
        # self.conv4 = nn.Sequential(
        #                nn.Conv3d(128, 256, 3, padding=1, bias=True),
        #                nn.ReLU(inplace=True)
        #                )
        # self.pool4 = nn.MaxPool3d((1,2,2), return_indices=True) # 18,32
        self.same = nn.Sequential(
                       nn.Conv3d(64, 64, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        
        # up
        # self.unpool4 = nn.MaxUnpool3d((1,2,2))
        # self.upconv4 = nn.Sequential(
        #                nn.Conv3d(256, 128, 3, padding=1, bias=True),
        #                nn.ReLU(inplace=True)
        #                )
        # self.unpool3 = nn.MaxUnpool3d(2)
        # self.upconv3 = nn.Sequential(
        #                nn.Conv3d(128, 32, 3, padding=1, bias=True),
        #                nn.ReLU(inplace=True)
        #                )
        self.unpool2 = nn.MaxUnpool3d((2,4,4))
        self.upconv2 = nn.Sequential(
                       nn.Conv3d(64, 16, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        self.unpool1 = nn.MaxUnpool3d((2,4,4))
        self.upconv1 = nn.Sequential(
                       nn.Conv3d(16, 1, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)
                       )
        
        #self.final = nn.Conv3d(16, 1, 3, padding=1, bias=True)
        
    def forward(self, ip):
        # Combine batches and sequences
        f = ip.reshape(-1, *ip.shape[2:])
        f = self.downsample(f)
        f = f.reshape(*ip.shape[:3], *self.inSize)
        f = f.permute(0,2,1,3,4) # ip.view(-1, *ip.shape[2:])

        f = self.conv1(f); f,ind1 = self.pool1(f)
        f = self.conv2(f); f,ind2 = self.pool2(f)
        # f = self.conv3(f); f,ind3 = self.pool3(f)
        # f = self.conv4(f); f,ind4 = self.pool4(f)
        
        f = self.same(f)
        
        # f = self.unpool4(f,ind4); f = self.upconv4(f)
        # f = self.unpool3(f,ind3); f = self.upconv3(f)
        f = self.unpool2(f,ind2); f = self.upconv2(f)
        f = self.unpool1(f,ind1); f = self.upconv1(f)
        
        #f = self.final(f)
        
        f = f.permute(0,2,1,3,4)
        f = f.reshape(-1, 1, *self.inSize)
        f = self.upsample(f)
        f = f.reshape(*ip.shape[:2], 1, *self.outSize)
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
