import torch.nn as nn
import matplotlib.pyplot as plt
import torch

class LossFrame(nn.Module):
    def __init__(self):
        super(LossFrame, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def softmax(self, i):
        return i / i.sum(dim=(1, 2, 3),keepdim=True)
    
    def forward(self, expected, predicted):
        expected = expected.view(-1, *expected.shape[2:])
        predicted = predicted.view(-1, *predicted.shape[2:])
        # fig, ax = plt.subplots(1,2)

        # ax[1].imshow(predicted[0, 0, :, :].cpu().detach().numpy(), cmap='bone')#, vmin=0,vmax=.00001)
        # ax[0].imshow(self.softmax(predicted)[0, 0, :, :].cpu().detach().numpy(), cmap='bone')#, vmin=0,vmax=.00001)
        # plt.show()
        # print(predicted[0, :, :, :].sum())
        predicted = (self.softmax(predicted))
        predicted = predicted.log() # see KLDiv        
        expected = self.softmax(expected)

        loss = self.criterion(predicted, expected)
        return loss
        
class LossSequence(nn.Module):
    def __init__(self):
        super(LossSequence, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        # import numpy as np
        # self.dummy = np.append(np.ones((2,25,1,288,512)), np.zeros((3,25,1,288,512)),axis=0)
        # self.dummy = torch.from_numpy(self.dummy)
        # self.softmax(self.dummy)
        # exit()

    def softmax(self, i):
        return i / i.sum(dim=(1, 2, 3, 4),keepdim=True)
        
    def forward(self, expected, predicted):
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(expected[0, 0, 0, :, :].cpu().detach().numpy(), cmap='bone', vmin=0,vmax=1)
        # ax[1].imshow(predicted[0, 0, 0, :, :].cpu().detach().numpy(), cmap='bone', vmin=0,vmax=1)
        # plt.show()

        predicted = (self.softmax(predicted+1e-8)).log() # see KLDiv
        expected = self.softmax(expected+1e-8)

        loss = self.criterion(predicted, expected)
        return loss