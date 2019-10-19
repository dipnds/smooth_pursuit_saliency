import torch.nn as nn

class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv2d(2, 8, 6, stride=6, padding=0, bias=True),
                                nn.ReLU(inplace=True)
                                )
        self.fc = nn.Sequential(
                                nn.Linear(17*5*16, 128, bias=True)
                                )                             
        
    def forward(self, ip):
        
        f = ip.squeeze(); print(f.shape)
        ip = self.conv(ip)#; ip = ip.view(-1, 17*5*16); ip = self.fc(ip)
        return ip

    @property
    def is_cuda(self):
        
        return next(self.parameters()).is_cuda