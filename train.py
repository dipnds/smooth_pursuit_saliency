from dataprep import prep
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import cuda
from torch import save as torchsave
from network0 import net
import torch.nn as nn
import numpy as np
from datetime import datetime as dt

batch_size = 2
log_nth = 10
down_factor = 1
epochs = 20

# device = torch.device("cuda:0" if cuda.is_available() else "cpu")
device = torch.device('cpu')
tr_set = prep('train/',augment=False,down_factor=down_factor)
ev_set = prep('eval/',augment=False,down_factor=down_factor)
eval_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=True,num_workers=1)
train_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=1)

def train_net(n_epochs):

    model.train()
    tr_loss = []; ev_loss = []
    s = dt.now().strftime('%dth%b_%H_%M')
    
    for epoch in range(n_epochs):
        
        # train
        for batch_i, data in enumerate(train_loader):
            ip = data['ip']; ip = ip.to(device)
            op = data['op']; op = op.to(device)

            optimizer.zero_grad()
            pred = model(ip); pred = pred.double().to(device)
            loss = criterion(op, pred)
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())

            if (batch_i+1) % log_nth == 0:
                print('Train E: {}, B: {}, L: {}'.format(epoch+1,batch_i+1,(tr_loss[-1])/batch_size))
        torchsave(model, "descriptor.model") # cannot resume training if saved like this
        np.save(s+'_tr',np.array(tr_loss))
        
        # eval
        model.eval()
        with torch.no_grad():
            for batch_i, data in enumerate(eval_loader):
                ip = data['ip']; ip = ip.to(device)
                op = data['op']; op = op.to(device)

                pred = model(ip); pred = pred.double().to(device)
                loss = criterion(op, pred)
                ev_loss.append(loss.item())
        
        temp_loss = (np.mean(ev_loss[-len(eval_loader):]))/batch_size
        print('Eval E: {}, L: {}'.format(epoch+1,temp_loss))
        
        model.train()
        scheduler.step(temp_loss)
        np.save(s+'_ev',np.array(ev_loss))
        
    print('fin.')
    
model = net(); model = model.to(device)
criterion = nn.MSELoss(reduction='mean') # dummy, MSELoss won't work
optimizer = optim.Adam(model.parameters(), lr=2e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
train_net(epochs)
