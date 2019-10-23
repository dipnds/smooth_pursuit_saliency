import torch, tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from networks.network0 import Net, NaiveLoss
from dataprep import Prep

batch_size = 2
log_nth = 10
down_factor = 1
epochs = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tr_set = Prep('train/', sequence_length=100, augment=False, down_factor=down_factor)
ev_set = Prep('eval/', sequence_length=100, augment=False, down_factor=down_factor)
eval_loader = DataLoader(ev_set, batch_size=batch_size, shuffle=True, num_workers=1)
train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=1)

writer = SummaryWriter()

def train_net(n_epochs):
    model.train()
    for epoch in range(n_epochs):
        # train
        train_batching = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        tr_loss = []
        for batch_i, data in train_batching:
            ip = data['ip'].to(device)
            # op[0] is all saliency
            op = data['op'][:, :, 0:1, :, :].to(device)

            optimizer.zero_grad()
            pred = model(ip)
            pred = pred.float().to(device)
            loss = criterion(op, pred)
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item()/batch_size)
            if (batch_i+1) % log_nth == 0:
                train_batching.set_description(f'Train E: {epoch+1}, B: {batch_i+1}, L:{tr_loss[-1]:.2E}')

        writer.add_scalar('Loss/train', np.mean(tr_loss), epoch)
        torch.save(model, "descriptor.model")
        # eval
        model.eval()
        with torch.no_grad():
            eval_batching = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
            ev_loss = []
            for batch_i, data in eval_batching:
                ip = data['ip'].to(device)
                # op[0] is all saliency
                op = data['op'][:, :, 0:1, :, :].to(device)

                pred = model(ip)
                pred = pred.float().to(device)
                loss = criterion(op, pred)
                ev_loss.append(loss.item()/batch_size)
        
            loss = np.mean(ev_loss)
            print(f'\nEval E: {epoch+1}, L: {loss:.2E}\n')
            writer.add_scalar('Loss/eval', np.mean(loss), epoch)
        
        model.train()
        scheduler.step(loss)
        writer.flush()
        
    print('fin.')
    
model = Net().to(device)
criterion = NaiveLoss(downsample_dims=(9, 16)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
train_net(epochs)
