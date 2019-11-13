import torch, tqdm
import numpy as np
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from networks.network6 import *
import networks.network6 as network
from dataprep import Prep

from metrics import NSS
import matplotlib.pyplot as plt

batch_size = 8
log_nth = 10
down_factor = 1
epochs = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tr_set = Prep('train/', sequence_length=25, augment=True, down_factor=down_factor)
ev_set = Prep('eval/', sequence_length=25, augment=False, down_factor=down_factor)
eval_loader = DataLoader(ev_set, batch_size=batch_size, shuffle=True, num_workers=1)
train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=1)

name = network.__name__.split('.')[1]

writer = SummaryWriter(comment=name)

def train_net(n_epochs):
    model.train()
    for epoch in range(n_epochs):
        # train
        train_batching = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        tr_loss = []; metric = []
        for batch_i, data in train_batching:
            ip = data['ip'][:, :-1, :, :, :].to(device)
            # op[0] is all saliency
            op = data['op'][:, :-1, 0:1, :, :].to(device)

            optimizer.zero_grad()
            pred = model(ip)
            pred = pred.float().to(device)

            # sample = pred[0][0].squeeze()
            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(sample.detach().cpu().numpy())
            # ax[1].imshow(data['op'][0, 0, 0, :, :])
            # plt.show()
            # exit()

            loss = criterion(op, pred)
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())
            with torch.no_grad():
                metric.append(NSS(op,pred))
            if (batch_i+1) % log_nth == 0:
                train_batching.set_description(f'Train E: {epoch+1}, B: {batch_i+1}, L:{tr_loss[-1]:.2E}')

        ipImg = ip[0,0,:,:,:]
        ipImg = (ipImg - ipImg.min()) / (ipImg.max() - ipImg.min())
        writer.add_scalar('Loss/train', np.mean(tr_loss), epoch)
        writer.add_scalar('NSS/train', np.mean(metric), epoch)
        writer.add_image('Input/train', ipImg,global_step=epoch,dataformats='CHW')
        writer.add_image('Target/train',(op[0,0,:,:,:]+1)/2,global_step=epoch,dataformats='CHW')
        writer.add_image('Prediction/train',(pred[0,0,:,:,:]+1)/2,global_step=epoch,dataformats='CHW')
        torch.save(model, "descriptor.model")
        # eval
        model.eval()
        with torch.no_grad():
            eval_batching = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
            ev_loss = []; metric = []
            for batch_i, data in eval_batching:
                ip = data['ip'][:, :-1, :, :, :].to(device)
                # op[0] is all saliency
                op = data['op'][:, :-1, 0:1, :, :].to(device)

                pred = model(ip)
                pred = pred.float().to(device)
                loss = criterion(op, pred)
                ev_loss.append(loss.item())
                metric.append(NSS(op,pred))

            #loss = np.mean(ev_loss)
            ipImg = ip[0,0,:,:,:]
            ipImg = (ipImg - ipImg.min()) / (ipImg.max() - ipImg.min())
            print(f'\nEval E: {epoch+1}, L: {np.mean(ev_loss):.2E}\n')
            writer.add_scalar('Loss/eval', np.mean(ev_loss), epoch)
            writer.add_scalar('NSS/eval', np.mean(metric), epoch)
            writer.add_image('Input/eval', ipImg,global_step=epoch,dataformats='CHW')
            writer.add_image('Target/eval',(op[0,0,:,:,:]+1)/2,global_step=epoch,dataformats='CHW')
            writer.add_image('Prediction/eval',(pred[0,0,:,:,:]+1)/2,global_step=epoch,dataformats='CHW')
        
        model.train()
        scheduler.step(loss)
        
    print('fin.')
    
model = Net().to(device)
criterion = LossSequence().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
train_net(epochs)
