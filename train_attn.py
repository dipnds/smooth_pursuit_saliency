import torch, tqdm
import numpy as np
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from networks.network13 import *
import networks.network13 as network
from networks.losses import *
from loaders import *

from metrics import NSS
import matplotlib.pyplot as plt

batch_size = 4
log_nth = 10
down_factor = 1
epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tr_set = VideoSet('train/', sequence_length=24, augment=True, down_factor=down_factor)
ev_set = VideoSet('eval/', sequence_length=24, augment=False, down_factor=down_factor)
eval_loader = DataLoader(ev_set, batch_size=batch_size, shuffle=True, num_workers=5)
train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=5)
# tr_set.check(); ev_set.check()

name = network.__name__.split('.')[1]

writer = SummaryWriter(comment=name)
# torch.autograd.set_detect_anomaly(True)

def train(model, epochNum):
    criterion = LossSequence(); crit = nn.BCELoss()
    tr_loss_fix = []; tr_loss_sp = []; metric = []; tr_loss_attn = []
    train_batching = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_i, data in train_batching:
        ip = data['ip'][:, :, :, :, :].to(device)
        # op[0] is all saliency
        op = data['op'][:, :, :, :, :].to(device)

        optimizer.zero_grad()
        pred_sp, attn = model(ip) # pred_fix,

#        fig, ax = plt.subplots(1,3)
#        ax[0].imshow(pred_sp[0, 0, 0, :, :].detach().cpu(), vmin=0, vmax=1)
#        ax[1].imshow(pred_sp[0, 12, 0, :, :].detach().cpu(), vmin=0, vmax=1)
#        ax[2].imshow(pred_sp[0, 24, 0, :, :].detach().cpu(), vmin=0, vmax=1)
#        plt.show()
#        exit()

#         loss_fix = criterion(op[:,:,1:2,:,:], pred_fix[:,:,0:1,:,:])
        loss_sp = criterion(op[:,:,1:2,:,:], pred_sp[:,:,0:1,:,:])

        gt_attn = op[:,:,1:2,:,:].sum(dim=(2, 3, 4),keepdim=True).sign()
        loss_attn = crit(attn,gt_attn)

        loss_sp.backward(retain_graph=True) # (loss_fix+loss_sp).backward()
        loss_attn.backward()
        optimizer.step()
#         tr_loss_fix.append(loss_fix.detach().item())
        tr_loss_sp.append(loss_sp.detach().item())
        tr_loss_attn.append(loss_attn.detach().item())
        # with torch.no_grad():
            # metric.append(NSS(op,pred))
        if (batch_i+1) % log_nth == 0:
            train_batching.set_description(f'Train E: {epoch+1}, B: {batch_i+1}, L:{np.mean(tr_loss_sp):.2E}')#+np.mean(tr_loss_fix)

    ipImg = ip[0,0,:,:,:].detach()
    ipImg = (ipImg - ipImg.min()) / (ipImg.max() - ipImg.min())
#    writer.add_scalar('LossFix/fix_train', np.mean(tr_loss_fix), epoch)
    writer.add_scalar('LossSP/sp_train', np.mean(tr_loss_sp), epoch)
    writer.add_scalar('Attention/train', np.mean(tr_loss_attn), epoch)
#    writer.add_scalar('Loss/train', np.mean(tr_loss_fix) + np.mean(tr_loss_sp), epoch)
    writer.add_image('Input/train', ipImg,global_step=epoch,dataformats='CHW')
#    writer.add_image('TargetFix/fix_train',op[0,12,0:1,:,:].detach(),global_step=epoch,dataformats='CHW')
    writer.add_image('TargetSP/sp_train',op[0,12,1:2,:,:].detach(),global_step=epoch,dataformats='CHW')
#    writer.add_image('PredictionFix/fix_train',pred_fix[0,12,0:1,:,:].detach(),global_step=epoch,dataformats='CHW')
    writer.add_image('PredictionSP/sp_train',pred_sp[0,12,0:1,:,:].detach(),global_step=epoch,dataformats='CHW')

    torch.save(model, f"{name}.model")

def evaluate(model, epoch, best_eval, scheduler):
    criterion = LossSequence(); crit = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        ev_loss_fix = []; ev_loss_sp = []; metric = []; ev_loss_attn = []
        eval_batching = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
        for batch_i, data in eval_batching:
            ip = data['ip'][:, :, :, :, :].to(device)
            # op[0] is all saliency
            op = data['op'][:, :, :, :, :].to(device)

            pred_sp, attn = model(ip) # pred_fix,
#            loss_fix = criterion(op[:,:,1:2,:,:], pred_fix[:,:,0:1,:,:])
            loss_sp = criterion(op[:,:,1:2,:,:], pred_sp[:,:,0:1,:,:])

            gt_attn = op[:,:,1:2,:,:].sum(dim=(2, 3, 4),keepdim=True).sign()
            loss_attn = crit(attn,gt_attn)
#            ev_loss_fix.append(loss_fix.detach().item())
            ev_loss_sp.append(loss_sp.detach().item())
            ev_loss_attn.append(loss_attn.detach().item())
            # metric.append(NSS(op,pred))

        loss = np.mean(ev_loss_sp)# + np.mean(ev_loss_fix)
        ipImg = ip[0,0,:,:,:]
        ipImg = (ipImg - ipImg.min()) / (ipImg.max() - ipImg.min())
        print(f'\nEval E: {epoch+1}, L: {loss:.2E}\n')
        if best_eval is None or loss < best_eval:
            best_eval = loss
            torch.save(model, f"best_{name}.model")

#        writer.add_scalar('LossFix/fix_eval', np.mean(ev_loss_fix), epoch)
        writer.add_scalar('LossSP/sp_eval', np.mean(ev_loss_sp), epoch)
        writer.add_scalar('Attention/eval', np.mean(ev_loss_attn), epoch)
#        writer.add_scalar('Loss/eval', np.mean(ev_loss_fix) + np.mean(ev_loss_sp), epoch)
        writer.add_image('Input/eval', ipImg,global_step=epoch,dataformats='CHW')
#        writer.add_image('TargetFix/fix_eval',op[0,12,0:1,:,:].detach(),global_step=epoch,dataformats='CHW')
        writer.add_image('TargetSP/sp_eval',op[0,12,1:2,:,:].detach(),global_step=epoch,dataformats='CHW')
#        writer.add_image('PredictionFix/fix_eval',pred_fix[0,12,0:1,:,:].detach(),global_step=epoch,dataformats='CHW')
        writer.add_image('PredictionSP/sp_eval',pred_sp[0,12,0:1,:,:].detach(),global_step=epoch,dataformats='CHW')
    
    scheduler.step()
    return best_eval
    
model = Net().to(device)
# model = torch.load('best_network1_793.model')
optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

best_eval = None
for epoch in range(epochs):
    train(model, epoch)
    best_eval = evaluate(model, epoch, best_eval, scheduler)
