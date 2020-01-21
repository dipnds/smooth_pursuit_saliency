import torch, tqdm
import numpy as np

import skvideo.io

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from loaders import *
from metrics import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ev_set = VideoSet('mikhail/', sequence_length=24, augment=False, down_factor=1, output_raw=True)
ev_set = VideoSet('gazecom/', sequence_length=24, augment=False, down_factor=1, output_raw=True)

eval_loader = DataLoader(ev_set, batch_size=1, shuffle=False, num_workers=0)

nss_fix_fix = []
nss_sp_sp = []

nss_fix_sp = []
nss_sp_fix = []

aucb_fix = []
aucb_sp = []

sim_fix = []
sim_sp = []

cc_fix = []
cc_sp = []

points_fix = []
points_sp = []
op_fix = []
op_sp = []
preds = []

currentName = None

def metricsVideo():
    global op_fix, op_sp, points_fix, points_sp, preds, nss_fix_fix, nss_sp_sp, nss_sp_fix, nss_fix_sp, aucj_fix, aucj_sp, aucb_fix, aucb_sp, sim_fix, sim_sp, cc_fix, cc_sp
    print(f'Current Video: {currentName}')
    points_fix = torch.cat(points_fix, dim=1)
    points_sp = torch.cat(points_sp, dim=1)
    op_fix = torch.cat(op_fix, dim=1)
    op_sp = torch.cat(op_sp, dim=1)
    preds = torch.cat(preds, dim=1)
    # print(points_sp.shape, preds.shape)

    nss_fix_fix.append(NSS(preds[:, :, :, :], points_fix))
    nss_sp_sp.append(NSS(preds[:, :, :, :], points_sp))

    aucb_fix.append(auc_Borji(preds[:, :, :, :], points_fix, 10))
    aucb_sp.append(auc_Borji(preds[:, :, :, :], points_sp, 10))

    sim_fix.append(sim(preds[:, :, :, :], op_fix))
    sim_sp.append(sim(preds[:, :, :, :], op_sp))

    cc_fix.append(cc(preds[:, :, :, :], op_fix))
    cc_sp.append(cc(preds[:, :, :, :], op_sp))
    # nss_fix_sp.append(NSS(preds[:, :, 0, :, :], points_sp))
    # nss_sp_fix.append(NSS(preds[:, :, 0, :, :], points_fix))

    # nss_fix_fix_arr = np.array(nss_fix_fix)
    # nss_fix_fix_arr = nss_fix_fix_arr[~np.isnan(nss_fix_fix_arr)]

    # nss_sp_sp_arr = np.array(nss_sp_sp)
    # nss_sp_sp_arr = nss_sp_sp_arr[~np.isnan(nss_sp_sp_arr)]

    print(f'NSS Fix: {np.nanmean(nss_fix_fix):2E}  Std: {np.nanstd(nss_fix_fix):2E}  Min: {np.nanmin(nss_fix_fix):2E}  Max: {np.nanmax(nss_fix_fix):2E}')
    print(f'NSS SP: {np.nanmean(nss_sp_sp):2E}  Std: {np.nanstd(nss_sp_sp):2E}  Min: {np.nanmin(nss_sp_sp):2E}  Max: {np.nanmax(nss_sp_sp):2E}')

    print(f'AUC Borji Fix: {np.nanmean(aucb_fix):2E}  Std: {np.nanstd(aucb_fix):2E}  Min: {np.nanmin(aucb_fix):2E}  Max: {np.nanmax(aucb_fix):2E}')
    print(f'AUC Borji SP: {np.nanmean(aucb_sp):2E}  Std: {np.nanstd(aucb_sp):2E}  Min: {np.nanmin(aucb_sp):2E}  Max: {np.nanmax(aucb_sp):2E}')

    print(f'SIM Fix: {np.nanmean(sim_fix):2E}  Std: {np.nanstd(sim_fix):2E}  Min: {np.nanmin(sim_fix):2E}  Max: {np.nanmax(sim_fix):2E}')
    print(f'SIM SP: {np.nanmean(sim_sp):2E}  Std: {np.nanstd(sim_sp):2E}  Min: {np.nanmin(sim_sp):2E}  Max: {np.nanmax(sim_sp):2E}')

    print(f'CC Fix: {np.nanmean(cc_fix):2E}  Std: {np.nanstd(cc_fix):2E}  Min: {np.nanmin(cc_fix):2E}  Max: {np.nanmax(cc_fix):2E}')
    print(f'CC SP: {np.nanmean(cc_sp):2E}  Std: {np.nanstd(cc_sp):2E}  Min: {np.nanmin(cc_sp):2E}  Max: {np.nanmax(cc_sp):2E}')

    points_fix = []
    points_sp = []
    op_fix = []
    op_sp = []
    preds = []

model = torch.load('bests/best_network4_l1reg.model')
# writer = skvideo.io.FFmpegWriter("inf_sp.mp4", outputdict={'-r': '25', '-b': '300000000'})
with torch.no_grad():
    batching = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
    for batch_i, data in batching:
        if currentName is None:
            currentName = data['file']
        if data['file'] != currentName:
            metricsVideo()
            currentName = data['file']

        ip = data['ip'][:, :, :, :, :].to(device)
        op_fix.append(data['op'][:, :, 1:2, :, :].cpu())
        op_sp.append(data['op'][:, :, 2:3, :, :].cpu())
        points_fix.append(data['pointMap'][:, :, :, :, 1].cpu())
        points_sp.append(data['pointMap'][:, :, :, :, 2].cpu())

        pred_sp, _ = model(ip)
        preds.append(pred_sp[:, :, 0, :, :].detach().cpu())

        # if data['pointMap'][:, :, :, :, 2].cpu().sum() > 1:
        #     print(data['pointMap'][:, :, :, :, 2].cpu().sum())
        #     torch.save(data['op'][:, :, 2:3, :, :], 'eval_op.pt')
        #     torch.save(pred_sp, 'eval_pred.pt')
        #     torch.save(data['pointMap'][:, :, :, :, 2].cpu(), 'eval_points.pt')
        #     exit()

        # ip = np.transpose((data['raw'].squeeze().numpy()).astype(np.uint8), (0, 3, 1, 2))
        # # op_fix = (np.squeeze(np.repeat(data['op'][:, :, 1:2, :, :].cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)
        # # pred_fix = (np.squeeze(np.repeat(pred_fix.cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)
        # sp_out = (np.squeeze(np.repeat(data['op'][:, :, 1:2, :, :].cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)
        # pred_sp = (np.squeeze(np.repeat(pred_sp.cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)

        # video = np.transpose(np.concatenate((ip, pred_sp, sp_out), axis=3), (0, 2, 3, 1))
        # for index, frame in enumerate(video):
        #     if ip[index].sum() != 0:
        #         writer.writeFrame(frame)

    metricsVideo()


# writer.close()
# print(nss_fix_fix)
# print(nss_sp_sp)
