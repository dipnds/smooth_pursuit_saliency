import torch, tqdm
import numpy as np

import skvideo.io

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from loaders import *
from metrics import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ev_set = VideoSet('eval/', sequence_length=24, augment=False, down_factor=1, output_raw=True)
eval_loader = DataLoader(ev_set, batch_size=1, shuffle=False, num_workers=0)

nss_fix_fix = []
nss_sp_sp = []

nss_fix_sp = []
nss_sp_fix = []

points_fix = []
points_sp = []
preds = []

currentName = None

def nssVideo():
    global points_fix, points_sp, preds, nss_fix_fix, nss_sp_sp, nss_sp_fix, nss_fix_sp
    print(f'Current Video: {currentName}')
    points_fix = torch.cat(points_fix, dim=1)
    points_sp = torch.cat(points_sp, dim=1)
    preds = torch.cat(preds, dim=1)
    print(points_sp.shape, preds.shape)

    nss_fix_fix.append(NSS(preds[:, :, :, :], points_fix))
    nss_sp_sp.append(NSS(preds[:, :, :, :], points_sp))
    # nss_fix_sp.append(NSS(preds[:, :, 0, :, :], points_sp))
    # nss_sp_fix.append(NSS(preds[:, :, 0, :, :], points_fix))

    nss_fix_fix_arr = np.array(nss_fix_fix)
    nss_fix_fix_arr = nss_fix_fix_arr[~np.isnan(nss_fix_fix_arr)]

    nss_sp_sp_arr = np.array(nss_sp_sp)
    nss_sp_sp_arr = nss_sp_sp_arr[~np.isnan(nss_sp_sp_arr)]

    print(f'NSS Fix: {np.mean(nss_fix_fix_arr):2E}  Std: {np.std(nss_fix_fix_arr):2E}  Min: {np.min(nss_fix_fix_arr):2E}  Max: {np.max(nss_fix_fix_arr):2E}')
    print(f'NSS SP: {np.mean(nss_sp_sp_arr):2E}  Std: {np.std(nss_sp_sp_arr):2E}  Min: {np.min(nss_sp_sp_arr):2E}  Max: {np.max(nss_sp_sp_arr):2E}')

    points_fix = []
    points_sp = []
    preds = []

model = torch.load('bests/best_network2_sp.model')
with torch.no_grad():
    batching = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
    for batch_i, data in batching:
        if currentName is None:
            currentName = data['file']
        if data['file'] != currentName:
            currentName = data['file']
            nssVideo()


        ip = data['ip'][:, :, :, :, :].to(device)
        # op_fix.append(data['op'][:, :, 1:2, :, :].to(device))
        # op_sp.append(data['op'][:, :, 2:3, :, :].to(device))
        points_fix.append(data['pointMap'][:, :, :, :, 1].cpu())
        points_sp.append(data['pointMap'][:, :, :, :, 2].cpu())

        pred_sp = model(ip)
        preds.append(pred_sp[:, :, 0, :, :].detach().cpu())

        # if data['pointMap'][:, :, :, :, 2].cpu().sum() > 1:
        #     print(data['pointMap'][:, :, :, :, 2].cpu().sum())
        #     torch.save(data['op'][:, :, 2:3, :, :], 'eval_op.pt')
        #     torch.save(pred_sp, 'eval_pred.pt')
        #     torch.save(data['pointMap'][:, :, :, :, 2].cpu(), 'eval_points.pt')
        #     exit()

    nssVideo()



        #if batch_i > 10:
            #break

# nss_fix_fix = np.array(nss_fix_fix)
# nss_fix_fix = nss_fix_fix[~np.isnan(nss_fix_fix)]

# nss_sp_sp = np.array(nss_sp_sp)
# nss_sp_sp = nss_sp_sp[~np.isnan(nss_sp_sp)]

# nss_fix_sp = np.array(nss_fix_sp)
# nss_fix_sp = nss_fix_sp[~np.isnan(nss_fix_sp)]

# nss_sp_fix = np.array(nss_sp_fix)
# nss_sp_fix = nss_sp_fix[~np.isnan(nss_sp_fix)]

# print(f'NSS Fix: {np.mean(nss_fix_fix):2E}  Std: {np.std(nss_fix_fix):2E}  Min: {np.min(nss_fix_fix):2E}  Max: {np.max(nss_fix_fix):2E}')
# print(f'NSS SP: {np.mean(nss_sp_sp):2E}  Std: {np.std(nss_sp_sp):2E}  Min: {np.min(nss_sp_sp):2E}  Max: {np.max(nss_sp_sp):2E}')

# print(f'NSS Fix on SP GT: {np.mean(nss_fix_sp):2E}  Std: {np.std(nss_fix_sp):2E}  Min: {np.min(nss_fix_sp):2E}  Max: {np.max(nss_fix_sp):2E}')
# print(f'NSS SP on Fix GT: {np.mean(nss_sp_fix):2E}  Std: {np.std(nss_sp_fix):2E}  Min: {np.min(nss_sp_fix):2E}  Max: {np.max(nss_sp_fix):2E}')
