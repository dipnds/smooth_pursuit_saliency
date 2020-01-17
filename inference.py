import torch, tqdm
import numpy as np

import skvideo.io

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from networks.network4 import *
import networks.network4 as network
from loaders import *
from metrics import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ev_set = VideoSet('eval/', sequence_length=20, augment=False, down_factor=1, output_raw=True)
eval_loader = DataLoader(ev_set, batch_size=1, shuffle=True, num_workers=1)

nss_fix_fix = []
nss_sp_sp = []

nss_fix_sp = []
nss_sp_fix = []

model = torch.load('best_network4.model')
writer = skvideo.io.FFmpegWriter("inference.mp4", outputdict={'-r': '25', '-b': '300000000'})
with torch.no_grad():
    batching = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
    for batch_i, data in batching:
        ip = data['ip'][:, :, :, :, :].to(device)
        op_fix = data['op'][:, :, 1:2, :, :].to(device)
        op_sp = data['op'][:, :, 2:3, :, :].to(device)
        points_fix = data['pointMap'][:, :, :, :, 1].to(device)
        points_sp = data['pointMap'][:, :, :, :, 2].to(device)

        pred_fix, pred_sp = model(ip)

        nss_fix_fix.append(NSS(pred_fix[:, :, 0, :, :], points_fix))
        nss_sp_sp.append(NSS(pred_sp[:, :, 0, :, :], points_sp))
        nss_fix_sp.append(NSS(pred_fix[:, :, 0, :, :], points_sp))
        nss_sp_fix.append(NSS(pred_sp[:, :, 0, :, :], points_fix))

        ip = np.transpose((data['raw'].squeeze().numpy()).astype(np.uint8), (0, 3, 1, 2))
        op_fix = (np.squeeze(np.repeat(op_fix.cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)
        pred_fix = (np.squeeze(np.repeat(pred_fix.cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)
        op_sp = (np.squeeze(np.repeat(op_sp.cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)
        pred_sp = (np.squeeze(np.repeat(pred_sp.cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)

        fix = np.transpose(np.concatenate((ip, pred_fix, op_fix), axis=3), (0, 2, 3, 1))
        sp = np.transpose(np.concatenate((ip, pred_sp, op_sp), axis=3), (0, 2, 3, 1))
        video = np.concatenate((fix, sp), axis=1)
        for index, frame in enumerate(video):
            if ip[index].sum() != 0:
                writer.writeFrame(frame)

        # if batch_i > 20:
            # break

writer.close()

nss_fix_fix = np.array(nss_fix_fix)
nss_fix_fix = nss_fix_fix[~np.isnan(nss_fix_fix)]

nss_sp_sp = np.array(nss_sp_sp)
nss_sp_sp = nss_sp_sp[~np.isnan(nss_sp_sp)]

nss_fix_sp = np.array(nss_fix_sp)
nss_fix_sp = nss_fix_sp[~np.isnan(nss_fix_sp)]

nss_sp_fix = np.array(nss_sp_fix)
nss_sp_fix = nss_sp_fix[~np.isnan(nss_sp_fix)]

print(f'NSS Fix: {np.mean(nss_fix_fix):2E}  Std: {np.std(nss_fix_fix):2E}  Min: {np.min(nss_fix_fix):2E}  Max: {np.max(nss_fix_fix):2E}')
print(f'NSS SP: {np.mean(nss_sp_sp):2E}  Std: {np.std(nss_sp_sp):2E}  Min: {np.min(nss_sp_sp):2E}  Max: {np.max(nss_sp_sp):2E}')

print(f'NSS Fix on SP GT: {np.mean(nss_fix_sp):2E}  Std: {np.std(nss_fix_sp):2E}  Min: {np.min(nss_fix_sp):2E}  Max: {np.max(nss_fix_sp):2E}')
print(f'NSS SP on Fix GT: {np.mean(nss_sp_fix):2E}  Std: {np.std(nss_sp_fix):2E}  Min: {np.min(nss_sp_fix):2E}  Max: {np.max(nss_sp_fix):2E}')
