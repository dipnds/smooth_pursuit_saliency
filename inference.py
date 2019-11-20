import torch, tqdm
import numpy as np

import skvideo.io

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from networks.network1 import *
import networks.network1 as network
from loaders import *
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ev_set = VideoSet('eval/', sequence_length=25, augment=False, down_factor=1, output_raw=True)
eval_loader = DataLoader(ev_set, batch_size=1, shuffle=True, num_workers=1)

model = torch.load('best_network1_772.model')
writer = skvideo.io.FFmpegWriter("inference.mp4", inputdict={'-r': '3'})
with torch.no_grad():
    batching = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
    for batch_i, data in batching:
        ip = data['ip'][:, :, :, :, :].to(device)
        op = data['op'][:, :, 0:1, :, :].to(device)

        pred = model(ip)
        pred = pred.float()
        ip = np.squeeze(ip.cpu().numpy())
        ip = (data['raw'].squeeze().numpy()).astype(np.uint8)
        op = (np.squeeze(np.repeat(op.cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)
        pred = (np.squeeze(np.repeat(pred.cpu().numpy(), 3, axis=2)) * 255).astype(np.uint8)

        video = np.transpose(np.concatenate((ip, pred, op), axis=3), (0, 2, 3, 1))
        for index, frame in enumerate(video):
            if ip[index].sum() != 0:
                writer.writeFrame(frame)

writer.close()


        

