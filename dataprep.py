import os, random
from torch.utils.data import Dataset
from torch import from_numpy
import numpy as np

class Prep(Dataset):
    def __init__(self, path, sequence_length, augment=False, down_factor=1):
        self.path = path
        self.vid_list = os.listdir(f'{path}ip/')
        self.lab_list = os.listdir(f'{path}op/')

        self.vid_list.sort()
        self.lab_list.sort()
        
        self.sequence_length = sequence_length
        self.augment = augment
        self.down_factor = down_factor

        self.mean = np.load('frame_mean.npy')
        self.std = np.load('frame_std.npy')
        
    def __len__(self):
        return len(self.vid_list)

    def random(self):
        return bool(random.getrandbits(1))
    
    def __getitem__(self, idx):
        ip = np.load(f'{self.path}ip/{self.vid_list[idx]}')
        ip = ((ip/255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        # ip = (ip - self.mean) / self.std
        op = np.load(f'{self.path}op/{self.lab_list[idx]}')
        # assumes all videos and heatmaps are normalised
        
        if self.augment:
            if self.random():
                ip = np.flip(ip, axis=1)
                op = np.flip(op, axis=1)
            if self.random():
                ip = np.flip(ip, axis=2)
                op = np.flip(op, axis=2)
#        if self.down_factor != 1:
#            # not implemented yet
        
        return {'ip':self.prepare(ip), 'op':self.prepare(op)}

    def prepare(self, raw):
        array = np.zeros([self.sequence_length, raw.shape[1], raw.shape[2], raw.shape[3]])
        # Insert the data into the expected sequence length
        array[:raw.shape[0], :, :, :] = raw
        array = np.transpose(array, (0,3,1,2))
        array = from_numpy(array).float()
        return array
