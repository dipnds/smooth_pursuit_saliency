import os, random, torch
from torch.utils.data import Dataset
import numpy as np
import skvideo.io as skv

class NumpySet(Dataset):
    def __init__(self, path, sequence_length, augment=False, down_factor=1, output_raw=False):
        self.path = path
        self.vid_list = os.listdir(f'{path}ip/')
        self.lab_list = os.listdir(f'{path}op/')

        self.vid_list.sort()
        self.lab_list.sort()
        
        self.sequence_length = sequence_length
        self.augment = augment
        self.down_factor = down_factor
        self.output_raw = output_raw

        self.mean = np.load('frame_mean.npy')
        self.std = np.load('frame_std.npy')
        
    def __len__(self):
        return len(self.vid_list)

    def random(self):
        return bool(random.getrandbits(1))
    
    def __getitem__(self, idx):
        ip = np.load(f'{self.path}ip/{self.vid_list[idx]}')
        if self.output_raw:
            raw = ip
        ip = ((ip/255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        # ip = (ip - self.mean) / self.std
        op = (np.load(f'{self.path}op/{self.lab_list[idx]}') + 1) / 2 # careful that NSS has a similar conversion (now removed)
        # assumes all videos and heatmaps are normalised
        
        if self.augment:
            # if self.random():
            #     ip = np.flip(ip, axis=1)
            #     op = np.flip(op, axis=1)
            if self.random():
                ip = np.flip(ip, axis=2)
                op = np.flip(op, axis=2)
#        if self.down_factor != 1:
#            # not implemented yet
        if not self.output_raw:
            return {'ip':self.prepare(ip), 'op':self.prepare(op)}
        else:
            return {'ip':self.prepare(ip), 'op':self.prepare(op), 'raw':self.prepare(raw)}


    def prepare(self, raw):
        array = np.zeros([self.sequence_length, raw.shape[1], raw.shape[2], raw.shape[3]])
        # Insert the data into the expected sequence length
        array[:raw.shape[0], :, :, :] = raw
        array = np.transpose(array, (0,3,1,2))
        array = torch.from_numpy(array).float()
        return array

    def check(self):
        for i in range(len(self)):
            print(f'{self.path}ip/{self.vid_list[i]}')
            a = np.load(f'{self.path}ip/{self.vid_list[i]}')
            b = np.load(f'{self.path}op/{self.lab_list[i]}')

class VideoSet(Dataset):
    def __init__(self, path, sequence_length, augment=False, overlap=None, down_factor=1, output_raw=False):
        
        self.path = path
        self.vid_list = os.listdir(f'{path}ip/')
        self.vid_list.sort()

#        self.lab_list = os.listdir(f'{path}op/')
#        self.lab_list.sort()

        self.augment = augment
        self.sequence_length = sequence_length
        self.down_factor = down_factor
        self.output_raw = output_raw

        if overlap is None:
            overlap = int(np.floor(sequence_length/2))
        self.start_frames = np.zeros((0,2)).astype(int)
        for idx, vid in enumerate(self.vid_list):
            frame_count = int(skv.ffprobe(f'{path}ip/{vid}')['video']['@nb_frames'])
            a = np.arange(0, frame_count-sequence_length+1, sequence_length-overlap)
            b = idx * np.ones_like(a)
            self.start_frames = np.append(self.start_frames,(np.vstack((b,a))).T,axis=0)
        
    def __len__(self):
        return len(self.start_frames)

    def random(self):
        return bool(random.getrandbits(1))
    
    def __getitem__(self, idx):
        
        name = self.vid_list[self.start_frames[idx,0]]
        frame = self.start_frames[idx,1]
        
        vid = skv.vread((f'{self.path}ip/{name}'))
        ip = vid[frame:frame + self.sequence_length,:,:,:]
        ip = ((ip/255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        ip = np.transpose(ip,(0,3,1,2))
        
        vid = skv.vread((f'{self.path}op/{name}'))
        op = vid[frame:frame + self.sequence_length,:,:,:]
        op = np.transpose(op,(0,3,1,2))
        
        if self.augment:
            if self.random():
                ip = np.flip(ip, axis=2)
                op = np.flip(op, axis=2)
        
        ip = ip.copy(); ip = torch.from_numpy(ip).float()
        op = op.copy(); op = torch.from_numpy(op).float()
        return {'ip':ip, 'op':op}
        