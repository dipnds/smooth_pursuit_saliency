import os, random, torch, cv2
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
        #ip = ((ip/255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] # ResNet
        ip = ((ip/255) - [0.43216, 0.394666, 0.37645]) / [0.22803, 0.22145, 0.216989] # ResNet(2+1)D

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
        self.vid_list = os.listdir(f'{path}ip/')#[:10]
        self.vid_list.sort()

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
        start_frame = self.start_frames[idx,1]

        ip = np.zeros((self.sequence_length, 288, 512, 3))

        capture = cv2.VideoCapture(f'{self.path}ip/{name}')
        capture.set(1, start_frame)
        for i in range(self.sequence_length):
            ret, frame = capture.read()
            ip[i, :, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        capture.release()

        if self.output_raw:
            raw = ip
            pointMap = self.getPointMap(raw, name, start_frame)

        ip = ((ip/255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        ip = np.transpose(ip,(0,3,1,2))
        
        op = np.zeros((self.sequence_length, 288, 512, 3))

        capture = cv2.VideoCapture(f'{self.path}op/{name}')
        capture.set(1, start_frame)
        for i in range(self.sequence_length):
            ret, frame = capture.read()
            op[i, :, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        capture.release()
        op = op / 255
        op = np.transpose(op,(0,3,1,2))
        
        if self.augment:
            if self.random():
                ip = np.flip(ip, axis=3)
                op = np.flip(op, axis=3)
        
        ip = ip.copy(); ip = torch.from_numpy(ip).float()
        op = op.copy(); op = torch.from_numpy(op).float()
        if not self.output_raw:
            return {'ip':ip, 'op':op}
        else:
            return {'ip':ip, 'op':op, 'raw':raw, 'pointMap': pointMap}

    def getPointMap(self, raw, name, start_frame):
        points = np.load(f'{self.path}pts/{name[:-4]}.npy', allow_pickle=True)
        pointMap = np.zeros_like(raw)
        shape = pointMap.shape

        for index, subset in enumerate(points):
            for i in range(shape[0]):
                y = subset[:, 2][subset[:, 0] == (i + start_frame)]
                x = subset[:, 1][subset[:, 0] == (i + start_frame)]
                pointMap[i, :, :, index], _, _ = np.histogram2d(y, x, bins=(shape[1], shape[2]), range=[[-0.5, shape[1] - 0.5], [-0.5, shape[2] - 0.5]])
       
        return pointMap
