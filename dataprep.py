import os
from torch.utils.data import Dataset
from torch import from_numpy
import numpy as np

class prep(Dataset):
    
    def __init__(self, path, augment=False, down_factor=1):
        
        self.path = path
        self.vid_list = os.listdir(path+'ip/')
        self.lab_list = os.listdir(path+'op/')
        self.augment = augment
        self.down_factor = down_factor
        
    def __len__(self):
        
        return len(self.vid_list)
    
    def __getitem__(self, idx):
        
        name = self.vid_list[idx]
        ip = np.load(self.path+'ip/'+name)
        name = self.lab_list[idx]
        op = np.load(self.path+'op/'+name)
        # assumes all videos and heatmaps are normalised
        
#        if self.augment:
#            # not implemented yet
#        if self.down_factor != 1:
#            # not implemented yet
        
        ip = np.transpose(ip, (0,3,1,2))
        ip = from_numpy(ip).float()
        sample = {'ip':ip,'op':op}

        return sample