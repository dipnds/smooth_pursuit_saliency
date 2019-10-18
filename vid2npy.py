from scipy.io import arff
from scipy import ndimage
from sklearn.model_selection import train_test_split

import os, cv2, tqdm, skvideo.io, pathlib
import numpy as np

class DataLoader:
    def __init__(self, name, files):
        self.name = name
        pathlib.Path(f'{name}/ip/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'{name}/op/').mkdir(parents=True, exist_ok=True)
        self.files = files
        self.frame_count = 0 
        self.frame_std = 0
        self.frame_height = 72
        self.frame_width = 128

        self.frame_mean = np.zeros([1, self.frame_height, self.frame_width, 3])
        self.frame_std = np.zeros_like(self.frame_mean)

        self.seq_len = 100
        self.shift = 50
        self.file_count = 0

    def load(self):
        print(f'Loading {self.name}')
        self.file_progress = tqdm.tqdm(self.files)
        for f in self.file_progress:
            raw_video = skvideo.io.vread(f'all_videos/{f}')
            self.file_progress.set_description(f'Resizing: {f}')
            video = self.resize(raw_video, [raw_video.shape[0], self.frame_height, self.frame_width, 3], np.int32)

            self.frame_count += raw_video.shape[0]
            self.frame_mean += np.sum(video, axis=0)
            self.frame_std += np.sum(video ** 2, axis=0)

            self.file_progress.set_description(f'Calculating Saliency: {f}')
            saliencyMap = self.makeSaliency(f, raw_video.shape)
            self.file_progress.set_description(f'Writing: {f}')
            for i in range(int(np.ceil((video.shape[0] - self.seq_len)/self.shift)) + 1):
                self.file_count += 1
                np.save(f'{self.name}/ip/{self.file_count:03d}_vid{f[0:-4]}.npy', video[i*self.shift:np.min((i*self.shift+self.seq_len,video.shape[0])),:,:,:])
                np.save(f'{self.name}/op/{self.file_count:03d}_lab{f[0:-4]}.npy', saliencyMap[i*self.shift:np.min((i*self.shift+self.seq_len,video.shape[0])),:,:])


        self.frame_mean /= self.frame_count
        self.frame_std = np.sqrt((self.frame_std - self.frame_mean**2)/self.frame_count)
        self.frame_std[self.frame_std==0] = 1

    def resize(self, video, shape, dtype):
        resized = np.zeros(shape, dtype=dtype)
        if len(shape) == 4:
            for num, frame in enumerate(video):
                resized[num, :, :, :] = cv2.resize(frame, (shape[2], shape[1]))
        else:
            for num, frame in enumerate(video):
                resized[num, :, :] = cv2.resize(frame, (shape[2], shape[1]))
        return resized

    def makeSaliency(self, f, shape):
        saliencyMap = np.zeros(shape[0:-1])
        data_files = os.listdir(f'output_sp_tool_50_files/test/{f[0:-4]}/')
        for data_file in data_files:
            labels = arff.loadarff(f'output_sp_tool_50_files/test/{f[0:-4]}/{data_file}')
            interval = labels[0].size / shape[0]
            for i in range(shape[0]): # frame loop
                y = labels[0]['y'][(int(np.round(i*interval))):int(np.round((i+1)*interval))]
                x = labels[0]['x'][(int(np.round(i*interval))):int(np.round((i+1)*interval))]
                saliencyMap[i, :, :], _, _ = np.histogram2d(y, x, bins=(shape[1], shape[2]), range=[[-0.5, shape[1] - 0.5], [-0.5, shape[2] - 0.5]])

        sigma_t = 24.75/3; sigma_s = 26.178 # see Mikhail for presets
        ndimage.gaussian_filter(saliencyMap, sigma=[sigma_t, sigma_s, sigma_s], output=saliencyMap) # spatio-temporal smoothing
        resized = self.resize(saliencyMap, [saliencyMap.shape[0], self.frame_height, self.frame_width], np.float64)
        skvideo.io.vwrite('check.mp4',(255*resized/np.max(resized)).astype(np.uint8))
        resized = 2*resized/np.max(resized) - 1 # normalised per video
        return resized

    def normaliseFrames(self, loader):
        for vid_file in os.listdir(f'{self.name}/ip'):
            ip = np.load(f'{self.name}/ip/{vid_file}')
            ip = (ip - loader.frame_mean) / loader.frame_std
            np.save(f'{self.name}/ip/{vid_file}', ip)

    def writeDistribution(self):
        np.save('frame_mean.npy', self.frame_mean)
        np.save('frame_std.npy', self.frame_std)


train_videos, eval_videos = train_test_split(os.listdir('all_videos'), test_size=0.1, shuffle=False) # eval has 10% of train data

train_set = DataLoader('train', train_videos)
train_set.load()
train_set.normaliseFrames(train_set)
train_set.writeDistribution()

eval_set = DataLoader('eval', eval_videos)
eval_set.load()
eval_set.normaliseFrames(train_set)
