from scipy.io import arff
from functools import partial
import scipy.ndimage
from sklearn.model_selection import train_test_split

import os, cv2, tqdm, skvideo.io, pathlib, multiprocessing, enum, pickle
import numpy as np

class Labels(enum.Enum):
    FIX = 0
    SP = 1
    OTHER = 2

class DataLoader:
    def __init__(self, name, files):
        self.name = name
        pathlib.Path(f'{name}/ip/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'{name}/op/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'{name}/pts/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'salient_videos/').mkdir(parents=True, exist_ok=True)
        self.files = files

        self.frame_height = 288
        self.frame_width = 512

        self.frame_mean = np.zeros([1, self.frame_height, self.frame_width, 3])
        self.frame_std = np.zeros_like(self.frame_mean)
        self.frame_count = 0 

        self.seq_len = 100
        self.shift = 80
        self.framerate_scale = 5

    def load(self):
        print(f'Loading {self.name}')
        pool = multiprocessing.Pool(4)
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        file_count = manager.Value('i', 0)
        for _ in tqdm.tqdm(pool.imap_unordered(partial(self.process, file_count, lock), self.files), total=len(self.files)):
            pass

        pool.close()
        pool.join()

        # try:
        #     self.frame_mean = (self.frame_mean / self.frame_count).astype(np.float32)
        #     self.frame_std = np.sqrt(((self.frame_std/self.frame_count) - self.frame_mean**2).astype(np.float32))
        #     self.frame_std[self.frame_std==0] = 1
        # except TypeError as e:
        #     print(e, 'Type Error')
        #     pickle.dump(self.frame_count, open('frame_count_error.p', 'wb'))
        #     pickle.dump(self.frame_mean, open('frame_mean_error.p', 'wb'))
        #     pickle.dump(self.frame_std, open('frame_std_error.p', 'wb'))
        #     exit()

    def process(self, file_count, lock, f):
        raw_video = skvideo.io.vread(f'{self.name}/{f}')
        raw_shape = raw_video.shape
        video = self.resize(raw_video, [raw_video.shape[0], self.frame_height, self.frame_width, 3], np.uint8)
        skvideo.io.vwrite(f'{self.name}/ip/{f}.mp4', video, outputdict={'-r': '25', '-b': '300000000'})
        video = None
        raw_video = None

        saliencyMap = self.makeSaliency(f, raw_shape)
        skvideo.io.vwrite(f'{self.name}/op/{f}.mp4', saliencyMap, outputdict={'-r': '25', '-b': '300000000'})


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
        # saliencyMap = np.zeros(shape, dtype=np.float32)
        saliencyMap = np.memmap(f'/tmp/{f}', dtype=np.float32, mode='w+', shape=shape)
        data_files = os.listdir(f'annotation/{f[:-4]}/')
        for data_file in data_files:
            labels = arff.loadarff(f'annotation/{f[:-4]}/{data_file}')
            movementLabels = np.ones(labels[0]['EYE_MOVEMENT_TYPE'].shape, dtype=np.float32) * Labels.OTHER.value
            movementLabels[labels[0]['EYE_MOVEMENT_TYPE'] == bytes(Labels.FIX.name, 'utf-8')] = Labels.FIX.value
            movementLabels[labels[0]['EYE_MOVEMENT_TYPE'] == bytes(Labels.SP.name, 'utf-8')] = Labels.SP.value
            labelArray = np.vstack([labels[0]['time'], labels[0]['x'], labels[0]['y'], movementLabels]).T.astype(np.float32)
            try:
                frame_period = 1e6/eval(skvideo.io.ffprobe(f'{self.name}/{f}')['video']['@avg_frame_rate'])
                labelArray[:, 0] = (labelArray[:, 0]/frame_period).astype(int)
                labelArray[:, 0][labelArray[:, 0] >= shape[0]] = shape[0] - 1 
           
                samples = [ labelArray[labelArray[:, 3] != Labels.OTHER.value], 
                            labelArray[labelArray[:, 3] == Labels.FIX.value],
                            labelArray[labelArray[:, 3] == Labels.SP.value] ]          
            
                for index, subset in enumerate(samples):
                    for i in range(shape[0]): # frame loop
                        y = subset[:, 2][subset[:, 0] == i]
                        x = subset[:, 1][subset[:, 0] == i]
                        saliencyMap[i, :, :, index], _, _ = np.histogram2d(y, x, bins=(shape[1], shape[2]), range=[[-0.5, shape[1] - 0.5], [-0.5, shape[2] - 0.5]])
            except Exception as e:
                print(f'{e} at {f}')

        sigma_t = 24.75/3; sigma_s = 26.178 # see Mikhail for presets
        resizedSaliency = np.zeros([saliencyMap.shape[0], self.frame_height, self.frame_width, 3], dtype=np.uint8)
        # pointsMap = saliencyMap.astype(np.uint8)
        for index, _ in enumerate(samples):
            saliencyMap[:, :, :, index] = scipy.ndimage.gaussian_filter(saliencyMap[:, :, :, index], sigma=[sigma_t, sigma_s, sigma_s]) # spatio-temporal smoothing
            resized = self.resize(saliencyMap[:, :, :, index], [saliencyMap.shape[0], self.frame_height, self.frame_width], np.float32)
            maximum, minimum = np.max(resized), np.min(resized)
            resizedSaliency[:, :, :, index] = ( (resized - minimum) / ((maximum if maximum != 0 else 1) - minimum) ) * 255

        # self.renderVideo(f, saliencyMap)
        pathlib.Path(f'/tmp/{f}').unlink()
        

        return resizedSaliency

    def normaliseFrames(self, loader):
        self.frame_mean = loader.frame_mean
        self.frame_std = loader.frame_std
        pool = multiprocessing.Pool(8)
        files = os.listdir(f'{self.name}/ip')
        print("Normalising frames")
        for _ in tqdm.tqdm(pool.imap_unordered(self.normaliseFile, files), total=len(files)):
            pass

        pool.close()
        pool.join()          

    def normaliseFile(self, vid_file):
        ip = np.load(f'{self.name}/ip/{vid_file}')
        ip = (ip - self.frame_mean) / self.frame_std
        np.save(f'{self.name}/ip/{vid_file}', ip.astype(np.float16))

    def calcDistribution(self):
        files = os.listdir(f'{self.name}/ip')
        for f in tqdm.tqdm(files):
            ip = np.load(f'{self.name}/ip/{f}').astype(np.float64)
            self.frame_count += ip.shape[0]
            self.frame_mean += np.sum(ip, axis=0)
            self.frame_std += np.sum(ip ** 2, axis=0)

        self.frame_mean = (self.frame_mean / self.frame_count).astype(np.float32)
        self.frame_std = np.sqrt(((self.frame_std / self.frame_count) - self.frame_mean**2).astype(np.float32))
        self.frame_std[self.frame_std==0] = 1
  
    def writeDistribution(self):
        np.save('frame_mean.npy', self.frame_mean)
        np.save('frame_std.npy', self.frame_std)

    def renderVideo(self, f, saliencyMap):
        raw_video = skvideo.io.vread(f'{self.name}/{f}')
        raw_video[:, :, :, 0] = raw_video[:, :, :, 0].astype(np.float32) / 2.0
        for i in range(1, 3):
            maximum = np.max(saliencyMap[:, :, :, i])
            saliencyMap[:, :, :, i] /= maximum if maximum != 0 else 1
            saliencyMap[:, :, :, i] *= 255
            raw_video[:, :, :, i] = np.mean([raw_video[:, :, :, i], saliencyMap[:, :, :, i]], axis=0)
        skvideo.io.vwrite(f'salient_videos/{f}.mp4', raw_video.astype(np.uint8))

train_videos = [str(filename.name) for filename in pathlib.Path('train/').glob('*.avi')]
eval_videos = [str(filename.name) for filename in pathlib.Path('eval/').glob('*.avi')]
test_videos = [str(filename.name) for filename in pathlib.Path('test/').glob('*.avi')]

train_set = DataLoader('train', train_videos)
# train_set.calcDistribution()
train_set.load()
# train_set.writeDistribution()
# train_set.normaliseFrames(train_set)

eval_set = DataLoader('eval', eval_videos)
eval_set.load()
#eval_set.normaliseFrames(train_set)

# test_set = DataLoader('test', test_videos)
# test_set.load()
# test_set.normaliseFrames(train_set)

