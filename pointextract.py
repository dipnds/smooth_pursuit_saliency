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

class PointExtractor:
    def __init__(self, name, files):
        self.name = name
        pathlib.Path(f'{name}/pts/').mkdir(parents=True, exist_ok=True)
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
        pool = multiprocessing.Pool(16)
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        file_count = manager.Value('i', 0)
        for _ in tqdm.tqdm(pool.imap_unordered(partial(self.process, file_count, lock), self.files), total=len(self.files)):
            pass

        pool.close()
        pool.join()

    def process(self, file_count, lock, f):
        raw_video = skvideo.io.vread(f'{self.name}/{f}')
        raw_shape = raw_video.shape
        raw_video = None

        self.makeSaliency(f, raw_shape)

    def makeSaliency(self, f, shape):
        yratio = self.frame_height / shape[1]
        xratio = self.frame_width / shape[2]
        data_files = os.listdir(f'{self.name}/annotation/{f[:-4]}/')
        for data_file in data_files:
            labels = arff.loadarff(f'{self.name}/annotation/{f[:-4]}/{data_file}')
            movementLabels = np.ones(labels[0]['EYE_MOVEMENT_TYPE'].shape, dtype=np.float32) * Labels.OTHER.value
            movementLabels[labels[0]['EYE_MOVEMENT_TYPE'] == bytes(Labels.FIX.name, 'utf-8')] = Labels.FIX.value
            movementLabels[labels[0]['EYE_MOVEMENT_TYPE'] == bytes(Labels.SP.name, 'utf-8')] = Labels.SP.value
            labelArray = np.vstack([labels[0]['time'], labels[0]['x'] * xratio, labels[0]['y'] * yratio, movementLabels]).T.astype(np.float32)
            try:
                frame_period = 1e6/eval(skvideo.io.ffprobe(f'{self.name}/{f}')['video']['@avg_frame_rate'])
                labelArray[:, 0] = (labelArray[:, 0]/frame_period).astype(int)
                labelArray[:, 0][labelArray[:, 0] >= shape[0]] = shape[0] - 1 
           
                samples = [ labelArray[labelArray[:, 3] != Labels.OTHER.value], 
                            labelArray[labelArray[:, 3] == Labels.FIX.value],
                            labelArray[labelArray[:, 3] == Labels.SP.value] ]

                np.save(f'{self.name}/pts/{f}.npy', samples)          
            
            except Exception as e:
                print(f'{e} at {f}')


train_videos = [str(filename.name) for filename in pathlib.Path('train/').glob('*.avi')]
eval_videos = [str(filename.name) for filename in pathlib.Path('eval/').glob('*.avi')]
mikhail_videos = [str(filename.name) for filename in pathlib.Path('mikhail/').glob('*.avi')]
test_videos = [str(filename.name) for filename in pathlib.Path('gazecom/').glob('*.mp4')]

# train_set = PointExtractor('train', train_videos)
# train_set.load()

eval_set = PointExtractor('eval', eval_videos)
eval_set.load()

eval_set = PointExtractor('mikhail', mikhail_videos)
eval_set.load()

test_set = PointExtractor('gazecom', test_videos)
test_set.load()

