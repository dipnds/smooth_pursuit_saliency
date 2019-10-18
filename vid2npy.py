import skvideo.io
from scipy.io import arff
from scipy import ndimage
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
# import matplotlib.pyplot as plt

# H = 528; L = 720 # frame dim
H = 72; L = 128 # reduced frame dimension
frame_mean = np.zeros([1,H,L,3]); frame_std = np.zeros([1,H,L,3])
frame_count = 0;
seq = 100; shift = 50
file_count = 0

for vid_list in os.listdir('all_videos'):
    vidnpy = skvideo.io.vread('all_videos/'+vid_list)
    
##    resize, see Mikhail
    vid = np.zeros([vidnpy.shape[0],H,L,3],dtype=np.int32)
    for num,frame in enumerate(vidnpy):
        vid[num,:,:,:] = cv2.resize(frame,(L,H))
    
##    pad, see Mikhail not doing this
#    vid = np.zeros([vidnpy.shape[0],H,L,3],dtype=np.int32)
#    err = [int((H-vidnpy.shape[1])/2),int((L-vidnpy.shape[2])/2)]
#    vid[:,err[0]:err[0]+vidnpy.shape[1],err[1]:err[1]+vidnpy.shape[2],:] = vidnpy
    
    frame_count += vid.shape[0]
    frame_mean += np.sum(vid, axis=0)
    frame_std += np.sum(vid**2, axis=0)
    
#    plt.imshow(vid[0,:,:,:]); plt.show()
    
    op = np.zeros(vid.shape[0:-1])
    gt = np.zeros(vidnpy.shape[0:-1])
    for anno in os.listdir('output_sp_tool_50_files/test/'+vid_list[0:-4]+'/'): # annotator loop
        lab = arff.loadarff('output_sp_tool_50_files/test/'+vid_list[0:-4]+'/'+anno)
        interv = lab[0].size/frame_count
        for i in range(vid.shape[0]): # frame loop
            
            ## correcting gaze for resizing
            y = lab[0]['y'][(int(np.round(i*interv))):int(np.round((i+1)*interv))]
            x = lab[0]['x'][(int(np.round(i*interv))):int(np.round((i+1)*interv))]
            for j in range(x.shape[0]):
                if 0<=y[j]<vidnpy.shape[1]-1 and 0<=x[j]<vidnpy.shape[2]-1:
                    gt[i,int(np.round(y[j])),int(np.round(x[j]))] += 1
                
    sigma_t = 24.75/3; sigma_s = 26.178 # see Mikhail for presets
    temp = gt.copy()
    ndimage.gaussian_filter(gt,sigma=[sigma_t, sigma_s, sigma_s], output=gt) # spatio-temporal smoothing
    for num,frame in enumerate(gt):
        op[num,:,:] = cv2.resize(gt[num,:,:],(L,H))
#    skvideo.io.vwrite('check.mp4',(255*op/np.max(op)).astype(np.uint8))
    op = 2*op/np.max(op) - 1 # normalised per video

##            per frame Gaussian blur, dropped            
##            correcting gaze for padding
#            y = np.minimum((lab[0]['y'][(int(np.round(i*interv))):int(np.round((i+1)*interv))]) + err[0], H-1)
#            x = np.minimum((lab[0]['x'][(int(np.round(i*interv))):int(np.round((i+1)*interv))]) + err[1], L-1)
#            for j in range(x.shape[0]):
#                op[i,int(np.round(y[j])),int(np.round(x[j]))] += 1           
#            op[i,:,:] = cv2.GaussianBlur(op[i,:,:],(13,13),9) # change third 3 (sigma) as per exisitng code
                
    print(op.shape)
    
    count = int(np.ceil((vid.shape[0] - seq)/shift)) + 1
    for i in range(count):
        file_count += 1
        np.save('ip/vid'+(str(100000+file_count))[1:]+'_'+vid_list[0:-4]+'.npy',
                vid[i*shift:np.min((i*shift+seq,vid.shape[0])),:,:,:])
        np.save('op/lab'+(str(100000+file_count))[1:]+'_'+vid_list[0:-4]+'.npy',
                op[i*shift:np.min((i*shift+seq,vid.shape[0])),:,:])

frame_mean /= frame_count
frame_std = np.sqrt((frame_std - frame_mean**2)/frame_count)
np.save('frame_mean.npy',frame_mean)
np.save('frame_std.npy',frame_std)

for vid_list in os.listdir('ip'):
    ip = np.load('ip/'+vid_list)
    ip = (ip - frame_mean) / frame_std
    np.save('ip/'+vid_list,ip)

temp = os.listdir('ip'); temp.sort()
split = train_test_split(temp,test_size=0.1,shuffle=False) # eval has 10% of train data
list_tr = split[0]; list_ev = split[1]
with open('list_tr.pkl', 'wb') as f:
    pickle.dump(list_tr, f)
with open('list_ev.pkl', 'wb') as f:
    pickle.dump(list_ev, f)
