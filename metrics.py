import numpy as np
import torch, tqdm

def NSS(pred, op):
    # op is gt point map
    
    mean = pred - pred.mean(dim=(1, 2, 3),keepdim=True)
    std = pred.std(dim=(1, 2, 3),keepdim=True)
    std[std == 0] = 1
    norm = mean / std

    score = (norm * op).sum(dim=(1, 2, 3)) / op.sum(dim=(1, 2, 3))
    score = score.mean().cpu().detach().numpy()

    return float(score)


def auc_Cross(pred, op, cross_op, splits_count=100):
    # op is gt pointmap, cross_op is other gt point map
    pred = pred.cpu().detach().squeeze().numpy()
    op = np.sign(op.cpu().detach().squeeze().numpy())
    cross_op = np.sign(cross_op.cpu().detach().squeeze().numpy())

    positive_samples = np.argwhere(op == 1)
    cross_samples = np.argwhere(cross_op == 1)
    negative_samples = cross_samples

    replace = False
    if negative_samples.shape[0] < positive_samples.shape[0]:
        replace = True

    np.random.seed(0)
    aucs = []
    for _ in range(splits_count):
        negative_samples = negative_samples[np.random.choice(negative_samples.shape[0], positive_samples.shape[0], replace=replace)]

        all_samples = np.vstack([positive_samples, negative_samples]).astype(int)
        actual_labels = np.hstack([np.ones((positive_samples.shape[0])), np.zeros((negative_samples.shape[0]))])
        predicted = pred[all_samples[:, 0], all_samples[:, 1], all_samples[:, 2]]

        aucs.append(cal_auc(predicted, actual_labels))

    return np.mean(aucs)

def auc_Judd(pred, op):
    # op is gt point map
    
    pred = pred.cpu().detach().numpy()
    op = np.sign(op.cpu().detach().numpy())
    pred = pred.flatten() # default : order='C'
    op = op.flatten()
    
    score = cal_auc(pred,op)
    return score


def auc_Borji(pred, op, splits_count=100):
    # op is gt point map
    
    np.random.seed(0)
    aucs = []
    
    pred = pred[0]
    for _ in tqdm.tqdm(range(splits_count)):
        augmented_gt, actual_labels = augment_neg_samples(op)
        
        predicted = pred[(augmented_gt['data'][:,0],
                        augmented_gt['data'][:,1],
                        augmented_gt['data'][:,2])]
        aucs.append(cal_auc(predicted, actual_labels))
        
    score = np.mean(aucs)
    return score
    
    
def sim(pred, op):
    # op is gt blurred map
    op = op[:, :, 0, :, :]
    
    pred -= pred.min()
    if pred.sum() == 0:
        pred[:, :, :, :] = 1   
    pred /= pred.max() 
    pred /= pred.sum()
    
    op -= op.min()
    if op.sum() == 0:
        op[:, :, :, :] = 1
    op /= op.max()
    op /= op.sum()
    
    pred = pred.cpu().detach().numpy()
    op = op.cpu().detach().numpy()

    # print(op.shape, pred.shape)
    
    score = np.min([op, pred], axis=0).sum()
    return score
    
    
def cc(pred, op):
    # op is gt blurred map
    op = op[:, :, 0, :, :]
    if op.sum() == 0:
        return np.nan

    mean = pred - pred.mean(dim=(1, 2, 3),keepdim=True)
    std = pred.std(dim=(1, 2, 3),keepdim=True)
    std[std == 0] = 1
    pred = mean / std
    
    mean = op - op.mean(dim=(1, 2, 3),keepdim=True)
    std = op.std(dim=(1, 2, 3),keepdim=True)
    std[std == 0] = 1
    op = mean / std
    
    pred = pred.cpu().detach().numpy()
    op = op.cpu().detach().numpy()
    score = np.corrcoef(op.flatten(), pred.flatten())[0,1]
    return score
    

def cal_auc(pred, op):
    
    tp, fp = roc_curve(pred, op)
    h = np.diff(fp)
    auc = np.sum(h * (tp[1:] + tp[:-1])) / 2
    return auc

    
def roc_curve(predicted, actual, cls=1):
    
    prng = np.random.RandomState(1)
    # shuffle the points to get a uniform shuffling (to get ~50% AUCs for chance-models)
    permutation_indices = prng.permutation(len(predicted))
    predicted = predicted[permutation_indices]
    actual = actual[permutation_indices]

    sorted_by_prob = np.argsort(-predicted)
    tp = np.cumsum(np.single(actual[sorted_by_prob] == cls))
    fp = np.cumsum(np.single(actual[sorted_by_prob] != cls))
    tp /= max(np.sum(actual == cls), 1.0)
    fp /= max(np.sum(actual != cls), 1.0)
    tp = np.hstack((0.0, tp, 1.0))
    fp = np.hstack((0.0, fp, 1.0))
    return tp, fp


def augment_neg_samples(gt):
    
    gt = gt.squeeze().cpu().detach().numpy()
    res = {'size': gt.shape[1:],
           'data': np.sign(gt),
           'frame_count': gt.shape[0]}
    
    n_pos_samples = int(np.sum(res['data']))
    labels = np.ones(int(n_pos_samples))
    
    actual = np.array(np.nonzero(gt)).T
    res['data'] = actual
    if n_pos_samples != 0:
        n_samples_to_generate = 2*n_pos_samples
        
        random_negatives = np.hstack([
                np.random.randint(res['frame_count'], size=(n_samples_to_generate,1)), # time
                np.random.randint(res['size'][1], size=(n_samples_to_generate,1)), # x
                np.random.randint(res['size'][0], size=(n_samples_to_generate,1)) # y
                ])
        
        temp = np.zeros_like(gt)
        temp[(random_negatives[:,0],random_negatives[:,2],random_negatives[:,1])] = 1
        
        temp = np.clip(temp-gt,0,1)
        random_negatives = np.array(np.nonzero(temp)).T
        random_negatives = random_negatives[:n_pos_samples,:]
        
        
        labels = np.hstack([labels,np.zeros(random_negatives.shape[0])])
        res['data'] = np.vstack([actual,random_negatives])
    
    return res, labels
