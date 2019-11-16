import numpy as np

def NSS(op, pred):
    ma = pred - pred.mean(dim=(1, 2, 3, 4),keepdim=True)
    sd = pred.std(dim=(1, 2, 3, 4),keepdim=True); sd[sd==0] = 1
    ma = ma/sd
    # op = (op+1)/2
    score = (ma * (op.sign())).mean().cpu().detach().numpy()
    return score