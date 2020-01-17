def NSS(pred, op):
    mean = pred - pred.mean(dim=(1, 2, 3),keepdim=True)
    std = pred.std(dim=(1, 2, 3),keepdim=True)
    std[std == 0] = 1
    norm = mean / std

    score = (norm * op).sum(dim=(1, 2, 3)) / op.sum(dim=(1, 2, 3))  
    score = score.mean().cpu().detach().numpy()

    return float(score)