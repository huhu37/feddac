import copy
import torch

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedCD(w,S,A):
    w_cd = copy.deepcopy(w[0])
    for k in w_cd.keys():
        if 'shortcut.0' in k or 'conv' in k:
            row,col = w_cd[k].shape[0], w_cd[k].shape[1]
            w_cd[k] *= A[0][k].view(row,col,1,1)
            for i in range(1, len(w)):
                w_cd[k] += A[i][k].view(row,col,1,1) * w[i][k]
            w_cd[k] *= S[k].view(row,col,1,1)
        elif 'shortcut.1' in k or 'bn' in k or 'linear' in k:
            w_cd[k] *= A[0][k]
            for i in range(1, len(w)):
                w_cd[k] += A[i][k] * w[i][k]
            w_cd[k] *= S[k]
        else:
            print('w_cd error.')
    return w_cd


