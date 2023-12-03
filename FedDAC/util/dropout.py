import copy
import torch

def gen_left_neuron(net_dict, ratio, device):
    left_neuron = {}
    last_k = 'none'
    last_num = -1
    for idx,(k,v) in enumerate(net_dict.items()):
        if 'shortcut' in k or 'bn' in k:
            # left_neuron[k] = left_neuron[last_k]
            left_neuron[k] = torch.ones(last_num).to(device)
            last_k = k
            continue
        if 'linear' in k:
            num = v.shape[0]
            # left_neuron['linear.weight'] = left_neuron[last_k]
            left_neuron['linear.weight'] = torch.ones(num).to(device)
            left_neuron['linear.bias'] = torch.ones(num).to(device)
            return left_neuron
        num = v.shape[0]
        left_neuron[k] = torch.zeros(num).to(device)
        indices = torch.randperm(num)[:int(ratio*num)]
        left_neuron[k][indices] = 1
        last_k = k
        last_num = num

def gen_dual_left_neuron(left_neuron):
    dual_neuron = copy.deepcopy(left_neuron)

    for k in dual_neuron.keys():
        if 'conv' in k:
            dual_neuron[k] = 1.0 -dual_neuron[k]
        else:
            continue
    return dual_neuron

def gen_mask_neuron(left_neuron, net_dict, device):
    mask = {}
    for idx,(k,v) in enumerate(left_neuron.items()):
        if 'linear.weight' in k:
            row,col = net_dict[k].shape[0], net_dict[k].shape[1]
            m = torch.zeros(row,col).to(device)
            indices = torch.nonzero(v).squeeze()
            m[:,indices] = 1
        elif 'shortcut.0' in k or 'conv' in k:
            row,col = net_dict[k].shape[0], net_dict[k].shape[1]
            m = torch.zeros(row,col).to(device)
            indices = torch.nonzero(v).squeeze()
            m[indices, :] = 1
        elif 'shortcut.1' in k or 'bn' in k or 'linear.bias' in k:
            row = net_dict[k].shape[0]
            m = torch.zeros(row).to(device)
            indices = torch.nonzero(v).squeeze()
            m[indices] = 1
        else:
            print('gen_mask_neuron error.')
        mask[k]=m
    return mask

def get_neuron_net(net_dict, m):
    net = copy.deepcopy(net_dict)
    for idx,(k,v) in enumerate(net.items()):
        if 'shortcut.0' in k or 'conv' in k:
            row,col = v.shape[0], v.shape[1]
            net[k]*=m[k].view(row,col,1,1)
        elif 'shortcut.1' in k or 'bn' in k or 'linear' in k:
            net[k]*=m[k]
        else:
            print('get_neuron_net error.')
    return net