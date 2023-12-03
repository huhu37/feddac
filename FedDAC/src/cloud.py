from util.fedavg import FedAvg,FedCD,mFedCD,mFedAvg
from util.dropout import gen_left_neuron, gen_mask_neuron, get_neuron_net, gen_dual_left_neuron
import numpy as np
import torch
import os
import copy
import random
import time
def gen_mask_matrix(net,rate,device='cuda:0'):
    mask = {}
    for i,j in net.items():
        if 'shortcut.0' in i or 'conv' in i or 'linear.weight' in i:
            row,col = j.shape[0], j.shape[1]
            m = torch.zeros(row,col)
            indices = torch.randperm(row*col)[:int(row*col*rate)]
            m.view(-1)[indices] = 1
            mask[i]=m.to(device)
        elif 'shortcut.1' in i or 'bn' in i or 'linear.bias' in i:
            row = j.shape[0]
            v = torch.zeros(row)
            indices = torch.randperm(row)[:int(row*rate)]
            v[indices] = 1
            mask[i]=v.to(device)
        else:
            print('gen_mask_matrix error.')
    return mask

def gen_dual_mask_matrix(net_dict,rate,device='cuda:0'):
    mask1,mask2 = {},{}
    # mask1 = mask1.cuda()
    # mask2 = mask2.cuda()
    for i,j in net_dict.items():
        if 'shortcut.0' in i or 'conv' in i or 'linear.weight' in i:
            row,col = j.shape[0], j.shape[1]
            m1 = torch.zeros(row,col)
            m2 = torch.zeros(row,col)
            perm = torch.randperm(row*col)
            indices1 = perm[:int(row*col*rate)]
            indices2 = perm[int(row*col*rate):]
            m1.view(-1)[indices1] = 1
            m2.view(-1)[indices2] = 1
            mask1[i]=m1.to(device)
            mask2[i]=m2.to(device)
        elif 'shortcut.1' in i or 'bn' in i or 'linear.bias' in i:
            row = j.shape[0]
            v1 = torch.zeros(row)
            v2 = torch.zeros(row)
            perm = torch.randperm(row)
            indices1 = perm[:int(row*rate)]
            indices2 = perm[int(row*rate):]
            v1[indices1] = 1
            v2[indices2] = 1
            mask1[i]=v1.to(device)
            mask2[i]=v2.to(device)
        else:
            print('gen_dual_mask_matrix error.')
    return mask1,mask2

def gen_identical_model_state(net_dict, num, device='cuda:0'):
    sd = {}
    for i,j in net_dict.items():
        if 'shortcut.0' in i or 'conv' in i or 'linear.weight' in i:
            row,col = j.shape[0], j.shape[1]
            m = torch.ones(row,col)*num
            sd[i]=m.to(device)
        elif 'shortcut.1' in i or 'bn' in i or 'linear.bias' in i:
            row = j.shape[0]
            v = torch.ones(row)*num
            sd[i]=v.to(device)
        else:
            print('gen_identical_state_dict error.')
    return sd    

def calculate_scaling_matrix(M_list, device='cuda:0'):
    P = gen_identical_model_state(M_list[0],-1)
    lm = len(M_list)
    for k in P.keys():
        for i in range(lm):
            indices = (M_list[i][k]==1)
            P[k][indices] = torch.where(P[k][indices] == -1, torch.tensor(1.0).to(device), 1 / (1 / P[k][indices] + 1))
    return P

class Cloud:
    def __init__(self, worker, worker_num, trainer_num, agg, method, device):
        self.worker = worker
        self.model = self.worker.model
        self.worker_num = worker_num
        self.trainer_num = trainer_num
        self.device = device
        self.agg = agg
        self.method = method
        self.worker_info = {}

    def train_epoch(self, ratio_list, cover_num, epoch):

        winning_workers = self.choose_workers(cover_num)
        
        worker_loss, worker_acc, worker_nets_list, worker_mask_list = [], [], [], []
        cloud_net = self.get_cloud_net()

        # print('worker num is ', cover_num)
        for worker_id in winning_workers[:cover_num]:

            if self.method == 'my':
                low_mask, high_mask = gen_dual_mask_matrix(cloud_net, ratio_list[worker_id],self.device)
                low_net = cloud_net
                high_net = cloud_net
            elif self.method == 'fedavg':
                low_mask = gen_mask_matrix(cloud_net, ratio_list[worker_id],self.device)
                high_mask = gen_mask_matrix(cloud_net, 1-ratio_list[worker_id],self.device)
                low_net = cloud_net
                high_net = cloud_net
            elif self.method == 'roll':
                low_mask, high_mask = gen_dual_mask_matrix(cloud_net, ratio_list[worker_id])
            else:
                # print('other method')
                n1 = gen_left_neuron(cloud_net,ratio_list[worker_id],self.device)
                # n2 = gen_left_neuron(cloud_net,ratio_list[worker_id],self.device)
                n2 = gen_dual_left_neuron(n1)
                low_mask = gen_mask_neuron(n1,cloud_net,self.device)
                high_mask = gen_mask_neuron(n2,cloud_net,self.device)
                low_net = get_neuron_net(cloud_net,low_mask)
                high_net = get_neuron_net(cloud_net,high_mask)


            self.worker.set_cloud_nets(low_net)
            train_loss , train_acc , current_worker_nets = self.worker.train_epochs(worker_id, low_mask, epoch)
            worker_acc.append(train_acc)
            worker_loss.append(train_loss)
            worker_nets_list.append(current_worker_nets)
            worker_mask_list.append(low_mask)
            
            self.worker.set_cloud_nets(high_net)
            train_loss, train_acc, current_worker_nets = self.worker.train_epochs(self.worker_num - worker_id-1, high_mask, epoch)
            worker_acc.append(train_acc)
            worker_loss.append(train_loss)
            worker_nets_list.append(current_worker_nets)
            worker_mask_list.append(high_mask)

        scaling_matrix = calculate_scaling_matrix(worker_mask_list)
        if self.agg=='fedavg':
            global_nets = FedAvg(worker_nets_list)
        elif self.agg=='fedcd':
            global_nets = FedCD(worker_nets_list, scaling_matrix,worker_mask_list, device=self.device)
        else:  
            print('aggregation algrithm error.')

        self.set_global_nets(global_nets)
        worker_nets_list = None
        worker_mask_list = None

        return np.mean(worker_loss), np.mean(worker_acc)
    
    def choose_workers(self, cover_num):
        l = [random.randint(0, int(self.worker_num/2)) for _ in range(cover_num)] 
        for i in range(cover_num):
            l.append(self.worker_num - 1 - l[cover_num-i-1])
        # print('selected worker is ',l)
        return l
    
    def get_cloud_net(self):
        # current_net = copy.deepcopy(self.model.state_dict())
        return self.model.state_dict()
    
    def set_global_nets(self, global_nets):
        return self.model.load_state_dict(global_nets)

    def run_test(self):
        accuracy, test_loss = self.worker.run_test()
        return accuracy, test_loss

    

