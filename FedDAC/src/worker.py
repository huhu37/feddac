import torch
import numpy as np
import math
from model import ResNet34,ResNet18
from util.data import DataSet, DataSet100
import random
import torch.optim as optim
import copy
from torchvision import transforms

def g_mul_m(model, m, layer_name, device):
    for idx,j in enumerate(model.parameters()):
        l_name = layer_name[idx]
        if 'shortcut.0' in l_name or 'conv' in l_name:
            row,col = j.shape[0], j.shape[1]
            j.grad.data*=m[l_name].view(row,col,1,1)
        elif 'shortcut.1' in l_name or 'bn' in l_name or 'linear' in l_name:
            j.grad.data*=m[l_name]
        else:
            print('g_mul_m error.')

class Workers:
    def __init__(self, num_classes, worker_num, batch_size, lr, worker_epochs, mask_grad, load_name, iid_rate, method, device):
        self.model = ResNet18(num_classes).to(device)
        if load_name == 'none':
            pass
        else:
            self.model.load_state_dict(torch.load(f'{load_name}.ckpt'))
            print('load model ',load_name)
        self.method = method
        self.init_lr = lr
        self.lr = lr
        self.num_classes = num_classes
        self.bs = batch_size
        self.worker_epochs = worker_epochs
        self.workers_num = worker_num
        if num_classes==10:
            self.dataset = DataSet(worker_num, batch_size, int(iid_rate))
        else:
            self.dataset = DataSet100(worker_num, batch_size, int(iid_rate))

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.mask_grad = mask_grad
        self.device = device
        self.layer_names = [i for i,_ in self.model.named_parameters()]

        self.trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])


    def run_test(self):
        device = self.device
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataset.test):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

        accuracy = correct / total
        return accuracy, test_loss

    def train_epochs(self, worker_id, mask, epoch):
        device = self.device
       
        self.model.train()
        worker_train_data_loader = self.dataset.train[worker_id]
        worker_label_list = self.dataset.label_set[worker_id]

        label_mask = torch.zeros(self.num_classes).to(device)
        label_mask[worker_label_list] = 1
        train_loss = 0
        correct = 0
        total = 0
        if epoch<400:
            self.lr = self.init_lr
        elif epoch<1000:
            self.lr = self.init_lr*0.1
        else:
            self.lr = self.init_lr*0.01
        optimizer = optim.SGD(self.model.parameters(), self.lr, momentum=0.9, weight_decay=5e-4)

        for epoch in range(self.worker_epochs):
            # print('one worker one epoch')
            for batch_idx, (inputs, targets) in enumerate(worker_train_data_loader):
                inputs = self.trans(inputs)
                inputs, targets = inputs.to(device), targets.to(device)
                # print(self.model.state_dict()['linear.bias'])
                # print(mask['linear.bias'])
                optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.method == 'my':
                    outputs = outputs.masked_fill(label_mask == 0, 0)
                # loss1 = self.criterion(outputs[worker_label_list], targets[worker_label_list])
                # loss2 = self.criterion(torch.tensor([torch.sum(outputs[worker_label_list]),torch.sum(outputs)]), torch.tensor([1.,0.]))
                # loss = loss1+loss2
                loss = self.criterion(outputs,targets)
                loss.backward()

                if self.mask_grad == True:
                    g_mul_m(self.model, mask, self.layer_names,device)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        del optimizer
        accuracy = correct / total
        return train_loss, accuracy, self.model.state_dict()

    def set_cloud_nets(self, cloud_nets):
        self.model.load_state_dict(cloud_nets)


