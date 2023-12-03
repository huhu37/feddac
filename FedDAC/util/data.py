import torch
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset



def build_dataset():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform_test)
    
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data, label = self.samples[index]
        if self.transform:
            data = self.transform(data)
        return data, label

class DataSet:
    def __init__(self, worker_num, batch_size, label_num):
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform_test)
        
        if label_num == 0:
            print('IID data load.')
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform_train)
            worker_data_list = torch.utils.data.random_split(trainset, [len(trainset) // worker_num for _ in range(worker_num)])
            print(worker_data_list[0][0])
        else:
            transform_my = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_my)
            all_samples = [trainset[i] for i in range(len(trainset))]    
            sorted_samples = sorted(all_samples, key=lambda x: x[1])
            piece_num = 50000//100//label_num
            p = torch.arange(10)
            l_idx = [0]*10
            temp_list = []
            label_set_list = []
            for i in range(worker_num):
                a = []
                take_p = [x.item() for x in p[:label_num]]
                label_set_list.append(take_p)
                for r in take_p:
                    start = l_idx[r] + r*50000//10
                    end = l_idx[r]+piece_num + r*50000//10
                    a.extend(sorted_samples[start:end])
                    l_idx[r]+=piece_num
                temp_list.append(a)
                p = torch.roll(p, shifts=1, dims=0)
            worker_data_list = temp_list
        
        self.train = [torch.utils.data.DataLoader(worker_data_list[i], batch_size=batch_size, shuffle=True) for i in range(worker_num)] 
        self.label_set = label_set_list
        self.test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        del worker_data_list
        print(f'there are {len(self.train)} workers, one worker hold {len(self.train[2].dataset)} data')

class DataSet100:
    def __init__(self, worker_num, batch_size, label_num):
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,transform=transform_test)
        
        if label_num == 0:
            print('IID data load.')
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform_train)
            worker_data_list = torch.utils.data.random_split(trainset, [len(trainset) // worker_num for _ in range(worker_num)])
            print(worker_data_list[0][0])
        else:
            transform_my = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_my)
            all_samples = [trainset[i] for i in range(len(trainset))]    
            sorted_samples = sorted(all_samples, key=lambda x: x[1])
            piece_num = 50000//worker_num//label_num
            p = torch.arange(100)
            l_idx = [0]*100
            temp_list = []
            label_set_list = []
            for i in range(worker_num):
                a = []
                take_p = [x.item() for x in p[:label_num]]
                label_set_list.append(take_p)
                for r in take_p:
                    start = l_idx[r] + r*50000//100
                    end = l_idx[r]+piece_num + r*50000//100
                    a.extend(sorted_samples[start:end])
                    l_idx[r]+=piece_num
                temp_list.append(a)
                p = torch.roll(p, shifts=10, dims=0)
            worker_data_list = temp_list
        
        self.train = [torch.utils.data.DataLoader(worker_data_list[i], batch_size=batch_size, shuffle=True) for i in range(worker_num)] 
        self.label_set = label_set_list
        self.test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        del worker_data_list
        print(f'there are {len(self.train)} workers, one worker hold {len(self.train[2].dataset)} data')