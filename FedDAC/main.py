import torch
import torch.optim as optim
import argparse
import numpy as np
import random
import os
from src.cloud import Cloud
from src.worker import Workers
import time

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_parse():
    # parser for hyperparameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fixed_seed', type=bool, default=True)
    parser.add_argument('--worker_num', type=int, default=100)
    parser.add_argument('--trainer_num', type=int, default=10)
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--bs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--worker_epochs', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=10, help='cifar10')
    parser.add_argument('--gpu', type=str, default='0', help='gpu_id')
    parser.add_argument('--load_name', type=str, default='none', help='load which model')
    parser.add_argument('--method', type=str, default='my', help='use which method')
    parser.add_argument('--agg', type=str, default='fedavg', help='emply which aggregation algrithm')
    parser.add_argument('--low_device', type=float, default=0.4)
    parser.add_argument('--iid_rate', type=float, default=0.1)
    parser.add_argument('--mask_grad', action='store_true', help='Set this flag to True')
    args = parser.parse_args()
    return args

def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr)
    else:
        raise ValueError('unknown optimizer')

if __name__ == '__main__':

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = get_parse()
    if args.fixed_seed:
        setup_seed(args.seed)
    device = 'cuda:' + args.gpu
    # device = 'cpu'
    name = 'fed-worker{}_trainer{}_global_epochs{}_worker_epochs{}_low_device{}_high_device{}'.format(args.worker_num, args.trainer_num, args.epochs, args.worker_epochs,args.low_device, 1 - args.low_device)
    print(name)

    worker = Workers(args.num_classes, args.worker_num, args.bs, args.lr, args.worker_epochs, args.mask_grad, args.load_name, args.iid_rate,args.method, device)

    cloud = Cloud(worker, args.worker_num, args.trainer_num, args.agg, args.method, device)

    # optimizer = create_optimizer(args, cloud.model.parameters())
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 300], gamma=0.1)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    low_device, high_device= args.low_device, 1 - args.low_device
    l = torch.linspace(low_device, 0.5, steps=int(args.worker_num/2))
    r = (1 - l).flip(dims=(0,))
    ratio_list = torch.cat([l,r])

    print('Len of winning worker list is ',args.trainer_num)

    for epoch in range(0, args.epochs):
        if epoch%10==0:
            print('Epoch: %d' % epoch)

        t = -time.time()
        train_loss, train_acc = cloud.train_epoch(ratio_list, int(args.trainer_num/2), epoch)
        # print("Training Acc: {:.4f}, Loss: {:.4f}".format(train_acc, train_loss/8), end=' ')
        t+= time.time()
        # print('train time: ', t)

        if epoch%10==0 or epoch == args.epochs-1:
            test_acc, test_loss = cloud.run_test()
            print("Testing  Acc: {:.4f}, Loss: {:.4f}".format(test_acc, test_loss/79),end=' ')
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
        
        # scheduler.step()
        # save loss and acc
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        
        # if (epoch+1)%100==0:
        #     torch.save(cloud.model.state_dict(),f'r18c10e{epoch}.ckpt')
        #     print('mode save at epoch ',epoch)

    torch.save(cloud.model.state_dict(),f'r18c10{args.num_classes}.ckpt')
    print('mode save at end ',epoch)
    
