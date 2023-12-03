This is the source code for FedDAC.

## Run

Federated learning for ResNet18 on the CIFAR-10 is run with the following command:

> python main.py --gpu=0 --epochs=800   --lr=0.0002 --trainer_num=10 --iid_rate=5 --agg=fedcd --low_device=0.1 --method=my --bs=10 --mask_grad

For the CIFAR-100 dataset:

> python main.py --gpu=0 --epochs=1000   --lr=0.0001 --trainer_num=10 --iid_rate=50 --agg=fedcd --low_device=0.1 --method=my --bs=24 --mask_grad --num_classes=100
