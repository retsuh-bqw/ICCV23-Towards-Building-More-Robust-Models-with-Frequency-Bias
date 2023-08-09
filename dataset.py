import os
import torch
from torchvision import datasets, transforms
import torchvision

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

svhn_mean= (0.4376821, 0.4437697, 0.47280442) 
svhn_std= (0.19803012, 0.20101562, 0.19703614) 

# mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
# std = torch.tensor(cifar10_std).view(3,1,1).cuda()

# upper_limit = ((1 - mu)/ std)
# lower_limit = ((0 - mu)/ std)


class Dataset():
    def __init__(self, path:str, dataset:str, train:bool):
        
        assert dataset in ['CIFAR10', 'CIFAR100', 'SVHN', 'imagenette']

        if train:
            transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),  
                            transforms.ToTensor(),])

        else:    
            transform = transforms.Compose([     
                            transforms.ToTensor(),])

        if dataset=='CIFAR10':
            dataset = datasets.CIFAR10(root=path, train=train, transform=transform , download=True)
        elif dataset=='CIFAR100':
            dataset = datasets.CIFAR100(root=path, train=train, transform=transform, download=True)
        elif dataset=='SVHN':
            split = 'train' if train else 'test'
            dataset = datasets.SVHN(root=path, split=split, transform=transforms, download=True)
        elif dataset=='imagenette':
            if train:
                train_transform = transforms.Compose([
                    transforms.RandomCrop(160),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ])
                dataset =torchvision.datasets.ImageFolder(root='../imagenette2-160/train',transform=train_transform)
            else:
                valid_transform=transforms.Compose([
                    transforms.CenterCrop(160),
                    transforms.ToTensor(),
                    ])
                dataset =torchvision.datasets.ImageFolder(root='../imagenette2-160/val',transform=valid_transform)


        
        self.dataset = dataset


    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=8):

        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    