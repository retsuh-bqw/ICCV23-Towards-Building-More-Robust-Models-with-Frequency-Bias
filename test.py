import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as trans
import argparse
import torchattacks
import torch.distributed as dist
from dataset import Dataset
import torchattacks
import random
from wideresnet import WideResNet
from resnets import ResNet18, ResNet18_FNet
from PGD import PGD
from CW_inf_attack import CW_linf


def test(model, test_data, args, mode='test'):
    if args.attack != None and mode == 'test':
        if args.attack == 'AutoAttack':
            attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        elif args.attack == 'APGD':
            attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        elif args.attack == 'PGD':
            attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=args.step, random_start=True)
        elif args.attack == 'FGSM':
            attack = torchattacks.FGSM(model, eps=8/255)
        elif args.attack == 'CW':
            attack = CW_linf(model)
        elif args.attack == 'Square':
            attack = torchattacks.Square(model, norm='Linf', eps=8/255)
            
    test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=False)
    correct = 0
    correct_adv = 0
    all_samples = 0
    device = next(model.parameters()).device
    model = model.eval()
    batch_id = 0
    cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465])
    cifar10_std = torch.tensor([0.2471, 0.2435, 0.2616])
    # epoch = args.epoch
    # data_norm = trans.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    with tqdm(test_loader) as test_loader:
        for data, label in test_loader:
            batch_id+=1
            data, label = data.to(device), label.to(device)

            outputs = model(data)
            correct += data.shape[0] - torch.count_nonzero(outputs.argmax(dim=-1) - label)
            if args.attack != 'None' and mode == 'test':
                data = attack(data, label)
                outputs = model(data)
                correct_adv += data.shape[0] - torch.count_nonzero(outputs.argmax(dim=-1) - label)
            all_samples += data.shape[0]
            if args.attack == 'AutoAttack':
                print('batch_id: {} \t Attack: {} \t Accuracy = {}/{} = {:.3f}'.format(batch_id,args.attack, correct_adv, all_samples, correct_adv*1.0 / all_samples))
            

    acc_ori = correct*1.0 / all_samples
    print('Attack: None \t Accuracy = {}/{} = {:.3f}'.format(correct, all_samples, acc_ori))
    acc_adv = 0
    if args.attack != 'None' and mode == 'test':
        acc_adv = correct_adv*1.0 / all_samples
        print('Attack: {} \t Accuracy = {}/{} = {:.3f}'.format(args.attack, correct_adv, all_samples, acc_adv))
        print(acc_ori - acc_adv)
    if mode =='train':
        return acc_ori  
    else:
        return acc_ori, acc_adv


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('--weights', type=str, default='./checkpoint/BORT-WRN28-10-CIFAR10.pt', help='saved model path')
    parser.add_argument('--data_root', type=str, default='../data/', help='dataset path')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size')
    parser.add_argument('--attack', type=str, default='None', help='attack method')
    parser.add_argument('--step', type=int, default=10, help='attack method')
    parser.add_argument('--model', type=str, default='resnet18', help='tested model')
    args = parser.parse_args()
    device = 'cuda'

    num_classes = 10
    if args.dataset == 'CIFAR100':
        num_classes = 100
    
    assert args.model in ['resnet18', 'wrn-34-10']
    if args.model == 'resnet18':
        model = ResNet18_FNet(num_classes)
    else:
        model = WideResNet(num_classes=num_classes)

    weights = torch.load(args.weights)
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    model.load_state_dict(weights_dict, strict=False)

    model = model.to(device)
    test_data = Dataset(path = args.data_root, dataset = args.dataset, train = False)


    acc = test(model, test_data, args)
