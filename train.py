import argparse
import logging
from tqdm import tqdm
import os
# os.environ['CUDA_VISIBLE_DEVICES'] == '0,5'
import math
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataset import Dataset

from test import test
from torchsummary import summary
import torchattacks

from resnets import ResNet18, ResNet18_FNet
from wideresnet import WideResNet

from PGD import PGD
# import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def MART_loss(adv_logits, natural_logits, target, beta):
    # Based on the repo MART https://github.com/YisenWang/MART
    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(target)
    adv_probs = F.softmax(adv_logits, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(adv_logits, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(natural_logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss

def perturb_input(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf', epoch = 200):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv, epoch), dim=1),
                                   F.softmax(model(x_natural, epoch), dim=1),
                                   reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def train(model, train_data, test_data, args):

    
    train_loader = train_data.get_dataloader(args.batch_size, shuffle=True)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    t = 5
    T = args.epoch
    if args.scheduler == 'multistep':
        lambda_func = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  1  if epoch < int(0.7 * T) else 0.1 if epoch < int(0.9 * T) else 0.01  
    elif args.scheduler == 'cosine':
        lambda_func = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  (0.0001 + 0.5*(0.01-0.0001)*(1.0+math.cos( (epoch - t)/(args.epoch - t)*math.pi))) / 0.01

    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    device = next(model.parameters()).device
    attack = PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)

    beta = 6.0
    best_acc = 0
    best_robustness = 0
    logger.info('Epoch \t Test acc \t Train Loss \t Train Acc \t Train Clean Acc')
    for epoch in range(1,args.epoch+1):
        train_loss = 0
        loss_natural = 0
        loss_robust = 0
        train_acc = 0
        train_clean_acc = 0
        train_n = 0
        batch_id = 0
        model = model.train()
        with tqdm(train_loader) as loader:
            for data, label in loader:
                batch_id += 1
                loader.set_description(f"Epoch {epoch}")
                
                optimizer.zero_grad()
                data, label= data.to(device), label.to(device)

                clean_outputs = model(data, epoch)
                adv_data = attack(data, label, epoch)
                # adv_data = perturb_input(model, data, epoch = epoch)
                adv_outputs = model(adv_data, epoch)
                # adv_outputs = clean_outputs
                

                if args.loss == 'CE':
                    loss_robust = F.cross_entropy(adv_outputs, label)

                    loss = loss_robust 
                    loss_natural = loss_robust
                elif args.loss == 'trades':
                    loss_natural = criterion(clean_outputs, label)
                   
                    loss_robust = (1.0 / label.shape[0]) * criterion_kl(F.log_softmax(adv_outputs, dim=1),
                                                    F.softmax(clean_outputs, dim=1))

                    loss = loss_natural + beta * loss_robust
                elif args.loss == 'mart':
                    loss = MART_loss(adv_outputs, clean_outputs, label, beta)
                    loss_natural = loss
                    loss_robust = loss

                # Backward and optimize
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * label.size(0)
                clean_acc = 1.0 - torch.count_nonzero(clean_outputs.argmax(dim=-1) - label) / data.shape[0]
                # acc = 1.0 - torch.count_nonzero(adv_outputs.argmax(dim=-1) - label) / data.shape[0]
                acc = 1.0 - torch.count_nonzero(clean_outputs.argmax(dim=-1) - label) / data.shape[0]

                train_acc += acc * label.size(0)
                train_clean_acc += clean_acc * label.size(0)
                train_n += label.shape[0]
                
                loader.set_postfix(loss=round(loss.item(), 4), loss_natural=round(loss_natural.item(),4), \
                                loss_adv=round(loss_robust.item(), 6), accuracy='{:.3f}'.format(acc))

        test_acc = test(model, test_data, args, mode='train')
        # test_acc = 0

        logger.info('%d \t\t  %.4f \t\t%.4f \t\t %.4f \t\t %.4f \t\t %.4f',
            epoch, test_acc, train_loss/train_n, train_acc/train_n, train_clean_acc/train_n, optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
        if epoch%10==0 or epoch > int(0.7 * args.epoch):
            acc_ori, acc_adv = test(model, test_data, args, mode='test')
            if acc_ori + acc_adv > best_acc:
                best_acc = acc_ori + acc_adv
                saved_name = '{0}-{1}-{2}.pt'.format('ResNet18-', args.dataset, args.attack)
                torch.save(model.state_dict(), os.path.join(args.save_path, saved_name))
            if acc_adv > best_robustness:
                best_robustness = acc_adv
                saved_name = '{0}-{1}-{2}.pt'.format('best-ResNet18-', args.dataset, args.attack)
                torch.save(model.state_dict(), os.path.join(args.save_path, saved_name))

            logger.info('Test Acc \t PGD Acc')
            logger.info(' %.4f \t %.4f ', acc_ori, acc_adv)


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='saved weight path')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--data_root', type=str, default='../data/', help='dataset path')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--attack', type=str, default='PGD', help='attack method')
    parser.add_argument('--epoch', type=int, default=120, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--scheduler', type=str, default='multistep', help='learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=3.5e-3, help='weight decay')
    parser.add_argument('--out_dir', type=str, default='./log/', help='log output dir')
    parser.add_argument('--step', type=int, default=10, help='attack method') 
    parser.add_argument('--loss', type=str, default='trades', help='attack method') 
    parser.add_argument('--n_classes', type=int, default=10, help='num classes')
    parser.add_argument('--model', type=str, default='resnet18', help='model arch')
    args = parser.parse_args()


    logfile = os.path.join(args.out_dir, 'ResNet18-{}.log'.format(args.dataset))
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = Dataset(path = args.data_root, dataset = args.dataset, train = True)
    test_data = Dataset(path = args.data_root, dataset = args.dataset, train = False)

    num_classes = 10
    if args.dataset == 'CIFAR100':
        num_classes = 100
    
    assert args.model in ['resnet18', 'wrn-34-10']
    if args.model == 'resnet18':
        model = ResNet18_FNet(num_classes).cuda()
    else:
        model = WideResNet(num_classes=num_classes).cuda()
    if args.pretrained != None:
        model.load_state_dict(torch.load(args.pretrained), strict=False)
    
    # summary(model, (3,32,32))
    train(model, train_data, test_data, args)
