'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class LearnableFNetBlock(nn.Module):
    def __init__(self, dim, patches):
        super().__init__()
        self.projection = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)     
        self.learnable_freq_control = nn.Parameter(torch.ones(dim)) # Deprecated                             

    def forward(self, x, epoch):
        
        Freq = torch.fft.fft(torch.fft.fft(x.permute(0,2,1), dim=-1), dim=-2)

        b, patches, c = Freq.shape
        D_0 = patches // 2 + (patches // 8 - patches // 2) * (epoch / 120)
        lowpass_filter_l = torch.exp(-0.5 * torch.square(torch.linspace(0, patches // 2 - 1, patches // 2).unsqueeze(1).repeat(1,c).cuda() / (D_0))).view(1, patches // 2, c).cuda()
        lowpass_filter_r = torch.flip(torch.exp(-0.5 * torch.square(torch.linspace(1, patches // 2 , patches // 2).unsqueeze(1).repeat(1,c).cuda() / (D_0))).view(1, patches // 2, c).cuda(), [1])
        lowpass_filter = torch.concat((lowpass_filter_l, lowpass_filter_r), dim=1)
        
        low_Freq = Freq * lowpass_filter
        lowFreq_feature = torch.fft.ifft(torch.fft.ifft(low_Freq, dim=-2), dim=-1).real

        weights = 0.5 * torch.sigmoid(self.projection(x).permute(0,2,1).mean(dim=1)).unsqueeze(dim=1) + 0.5
        out = weights * lowFreq_feature + (1 - weights) * (x.permute(0,2,1) - lowFreq_feature)

        return out.permute(0,2,1)


class ResNet_FNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_FNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.attn1 = LearnableFNetBlock(64, 1024)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.attn2 = LearnableFNetBlock(128, 256)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.attn3 = LearnableFNetBlock(256, 64)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.attn4 = LearnableFNetBlock(512, 16)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, epoch=120):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        b, c, h, w = out.shape
        out = self.attn1(out.view(b, c, h*w), epoch) 

        out = self.layer2(out.view(b, c, h, w))
        b, c, h, w = out.shape
        out = self.attn2(out.view(b, c, h*w), epoch) 

        out = self.layer3(out.view(b, c, h, w))
        b, c, h, w = out.shape
        out = self.attn3(out.view(b, c, h*w), epoch) 

        out = self.layer4(out.view(b, c, h, w))
        b, c, h, w = out.shape
        out = self.attn4(out.view(b, c, h*w), epoch) 
        
        out = F.avg_pool2d(out.view(b, c, h, w), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18_FNet(num_classes=10):
    return ResNet_FNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


