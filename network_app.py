import re
import math
#import pywt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo

# from IPython import embed
from copy import deepcopy
from collections import OrderedDict
from torch.autograd import Variable

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet152'             : 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Network:
    def construct(self, net, obj, prefix=''):
        targetClass = getattr(self, net)
        instance = targetClass(obj)
        return instance

    ###########################################################################
    #############################      ResNet      ############################
    ###########################################################################
    
    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
    
    @staticmethod
    def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
    
    class BasicBlock(nn.Module):
        expansion = 1
    
        def __init__(self, inplanes, planes, last, stride=1, downsample=None):
            super(Network.BasicBlock, self).__init__()
            self.conv1 = Network.conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU(inplace=True)
            
            self.conv2 = Network.conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            if last:
                self.relu2 = nn.ReLU(inplace=False)
            else:
                self.relu2 = nn.ReLU(inplace=True)
            
            self.downsample = downsample
            self.stride = stride
    
        def forward(self, x):
            residual = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
    
            if self.downsample is not None:
                residual = self.downsample(x)
    
            out += residual
            out = self.relu2(out)
    
            return out
        
        def get_imp_layers(self):
            return [self.relu1, self.relu2]
        
        def get_conv_layers(self):
            return [self.conv1, self.conv2]
        
        def get_bn_layers(self):
            return [self.bn1, self.bn2]
    
    
    class Bottleneck(nn.Module):
        expansion = 4
    
        def __init__(self, inplanes, planes, last, stride=1, downsample=None):
            super(Network.Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
            if last:
                self.relu3 = nn.ReLU(inplace=False)
            else:
                self.relu3 = nn.ReLU(inplace=True)
                
            self.downsample = downsample
            self.stride = stride
    
        def forward(self, x):
            residual = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
    
            out = self.conv3(out)
            out = self.bn3(out)
    
            if self.downsample is not None:
                residual = self.downsample(x)
    
            out += residual
            out = self.relu3(out)
    
            return out
    
        def get_imp_layers(self):
            return [self.relu1, self.relu2, self.relu3]
        
        def get_conv_layers(self):
            return [self.conv1, self.conv2, self.conv3]
        
        def get_bn_layers(self):
            return [self.bn1, self.bn2, self.bn3]
    
    class ResNet(nn.Module):
    
        def __init__(self, obj, block, layers):
            self.obj = obj
            self.inplanes = 64
            super(Network.ResNet, self).__init__()
            
            if obj.resnet_type == 'big':
                self.conv1 = nn.Conv2d(obj.input_ch, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
            elif obj.resnet_type == 'small':
                self.conv1 = Network.conv3x3(obj.input_ch, 64)
                
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
            
            if obj.resnet_type == 'big':
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            block_widths = [64,128,256,512]
            if hasattr(obj,'width') and obj.width:
                assert 'Wide' in self.__class__.__name__
                for i in range(len(block_widths)-1):
                    block_widths[i] = int(block_widths[i] * obj.width)

            self.layer1, relus1, convs1, bns1 = self._make_layer(block, block_widths[0], layers[0])
            self.layer2, relus2, convs2, bns2 = self._make_layer(block, block_widths[1], layers[1], stride=2)
            self.layer3, relus3, convs3, bns3 = self._make_layer(block, block_widths[2], layers[2], stride=2)
            self.layer4, relus4, convs4, bns4 = self._make_layer(block, block_widths[3], layers[3], stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, obj.num_classes)
            
            self.relus = [self.relu1] + relus1 + relus2 + relus3 + relus4
            self.convs = [self.conv1] + convs1 + convs2 + convs3 + convs4
            self.bns = [self.bn1] + bns1 + bns2 + bns3 + bns4
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    
            if obj.pretrained:
                print("Using Pre-trained PyTorch Model (ResNet)")
                self.fc = nn.Linear(512 * block.expansion, 1000)
                self.load_state_dict(model_zoo.load_url(model_urls[obj.net.lower()]))
                if obj.reset_classifier:
                    self.fc = nn.Linear(512 * block.expansion, obj.num_classes)
        
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
    
            layers = []
            relus = []
            convs = []
            bns = []
            
            cur_layers = block(self.inplanes, planes, False, stride, downsample)
            cur_relus = cur_layers.get_imp_layers()
            cur_convs = cur_layers.get_conv_layers()
            cur_bns = cur_layers.get_bn_layers()

            layers.append(cur_layers)
            relus += cur_relus
            convs += cur_convs
            bns += cur_bns
            
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                
                last = (planes == 512) & (i == blocks-1)
                
                cur_layers = block(self.inplanes, planes, last)
                cur_relus = cur_layers.get_imp_layers()
                cur_convs = cur_layers.get_conv_layers()
                cur_bns = cur_layers.get_bn_layers()
                
                layers.append(cur_layers)
                relus += cur_relus
                convs += cur_convs
                bns += cur_bns
    
            return nn.Sequential(*layers), relus, convs, bns
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
            if hasattr(self, 'maxpool'):
                x = self.maxpool(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            
            x = x.view(x.size(0), -1)
            
            x = self.fc(x)
            
            return x
        
        def get_imp_layers(self):
            return [self.fc]
        
        def get_relu_layers(self):
            return self.relus
        
        def get_conv_layers(self):
            return self.convs + [self.fc]
        
        def get_bn_layers(self):
            return self.bns
            
    class ResNet18(ResNet):
        def __init__(self, obj):
            super(Network.ResNet18, self).__init__(obj, Network.BasicBlock, [2, 2, 2, 2])
    
    class ResNet50(ResNet):
        def __init__(self, obj):
            super(Network.ResNet50, self).__init__(obj, Network.Bottleneck, [3, 4, 6, 3])

    class ResNet152(ResNet):
        def __init__(self, obj):
            super(Network.ResNet152, self).__init__(obj, Network.Bottleneck, [3, 8, 36, 3])