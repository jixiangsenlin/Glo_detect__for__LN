import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class V_c_net(nn.Module):

    def __init__(self, name, pretraining=True, num_class=4):
        super(V_c_net, self).__init__()
        self.pretraining = pretraining
        # # vgg11
        # self.net_1 = models.vgg11(pretrained=self.pretraining).features
        # self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
        # self.net_2._modules['0'] = nn.Linear(131072, 4096)
        # self.net_2._modules['6'] = nn.Linear(4096, 40)

        # # vgg11
        # self.net_1 = models.vgg11(pretrained=self.pretraining).features
        # self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
        # self.net_2._modules['0'] = nn.Linear(131072, 4096)
        # self.net_2._modules['6'] = nn.Linear(4096, num_class)

        # vgg16
        self.net_1 = models.vgg16(pretrained=self.pretraining).features
        self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
        self.net_2._modules['0'] = nn.Linear(131072, 4096)
        self.net_2._modules['6'] = nn.Linear(4096, num_class)#40

        # # vgg19
        # self.net_1 = models.vgg19(pretrained=self.pretraining).features
        # self.net_2 = models.vgg19(pretrained=self.pretraining).classifier
        # self.net_2._modules['0'] = nn.Linear(131072, 4096)
        # self.net_2._modules['6'] = nn.Linear(4096, num_class)#40


    def forward(self, x):
        y = self.net_1(x)
        y = self.net_2(y.view(y.shape[0], -1))
        return y


