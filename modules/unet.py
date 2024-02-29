
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from models import TrainableModel
from utils import *

import pdb


class FiLM_spatial(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        # # self.conv_alpha = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        # self.conv_beta = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.conv_beta = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x_size, embed):
        if embed.size(-1) != x_size[-1]:
            embed = F.interpolate(
                embed, 
                size=(x_size[-2], x_size[-1]), 
                mode='bilinear',
                align_corners=False
            )
        # breakpoint()
        # embed = self.relu(self.conv1(embed))
        # alpha = (self.conv_alpha(embed))
        beta = (self.conv_beta(embed))
        return beta
    
class FiLM(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # self.alphas = nn.ModuleList(
        #     [nn.Linear(input_channel, output_channel) for i in range(3)]
        # )
        self.betas = nn.ModuleList(
            [nn.Linear(input_channel, output_channel) for i in range(3)]
        )
        # self.beta = nn.Linear(input_channel, output_channel)
        # self.alpha = nn.Linear(input_channel, output_channel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = torch.nn.ReLU()

    def forward(self, x_size, embed):
        # alphas = []
        betas = []
        embed = self.gap(embed)
        embed = embed.view(embed.size(0), -1)
        
        for i in range(3):
            # alpha = self.alphas[i](embed)
            # alphas.append(alpha[:,:,None,None])
            beta = self.betas[i](embed)
            betas.append(beta[:,:,None,None])

        # beta = self.beta(embed)[:,:,None,None]
        # alpha = self.alpha(embed)[:,:,None,None]

        # return alphas, betas
        return betas
        # return alpha, beta


class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True, film_layer=False, embed_channel=None, spatial_film=True):
        super().__init__()
        # print(film_layer)
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        if film_layer: 
            if spatial_film:
                self.film = FiLM_spatial(embed_channel, output_channel)
            else:
                self.film = FiLM(embed_channel, output_channel)
        # if film_layer: 
        #     self.film_alpha = nn.Parameter(torch.rand(1,output_channel,1,1)*0.01-0.005)
        #     self.film_beta = nn.Parameter(torch.rand(1,output_channel,1,1)*0.01-0.005)
        #     # self.film_alpha = nn.Parameter(torch.Tensor(1,output_channel,1,1).fill_(0.))
        #     # self.film_beta = nn.Parameter(torch.Tensor(1,output_channel,1,1).fill_(0.))
        self.relu = torch.nn.ReLU()
        self.up_sample = up_sample
        self.film_layer = film_layer

    def forward(self, prev_feature_map, x, embed=None):
        if self.up_sample:
            x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # if self.film_layer and embed is not None: x = self.film(x, embed)
        if self.film_layer and embed is not None: 
            beta = self.film(x.size(), embed)
            x = x + beta
            # x = x * (1. + alpha) + beta
        x = self.relu(x)

        # if self.film_layer and embed is not None: 
        #     betas = self.film(x.size(), embed)
        #     # alphas, betas = self.film(x.size(), embed)
        # x = (self.bn1(self.conv1(x)))
        # if self.film_layer and embed is not None: 
        #     x = x + betas[0]
        #     # x = x * (1 + alphas[0]) + betas[0]
        # x = self.relu(x)
        # x = (self.bn2(self.conv2(x)))
        # if self.film_layer and embed is not None: 
        #     x = x + betas[1]
        #     # x = x * (1 + alphas[1]) + betas[1]
        # x = self.relu(x)
        # x = self.bn3(self.conv3(x))
        # if self.film_layer and embed is not None: 
        #     x = x + betas[2]
        #     # x = x * (1 + alphas[2]) + betas[2]
        # # if self.film_layer and embed is not None: x = self.film(x, embed)
        # x = self.relu(x)

        return x


class UNet_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size=True, film_layer=False, embed_channel=None, spatial_film=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        # if film_layer: self.film = FiLM_spatial(embed_channel, output_channel)
        if film_layer: 
            if spatial_film:
                self.film = FiLM_spatial(embed_channel, output_channel)
            else:
                self.film = FiLM(embed_channel, output_channel)
        # if film_layer: 
        #     self.film_alpha = nn.Parameter(torch.rand(1,output_channel,1,1)*0.01-0.005)
        #     self.film_beta = nn.Parameter(torch.rand(1,output_channel,1,1)*0.01-0.005)
        #     # self.film_alpha = nn.Parameter(torch.Tensor(1,output_channel,1,1).fill_(0.))
        #     # self.film_beta = nn.Parameter(torch.Tensor(1,output_channel,1,1).fill_(0.))
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size
        self.film_layer = film_layer

    def forward(self, x, embed=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # if self.film_layer and embed is not None: x = self.film(x, embed)
        if self.film_layer and embed is not None: 
            beta = self.film(x.size(), embed)
            x = x + beta
            # x = x * (1. + alpha) + beta
        x = self.relu(x)

        # if self.film_layer and embed is not None: 
        #     # alphas, betas = self.film(x.size(), embed)
        #     betas = self.film(x.size(), embed)
        # x = (self.bn1(self.conv1(x)))
        # if self.film_layer and embed is not None: 
        #     x = x + betas[0]
        #     # x = x * (1 + alphas[0]) + betas[0]
        # x = self.relu(x)
        # x = (self.bn2(self.conv2(x)))
        # if self.film_layer and embed is not None: 
        #     x = x + betas[1]
        #     # x = x * (1 + alphas[1]) + betas[1]
        # x = self.relu(x)
        # x = self.bn3(self.conv3(x))
        # if self.film_layer and embed is not None: 
        #     x = x + betas[2]
        #     # x = x * (1 + alphas[2]) + betas[2]
        # x = self.relu(x)

        if self.down_size:
            x = self.max_pool(x)
        return x

class UNet_up_block_proxy(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        # self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        # self.bn2 = nn.GroupNorm(8, output_channel)
        # self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        # self.bn3 = nn.GroupNorm(8, output_channel)
        self.relu = torch.nn.ReLU()
        self.up_sample = up_sample

    def forward(self, prev_feature_map, x):
        if self.up_sample:
            x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        return x

# class UNet_up_block(nn.Module):
#     def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
#         super().__init__()
#         self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.down_sampling = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
#         self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
#         self.bn1 = nn.GroupNorm(8, output_channel)
#         self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
#         self.bn2 = nn.GroupNorm(8, output_channel)
#         self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
#         self.bn3 = nn.GroupNorm(8, output_channel)
#         self.relu = torch.nn.ReLU()
#         self.up_sample = up_sample

#     def forward(self, prev_feature_map, x):
#         if self.up_sample:
#             x = self.up_sampling(x)
#         else:
#             prev_feature_map = self.down_sampling(prev_feature_map)
#         x = torch.cat((x, prev_feature_map), dim=1)
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         return x


# class UNet_down_block(nn.Module):
#     def __init__(self, input_channel, output_channel, down_size=True):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
#         self.bn1 = nn.GroupNorm(8, output_channel)
#         self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
#         self.bn2 = nn.GroupNorm(8, output_channel)
#         self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
#         self.bn3 = nn.GroupNorm(8, output_channel)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#         self.down_size = down_size

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         if self.down_size:
#             x = self.max_pool(x)
#         return x


class UNet_adapt(TrainableModel):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3, film_layer=False, embed_channel=128, spatial_film=True):
        super().__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True, film_layer=film_layer, embed_channel=embed_channel, spatial_film=spatial_film) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i), film_layer=film_layer, embed_channel=embed_channel, spatial_film=spatial_film) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x, embed=None):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x, embed=embed)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x, embed=embed)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

class UNet(TrainableModel):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3, up_sample=[True]):
        super().__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        if len(up_sample)<downsample: up_sample = up_sample*downsample
        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i), up_sample=up_sample[i]) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        # self.last_conv2_rho = nn.Conv2d(16, 3, 1, padding=0)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class UNet_2dec(TrainableModel):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3, proxy_out_channels=1):
        super().__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.up_blocks_proxy = nn.ModuleList(
            [UNet_up_block_proxy(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)

        self.proxy_last_conv = nn.Conv2d(16, proxy_out_channels, 1, padding=0)
        
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        x_proxy = x.clone()

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
        
        for i in range(0, self.downsample)[::-1]:
            x_proxy = self.up_blocks_proxy[i](xvals[i], x_proxy)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)

        x_proxy = self.relu(self.proxy_last_conv(x_proxy))

        return torch.cat((x,x_proxy),dim=1)
    
    
class UNet_4dec(TrainableModel):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3, film_layer=False, embed_channel=128, proxy_out_channels=1):
        super().__init__()
        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True, film_layer=film_layer, embed_channel=embed_channel) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i), film_layer=film_layer, embed_channel=embed_channel) for i in range(0, downsample)]
        )

        self.up_blocks_proxy1 = nn.ModuleList(
            [UNet_up_block_proxy(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )
        self.up_blocks_proxy2 = nn.ModuleList(
            [UNet_up_block_proxy(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )
        self.up_blocks_proxy3 = nn.ModuleList(
            [UNet_up_block_proxy(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)

        self.proxy_last_conv1 = nn.Conv2d(16, proxy_out_channels, 1, padding=0)
        self.proxy_last_conv2 = nn.Conv2d(16, proxy_out_channels, 1, padding=0)
        self.proxy_last_conv3 = nn.Conv2d(16, proxy_out_channels, 1, padding=0)
        
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x, embed=None):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x, embed=embed)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        x_proxy1 = x.clone()
        x_proxy2 = x.clone()
        x_proxy3 = x.clone()

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x, embed=embed)
        
        for i in range(0, self.downsample)[::-1]:
            x_proxy1 = self.up_blocks_proxy1[i](xvals[i], x_proxy1)
        for i in range(0, self.downsample)[::-1]:
            x_proxy2 = self.up_blocks_proxy2[i](xvals[i], x_proxy2)
        for i in range(0, self.downsample)[::-1]:
            x_proxy3 = self.up_blocks_proxy3[i](xvals[i], x_proxy3)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))

        x_proxy1 = self.relu(self.proxy_last_conv1(x_proxy1))
        x_proxy2 = self.relu(self.proxy_last_conv2(x_proxy2))
        x_proxy3 = self.relu(self.proxy_last_conv3(x_proxy3))

        return torch.cat((x,x_proxy1,x_proxy2,x_proxy3),dim=1)
    
class UNet_aug(TrainableModel):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        # self.last_conv2_rho = nn.Conv2d(16, 3, 1, padding=0)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        
        blur_k=random.choice([3,5,7,9,11])
        dropout_p=random.uniform(0,0.7)
        dropout2d_p=random.uniform(0,0.4)
#         dropout_p=0
#         dropout2d_p=0
#         blur_k=0

        if blur_k>0:
            conv_smooth = nn.Conv2d(1,1,kernel_size=blur_k,padding=(blur_k-1)//2,padding_mode='reflect').cuda()
            conv_smooth.weight=torch.nn.Parameter(torch.ones_like(conv_smooth.weight)/(blur_k*blur_k))
            conv_smooth.bias=torch.nn.Parameter(torch.zeros_like(conv_smooth.bias))
        
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))
        

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)
            
            if blur_k>0: x = conv_smooth(x.view(-1,1,x.size(-2),x.size(-1))).view(x.size(0),-1,x.size(-2),x.size(-1))
            if dropout2d_p>0: x = F.dropout2d(x,training=True,p=dropout2d_p)/(1./(1-dropout2d_p))        
            if dropout_p>0: x = F.dropout(x,training=True,p=dropout_p)/(1./(1-dropout_p))
#             out_filters = x.size(1)
#             to_zero = list(set([random.choice(list(range(out_filters))) for _ in range(out_filters // 5)]))
#             x[:,to_zero] *= -1.
#             x=torch.flip(x,(1,))
#             _k=random.randint(1, 3)
#             x=torch.rot90(x,_k,(2,3))


        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        x = torch.clamp(x, min=0.,max=1.)

        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)
    

class UNetReshade(TrainableModel):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        x = x.clamp(max=1, min=0).mean(dim=1, keepdim=True)
        x = x.expand(-1, 3, -1, -1)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class UNetOld(TrainableModel):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.down_block1 = UNet_down_block(in_channels, 16, False) #   256
        self.down_block2 = UNet_down_block(16, 32, True) #   128
        self.down_block3 = UNet_down_block(32, 64, True) #   64
        self.down_block4 = UNet_down_block(64, 128, True) #  32
        self.down_block5 = UNet_down_block(128, 256, True) # 16
        self.down_block6 = UNet_down_block(256, 512, True) # 8
        self.down_block7 = UNet_down_block(512, 1024, True)# 4

        self.mid_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, 1024)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, 1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, 1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)

        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))

        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class ConvBlock(nn.Module):
    def __init__(self, f1, f2, kernel_size=3, padding=1, use_groupnorm=True, groups=8, dilation=1, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.conv = nn.Conv2d(f1, f2, (kernel_size, kernel_size), dilation=dilation, padding=padding*dilation)
        if self.transpose:
            self.convt = nn.ConvTranspose2d(
                f1, f1, (3, 3), dilation=dilation, stride=2, padding=dilation, output_padding=1
            )
        if use_groupnorm:
            self.bn = nn.GroupNorm(groups, f1)
        else:
            self.bn = nn.BatchNorm2d(f1)

    def forward(self, x):
        # x = F.dropout(x, 0.04, self.training)
        x = self.bn(x)
        if self.transpose:
            # x = F.upsample(x, scale_factor=2, mode='bilinear')
            x = F.relu(self.convt(x))
            # x = x[:, :, :-1, :-1]
        x = F.relu(self.conv(x))
        return x

class UNetOld2(TrainableModel):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.initial = nn.Sequential(
            ConvBlock(in_channels, 16, groups=3, kernel_size=1, padding=0),
            ConvBlock(16, 16, groups=4, kernel_size=1, padding=0)
        )
        self.down_block1 = UNet_down_block(16, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True) #   128
        self.down_block3 = UNet_down_block(32, 64, True) #   64
        self.down_block4 = UNet_down_block(64, 128, True) #  32
        self.down_block5 = UNet_down_block(128, 256, True) # 16
        self.down_block6 = UNet_down_block(256, 512, True) # 8
        self.down_block7 = UNet_down_block(512, 1024, True)# 4

        self.mid_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, 1024)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, 1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, 1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.initial(x)
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)

        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))

        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

class UNet_branched(TrainableModel):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        # self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.last_conv2 = nn.Conv2d(16, 16, 3, padding=1)

        #branch-main
        self.last_conv_b11 = nn.Conv2d(16, 8, 3, padding=1)
        self.last_conv_b12 = nn.Conv2d(8, 3, 1, padding=0)

        #branch-p1
        self.last_conv_b21 = nn.Conv2d(16, 8, 3, padding=1)
        self.last_conv_b22 = nn.Conv2d(8, 1, 1, padding=0)
        
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x_ = self.relu(self.last_conv2(x))

        #branch-main
        # pdb.set_trace()
        x_b1 = self.relu(self.last_conv_b11(x_))
        x_b1 = self.relu(self.last_conv_b12(x_b1))
        #branch-p1
        x_b2 = self.relu(self.last_conv_b21(x_))
        x_b2 = self.relu(self.last_conv_b22(x_b2))
        # x[:,:(self.out_channels//2)] = self.relu(x[:,:(self.out_channels//2)].clone())

        return torch.cat((x_b1,x_b2),dim=1)

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)