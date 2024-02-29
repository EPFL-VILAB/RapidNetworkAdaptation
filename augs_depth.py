## Here we add the augs for depth: simulated colmap, simkinect etc. 

import argparse
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional
import h5py
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import collections.abc as collections
import PIL.Image

from colmap_simulate import extractors#, logger
from colmap_simulate.utils.base_model import dynamic_load
from colmap_simulate.utils.tools import map_tensor
#from colmap_simulate.utils.parsers import parse_image_lists
#from colmap_simulate.utils.io import read_image, list_h5_names

import matplotlib.pyplot as plt
import random
import pickle as pkl
import torchvision

import pdb


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    'superpoint_max': {
        'output': 'feats-superpoint-n4096-rmax1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'r2d2': {
        'output': 'feats-r2d2-n5000-r1024',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 5000,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'sift': {
        'output': 'feats-sift',
        'model': {
            'name': 'dog'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'sosnet': {
        'output': 'feats-sosnet',
        'model': {
            'name': 'dog',
            'descriptor': 'sosnet'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    # Global descriptors
    'dir': {
        'output': 'global-feats-dir',
        'model': {'name': 'dir'},
        'preprocessing': {'resize_max': 1024},
    },
    'netvlad': {
        'output': 'global-feats-netvlad',
        'model': {'name': 'netvlad'},
        'preprocessing': {'resize_max': 1024},
    },
    'openibl': {
        'output': 'global-feats-openibl',
        'model': {'name': 'openibl'},
        'preprocessing': {'resize_max': 1024},
    }
}

conf = confs['superpoint_inloc']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Model = dynamic_load(extractors, conf['model']['name'])
model = Model(conf['model']).eval().to(device)

def colmap_sim_aug(rgb_batch, depth_batch, max_std=0.17):
    gray_input = rgb_batch.mean(1,keepdim=True) #preprocessing of superpoint involves grayscaling!

    # get keypoint preds
    data_input = {}
    data_input['image'] = gray_input
    with torch.no_grad():
        pred = model(map_tensor(data_input, lambda x: x.to(device)))

    keypoints = pred['keypoints']

    noisy_colmap_sim_depth = torch.Tensor().cuda()
    for image_ind in range(rgb_batch.size(0)):
        keypoint_curr = keypoints[image_ind].cpu().to(torch.int64)
        depth_sparse_full_keypoints = depth_batch[image_ind,:]

        #keypoint_mask = torch.zeros(depth_sparse_full_keypoints.size()).cuda()
        #keypoint_mask[:,keypoint_curr[:, 1], keypoint_curr[:, 0]] = 1.

        #depth_sparse_full_keypoints = gt_depth[image_ind,:] * keypoint_mask

        ###############################
        ### Not all the keypoints will be matched -- Sparsify full keypoints by randomly discarding some of them ####
        randomness_ratio = 0.45 * torch.rand(1) + 0.05 ## between 5% to 50% masking
        keep_percentage = 1. - randomness_ratio
        # keep_keypoint_indices = random.sample(range(len(keypoint_curr[:, 0])), int((keep_percentage * len(keypoint_curr[:, 0])).round()) )
        # keypoint_curr_sparse = keypoint_curr[keep_keypoint_indices, :]
        keypoint_curr_sparse = keypoint_curr

        keypoint_mask_sparse = torch.zeros(depth_sparse_full_keypoints.size()).cuda()
        keypoint_mask_sparse[:,keypoint_curr_sparse[:, 1], keypoint_curr_sparse[:, 0]] = 1.

        depth_sparse_sparse_keypoints = depth_batch[image_ind,:] * keypoint_mask_sparse

        #### Add some noise to sparse gt depth from sparse keypoint locs
        noise_or_clean = int(torch.rand(1)>0.20) # if 1 add noise, wp 0.80
        noise_std = max_std * torch.rand(1) ## max std is 0.17
        noise_added = noise_or_clean * noise_std * torch.randn(depth_sparse_sparse_keypoints.size())
        # depth_sparse_sparse_keypoints_noisy = depth_sparse_sparse_keypoints + noise_added.cuda()
        depth_sparse_sparse_keypoints_noisy = noise_added.cuda()  ## noise only -- for debugging
        depth_sparse_sparse_keypoints_noisy = depth_sparse_sparse_keypoints_noisy * keypoint_mask_sparse # we dont need noise for zero entries
        depth_sparse_sparse_keypoints_noisy = depth_sparse_sparse_keypoints_noisy.clamp(min=0.) #we dont want negative depth


        noisy_colmap_sim_depth = torch.cat((noisy_colmap_sim_depth, depth_sparse_sparse_keypoints_noisy.unsqueeze(0)), dim = 0)

    return noisy_colmap_sim_depth




if __name__ == "__main__":
	## Load sample data
    data_rgb_gtdepth = pkl.load(open('./data_rgb_gtdepth.pkl','rb'))

    rgb_input = data_rgb_gtdepth['x']
    gt_depth = data_rgb_gtdepth['y^']

    noisy_colmap_sim = colmap_sim_aug(rgb_input, gt_depth)

    torchvision.utils.save_image(rgb_input, 'x.png')
    torchvision.utils.save_image(gt_depth, 'y.png')
    torchvision.utils.save_image(noisy_colmap_sim, 'y_n.png')



