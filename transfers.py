
import os, sys, math, random, itertools, functools
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as util_checkpoint
from torchvision import models

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import get_task, task_map, get_model, Task, RealityTask

from modules.unet import UNet, UNet_adapt
from modules.encoder import Encoder

from fire import Fire
import IPython

import pdb


pretrained_transfers = {

    ('rgb', 'normal'):
        (lambda: UNet_adapt(out_channels=3,film_layer=True, spatial_film=True).cuda(), None),
    ('rgb', 'depth_zbuffer'):
        (lambda: UNet_adapt(out_channels=1,film_layer=True, spatial_film=True).cuda(), None),
    ('normal', 'rgb'):
        (lambda: Encoder(downsample=3, in_channels=3).cuda(), None),
    ('depth_zbuffer', 'rgb'):
        (lambda: Encoder(downsample=3, in_channels=2).cuda(), None),

}

class Transfer(nn.Module):

    def __init__(self, src_task, dest_task,
        checkpoint=True, name=None, model_type=None, path=None,
        pretrained=True, finetuned=False
    ):
        super().__init__()
        if isinstance(src_task, str) and isinstance(dest_task, str):
            src_task, dest_task = get_task(src_task), get_task(dest_task)

        self.src_task, self.dest_task, self.checkpoint = src_task, dest_task, checkpoint
        self.name = name or f"{src_task.name}2{dest_task.name}"
        saved_type, saved_path = None, None
        if model_type is None and path is None:
            saved_type, saved_path = pretrained_transfers.get((src_task.name, dest_task.name), (None, None))

        self.model_type, self.path = model_type or saved_type, path or saved_path
        self.model = None

        if self.model_type is None:

            if src_task.kind == dest_task.kind and src_task.resize != dest_task.resize:

                class Module(TrainableModel):

                    def __init__(self):
                        super().__init__()

                    def forward(self, x):
                        return resize(x, val=dest_task.resize)

                self.model_type = lambda: Module()
                self.path = None

        if not pretrained:
            print ("Not using pretrained [heavily discouraged]")
            self.path = None

    def load_model(self):
        if self.model is None:
            if self.path is not None:
                self.model = DataParallelModel.load(self.model_type().to(DEVICE), self.path)
                # if optimizer:
                #     self.model.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
            else:
                self.model = self.model_type()
                if isinstance(self.model, nn.Module):
                    self.model = DataParallelModel(self.model)
        return self.model

    def __call__(self, *args, **kwargs):
        self.load_model()
        # preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        preds = util_checkpoint(self.model, *args, **kwargs) if self.checkpoint else self.model(*args, **kwargs)
        # preds.task = self.dest_task
        return preds

    def __repr__(self):
        return self.name or str(self.src_task) + " -> " + str(self.dest_task)


class RealityTransfer(Transfer):

    def __init__(self, src_task, dest_task):
        super().__init__(src_task, dest_task, model_type=lambda: None)

    def load_model(self, optimizer=True):
        pass

    def __call__(self, x):
        assert (isinstance(self.src_task, RealityTask))
        return self.src_task.task_data[self.dest_task].to(DEVICE)


functional_transfers = (
    Transfer('rgb', 'normal', name='n'),
    Transfer('rgb', 'depth_zbuffer', name='d'),
)


TRANSFER_MAP = {t.name:t for t in functional_transfers}
functional_transfers = namedtuple('functional_transfers', TRANSFER_MAP.keys())(**TRANSFER_MAP)

def get_transfer_name(transfer):
    for t in functional_transfers:
        if transfer.src_task == t.src_task and transfer.dest_task == t.dest_task:
            return t.name
    return transfer.name

(n, d) = functional_transfers

if __name__ == "__main__":
    y = g(F(f(x)))
    print (y.shape)






