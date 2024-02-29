import os, sys, math, random, itertools
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from task_configs import tasks, get_task, ImageTask
from transfers import functional_transfers, get_transfer_name, Transfer
from datasets import TaskDataset, load_train_val

from matplotlib.cm import get_cmap

import IPython

def get_energy_loss(
    config="",
    pretrained=True, finetuned=True, **kwargs,
):
    """ Loads energy loss from config dict. """
    return DefaultEnergyLoss(**energy_configs[config],
        pretrained=pretrained, finetuned=finetuned, **kwargs
    )


energy_configs = {

    "baseline_depth_loop_film": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "mask": [tasks.mask_valid],
            "n(x)": [tasks.rgb, tasks.depth_zbuffer],
            "n(x1)": [tasks.rgb, tasks.depth_zbuffer, tasks.rgb, tasks.depth_zbuffer],
        },
        "freeze_list": [],
        "losses": {
            "main": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "main_t1": {
                ("train", "val"): [
                    ("n(x1)", "y^"),
                ],
            },
            "main_dummy": {
                ("train", "val"): [
                    ("mask", "mask"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test",""),
                paths=[
                    "x",
                    "y^",
                    "mask",
                    "n(x)",
                    "n(x1)",
                ]
            ),
        },
    },


}



def coeff_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1


class EnergyLoss(object):

    def __init__(self, paths, losses, plots,
        pretrained=True, finetuned=False, freeze_list=[]
    ):

        self.paths, self.losses, self.plots = paths, losses, plots
        self.freeze_list = [str((path[0].name, path[1].name)) for path in freeze_list]
        self.metrics = {}

        self.tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    self.tasks += self.paths[path1] + self.paths[path2]

        for name, config in self.plots.items():
            for path in config["paths"]:
                self.tasks += self.paths[path]
        self.tasks = list(set(self.tasks))

    def compute_paths(self, graph, reality=None, paths=None):
        path_cache = {}
        paths = paths or self.paths
        
        paths_keys = sorted(paths, key=lambda k: len(paths[k]), reverse=False)
        paths_sorted = {k:paths[k] for k in paths_keys}
        
        path_values = {
            name: graph.sample_path(path,
                reality=reality, use_cache=True, cache=path_cache, name=name
            ) for name, path in paths_sorted.items()
        }
        
        del path_cache
        return {k: v for k, v in path_values.items() if v is not None}

    def get_tasks(self, reality):
        tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                if reality in realities:
                    for path1, path2 in losses:
                        tasks += [self.paths[path1][0], self.paths[path2][0]]

        for name, config in self.plots.items():
            if reality in config["realities"]:
                for path in config["paths"]:
                    tasks += [self.paths[path][0]]

        return list(set(tasks))

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, batch_mean=True, use_l1=False):
        loss = {}
        for reality in realities:
            loss_dict = {}
            losses = []
            all_loss_types = set()
            for loss_type, loss_item in self.losses.items():
                all_loss_types.add(loss_type)
                loss_dict[loss_type] = []
                for realities_l, data in loss_item.items():
                    if reality.name in realities_l:
                        loss_dict[loss_type] += data
                        if loss_types is not None and loss_type in loss_types:
                            losses += data

            path_values = self.compute_paths(graph,
                paths={
                    path: self.paths[path] for path in \
                    set(path for paths in losses for path in paths)
                    },
                reality=reality)

            if reality.name not in self.metrics:
                self.metrics[reality.name] = defaultdict(list)

            # mask = ImageTask.build_mask(path_values["y^"][:,:3], val=self.paths['y^'][0].mask_val).float()[:,:1]
            mask = path_values["mask"]
            for loss_type, losses in sorted(loss_dict.items()):
                if loss_type not in (loss_types or all_loss_types):
                    continue
                if loss_type == 'main_dummy': continue
                if loss_type not in loss:
                    loss[loss_type] = 0
                for path1, path2 in losses:
                    output_task = self.paths[path1][-1]
                    if loss_type not in loss:
                        loss[loss_type] = 0
                    for path1, path2 in losses:
                        output_task = self.paths[path1][-1]
                        # breakpoint()
                        path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean, compute_mse=False, mask=mask)
                        if loss_type == "main_t1": loss[loss_type] += path_loss
                        # if loss_type == "main": loss[loss_type] += path_loss
                        self.metrics[reality.name]["mae : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]
                        path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean, compute_mse=True, mask=mask)
                        self.metrics[reality.name]["mse : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

        return loss

    def logger_hooks(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    if (loss_type=='nll'):
                        name = "nll : "+path1 + " -> " + path2
                        name_to_realities[name] += list(realities)
                    name = "mae : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    name = "mse : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):
                names = [f"{reality}_{name}" for reality in realities]
                if not all(x in data for x in names):
                    return
                data = np.stack([data[x] for x in names], axis=1)
                logger.plot(data, name, opts={"legend": names})

            logger.add_hook(partial(jointplot, name=name, realities=realities), feature=f"{realities[-1]}_{name}", freq=1)


    def logger_update(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    if (loss_type=='nll'):
                        name = "nll : "+path1 + " -> " + path2
                        name_to_realities[name] += list(realities)
                    name = "mae : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    name = "mse : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            for reality in realities:
                # IPython.embed()
                if reality not in self.metrics: continue
                if name not in self.metrics[reality]: continue
                if len(self.metrics[reality][name]) == 0: continue

                logger.update(
                    f"{reality}_{name}",
                    torch.mean(torch.stack(self.metrics[reality][name])),
                )
        self.metrics = {}

    def plot_paths(self, graph, logger, realities=[], plot_names=None, epochs=0, tr_step=0,prefix=""):


        path_values = {}
        realities_map = {reality.name: reality for reality in realities}
        for name, config in (plot_names or self.plots.items()):
            paths = config["paths"]

            cmap = get_cmap("jet")
            realities = config["realities"]
            ind = np.diag_indices(3)
            for reality in realities:

                if reality == '': continue 

                with torch.no_grad():

                    path_values[reality] = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])

                    if reality is 'test': #compute error map
                        mask_task = self.paths["y^"][-1]
                        mask = ImageTask.build_mask(path_values[reality]["y^"], val=mask_task.mask_val)
                        errors = ((path_values[reality]["y^"][:,:3]-path_values[reality]["n(x1)"][:,:3])**2).mean(dim=1, keepdim=True)
                        # errors = ((path_values[reality]["y^"][:,:3]-path_values[reality]["n(y)"][:,:3])**2).mean(dim=1, keepdim=True)
                        errors = (3*errors/(mask_task.variance)).clamp(min=0, max=1)
                        log_errors = torch.log(errors + 1)
                        log_errors = log_errors / log_errors.max()
                        log_errors = torch.tensor(cmap(log_errors.cpu()))[:, 0].permute((0, 3, 1, 2)).float()[:, 0:3]
                        log_errors = log_errors.clamp(min=0, max=1).to(DEVICE)
                        log_errors[~mask.expand_as(log_errors)] = 0.505
                        path_values[reality]['error']= log_errors

                    for p in  self.plots['']['paths']:
                        if p in ['x']: continue
                        if ('depth' in p or 'normal' in p): continue
                        if (p=='n(x)'): 
                            if reality=='test': path_values[reality][f'y^'] = path_values[reality].pop('y^')

                    path_values[reality] = {k:v.clamp(min=0,max=1).cpu() for k,v in path_values[reality].items()}

        # more processing
        def reshape_img_to_rows(x_):
            downsample = lambda x: F.interpolate(x.unsqueeze(0),scale_factor=0.8,mode='bilinear').squeeze(0)
            x_list = [downsample(x_[i]) for i in range(x_.size(0))]
            x=torch.cat(x_list,dim=-1)
            return x


        all_images = {}
        for reality in realities:
            if reality == '': continue 
            all_imgs_reality = []
            plot_name = ''
            for k in path_values[reality].keys():
                plot_name += k+'_'
                img_row = reshape_img_to_rows(path_values[reality][k])
                if img_row.size(0) == 1: img_row = img_row.repeat(3,1,1)
                all_imgs_reality.append(img_row)
            plot_name = plot_name[:-1]
            all_images[reality+'_'+plot_name] = torch.cat(all_imgs_reality,dim=-2)

        return all_images


    def __repr__(self):
        return str(self.losses)


class DefaultEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 3)
        self.random_select = kwargs.pop('random_select', False)
        self.running_stats = {}
        self.target_task = kwargs['paths']['y^'][0].name

        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
        self.main_losses = [key[5:] for key in self.losses.keys() if key[0:5] == "main_"]
        # print (self.percep_losses)


    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):

        loss_types = ["main"] + [("percep_" + loss) for loss in self.percep_losses] + [("main_" + loss) for loss in self.main_losses]
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, batch_mean=False)
        loss_dict["main_t1"] = loss_dict["main_t1"].mean()
        # loss_dict["main"] = loss_dict["main"].mean()

        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)


