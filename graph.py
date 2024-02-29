import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, WrapperModel
from datasets import TaskDataset
from task_configs import get_task, task_map, tasks, get_model, RealityTask
from transfers import Transfer, RealityTransfer, get_transfer_name

# from augs_depth import colmap_sim_aug

class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, tasks=tasks, edges=None, edges_exclude=None,
        pretrained=True, finetuned=False,
        reality=[], task_filter=[tasks.segment_semantic],
        freeze_list=[], lazy=False, initialize_from_transfer=True,
        target_task=None, noise_max_std=0.17,
    ):

        super().__init__()
        self.tasks = list(set(tasks) - set(task_filter))
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.edge_list, self.edge_list_exclude = edges, edges_exclude
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges, self.adj, self.in_adj = [], defaultdict(list), defaultdict(list)
        self.edge_map, self.reality = {}, reality
        self.initialize_from_transfer = initialize_from_transfer
        print('graph tasks', self.tasks)
        self.params = {}
        self.target_task = target_task

        self.noise_max_std = noise_max_std

        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = (src_task, dest_task)
            if edges is not None and key not in edges: continue
            if edges_exclude is not None and key in edges_exclude: continue
            if src_task == dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            print (src_task, dest_task)
            transfer = None
            if isinstance(src_task, RealityTask):
                if dest_task not in src_task.tasks: continue
                transfer = RealityTransfer(src_task, dest_task)
            else:
                transfer = Transfer(src_task, dest_task,
                    pretrained=pretrained, finetuned=finetuned
                )
                transfer.name = get_transfer_name(transfer)
                if not self.initialize_from_transfer:
                    transfer.path = None
            if transfer.model_type is None:
                continue
            print ("Added transfer", transfer)
            self.edges += [transfer]
            self.adj[src_task.name] += [transfer]
            self.in_adj[dest_task.name] += [transfer]
            self.edge_map[str((src_task.name, dest_task.name))] = transfer
            if isinstance(transfer, nn.Module):
                if str((src_task.name, dest_task.name)) not in freeze_list:
                    self.params[str((src_task.name, dest_task.name))] = transfer
                else:
                    print("freezing " + str((src_task.name, dest_task.name)))
                try:
                    if not lazy: transfer.load_model()
                    if src_task.name == 'rgb' and dest_task.name == target_task:
                        for n,p in transfer.model.named_parameters():
                            if 'film' not in n:
                                p.requires_grad=False
                except:
                    IPython.embed()

        self.params = nn.ModuleDict(self.params)

    def edge(self, src_task, dest_task):
        key1 = str((src_task.name, dest_task.name))
        key2 = str((src_task.kind, dest_task.kind))
        if key1 in self.edge_map: return self.edge_map[key1]
        return self.edge_map[key2]

    def sample_path(self, path, reality=None, use_cache=False, cache={}, name=None):
        path = [reality or self.reality[0]] + path
        x = None
        for i in range(1, len(path)):
            try:
                # breakpoint()
                if path[i-1].name==self.target_task and path[i].name=='rgb':
                    x = x.detach()
                    p = random.uniform(0,0.0025) # if random.uniform(0,1) < 0.8 else 0
                    mask = torch.cuda.FloatTensor(x[:,:1].size()).uniform_() < p
                    # selected images have no gt
                    # p = random.uniform(0,1.)
                    # selected_ind = random.sample(range(x.size(0)),int(p*x.size(0)))
                    # mask[selected_ind] = 0.
                    # rgb = cache[tuple(path[:2])]
                    path_gt = path[0:2]
                    path_gt[-1] = get_task(self.target_task)
                    gt = cache[tuple(path_gt)]
                    # proxy = colmap_sim_aug(rgb,gt,self.noise_max_std)   # colmap
                    path_mask = path[0:2]
                    path_mask[-1] = get_task('mask_valid')
                    task_mask = cache[tuple(path_mask)]
                    # task_mask = get_task(self.target_task).build_mask(gt,val=get_task(self.target_task).mask_val).float()
                    final_mask = task_mask[:,:1]*mask
                    # rgb = cache[tuple(path[:2])]
                    # path_edge = path[0:3]
                    # path_edge[-1] = get_task('sobel_edges')
                    # edges = cache[tuple(path_edge)]
                    # x = torch.cat((gt*final_mask,x[:,:3]*task_mask,edges),dim=1)
                    x = torch.cat((gt*final_mask,x*final_mask),dim=1)
                    # x = torch.cat((proxy*task_mask,x[:,:3]*task_mask),dim=1)
                    # x = proxy*task_mask
                    # x = task_mask*0.
                    # x = gt*final_mask
                    # x = torch.cat((gt*final_mask,edges),dim=1)
                    
                    # x = cache[tuple(path[:2])]
                    
                ## used for renorm film layers
                if i>=4 and path[i-1].name[:3]=='rgb' and path[i].name==self.target_task:
                    rgb = cache[tuple(path[:2])]
                    x = self.edge(path[i-1], path[i])(rgb,x)
                    cache[tuple(path[0:(i+1)])] = x
                    continue          
                x = cache.get(tuple(path[0:(i+1)]),
                    self.edge(path[i-1], path[i])(x)
                )
            except KeyError:
                return None
            except Exception as e:
                print(e)
                breakpoint()

            if use_cache: cache[tuple(path[0:(i+1)])] = x
        return x

    def save(self, weights_file=None, weights_dir=None):

        ### TODO: save optimizers here too
        if weights_file:
            torch.save({
                key: model.state_dict() for key, model in self.edge_map.items() \
                if not isinstance(model, RealityTransfer)
            }, weights_file)

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            for key, model in self.edge_map.items():
                if isinstance(model, RealityTransfer): continue
                if not isinstance(model.model, TrainableModel): continue
                model.model.save(f"{weights_dir}/{model.name}.pth")
            torch.save(self.optimizer, f"{weights_dir}/optimizer.pth")

    # def load_weights(self, weights_file=None, key_filter=None):
    #     for key, state_dict in torch.load(weights_file).items():
    #         if key in self.edge_map and (key_filter is None or key in key_filter):
    #             print('loading', key)
    #             self.edge_map[key].load_state_dict(state_dict)

    def load_weights(self, weights_file=None, key_filter=None, strict=True):
        for key, state_dict in torch.load(weights_file).items():
            if key in self.edge_map and (key_filter is None or key in key_filter):
                print('loading', key)
                self.edge_map[key].load_state_dict(state_dict, strict=strict)

#    def load_weights(self, weights_file=None):
#        for key, state_dict in torch.load(weights_file).items():
#            if key in self.edge_map:
#                self.edge_map[key].load_state_dict(state_dict)

    # def load_weights(self, weights_file=None):
    #     loaded_something = False
    #     for key, state_dict in torch.load(weights_file).items():
    #         if key in self.edge_map:
    #             loaded_something = True
    #             self.edge_map[key].load_model()
    #             self.edge_map[key].load_state_dict(state_dict)
    #     if not loaded_something:
    #         raise RuntimeError(f"No edges loaded from file: {weights_file}")
