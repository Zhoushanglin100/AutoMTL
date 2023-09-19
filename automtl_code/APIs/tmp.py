import numpy as np
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import OrderedDict
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mtl_pytorch.layer_node import Conv2dNode, BN2dNode
from mtl_pytorch.base_node import BasicNode

from mtl_pytorch.trainer import Trainer
from data.heads.pixel2pixel import ASPPHeadNode

from data.dataloader.cityscapes_dataloader import CityScapes
from data.metrics.pixel2pixel_loss import CityScapesCriterions
from data.metrics.pixel2pixel_metrics import CityScapesMetrics

from mobilenetv2 import mobilenet_v2

import platform
# print(platform.platform())

# -----------------------------------

dataroot = '../datasets/cityscapes/'
tasks = ['segment_semantic', 'depth_zbuffer']
task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1}

headsDict = nn.ModuleDict()
trainDataloaderDict = {task: [] for task in tasks}
valDataloaderDict = {}
criterionDict = {}
metricDict = {}

for task in tasks:
    headsDict[task] = ASPPHeadNode(1280, task_cls_num[task])

    # For model trainer
    dataset = CityScapes(dataroot, 'train', task, crop_h=224, crop_w=224)
    trainDataloaderDict[task].append(DataLoader(dataset, 16, shuffle=True))
    dataset1 = CityScapes(dataroot, 'train1', task, crop_h=224, crop_w=224)
    trainDataloaderDict[task].append(DataLoader(dataset1, 16, shuffle=True)) # for network param training
    dataset2 = CityScapes(dataroot, 'train2', task, crop_h=224, crop_w=224)
    trainDataloaderDict[task].append(DataLoader(dataset2, 16, shuffle=True)) # for policy param training

    dataset = CityScapes(dataroot, 'test', task)
    valDataloaderDict[task] = DataLoader(dataset, 8, shuffle=True)

    criterionDict[task] = CityScapesCriterions(task)
    metricDict[task] = CityScapesMetrics(task)


mtlmodel = mobilenet_v2(False, heads_dict=headsDict)
mtlmodel = mtlmodel.cuda()

checkpoint = 'checkpoint/'
trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                  print_iters=100, val_iters=500, save_num=1, policy_update_iters=100)

### pretrain
# trainer.pre_train(iters=1, lr=0.0001, savePath=checkpoint+'Cityscapes/')
trainer.pre_train(iters=10000, lr=0.0001, savePath=checkpoint+'Cityscapes/')

### alter train
loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1, 'policy':0.0005}
trainer.alter_train_with_reg(iters=20000, policy_network_iters=(100,400), 
                             policy_lr=0.01, network_lr=0.0001, 
                             loss_lambda=loss_lambda,
                             savePath=checkpoint+'Cityscapes/')

### sample policy
policy_list = {'segment_semantic': [], 'depth_zbuffer': []}
name_list = {'segment_semantic': [], 'depth_zbuffer': []}

for name, param in mtlmodel.named_parameters():
    if 'policy' in name :
        print(name)
        if 'segment_semantic' in name:
            policy_list['segment_semantic'].append(param.data.cpu().detach().numpy())
            name_list['segment_semantic'].append(name)
        elif 'depth_zbuffer' in name:
            policy_list['depth_zbuffer'].append(param.data.cpu().detach().numpy())
            name_list['depth_zbuffer'].append(name)


# sample_policy_dict = OrderedDict()
# for task in tasks:
#     for name, policy in zip(name_list[task], policy_list[task]):
#         # distribution = softmax(policy, axis=-1
#         distribution = softmax(policy, axis=-1)
#         distribution /= sum(distribution)

#         choice = np.random.choice((0,1,2), p=distribution)
#         if choice == 0:
#             sample_policy_dict[name] = torch.tensor([1.0, 0.0, 0.0])
#         elif choice == 1:
#             sample_policy_dict[name] = torch.tensor([0.0, 1.0, 0.0])
#         elif choice == 2:
#             sample_policy_dict[name] = torch.tensor([0.0, 0.0, 1.0])

shared = 10
sample_policy_dict = OrderedDict()
for task in tasks:
    count = 0
    for name, policy in zip(name_list[task], policy_list[task]):
        if count < shared:
            sample_policy_dict[name] = torch.tensor([1.0,0.0,0.0]).cuda()
        else:
            distribution = softmax(policy, axis=-1)
            distribution /= sum(distribution)
            choice = np.random.choice((0,1,2), p=distribution)
            if choice == 0:
                sample_policy_dict[name] = torch.tensor([1.0,0.0,0.0]).cuda()
            elif choice == 1:
                sample_policy_dict[name] = torch.tensor([0.0,1.0,0.0]).cuda()
            elif choice == 2:
                sample_policy_dict[name] = torch.tensor([0.0,0.0,1.0]).cuda()
        count += 1
        
sample_path = 'checkpoint/Cityscapes/'
sample_state = {'state_dict': sample_policy_dict}
torch.save(sample_state, sample_path + 'sample_policy.model')


### post train
loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1}
trainer.post_train(iters=30000, lr=0.001,
                   decay_lr_freq=4000, decay_lr_rate=0.5,
                   loss_lambda=loss_lambda,
                   savePath=checkpoint+'Cityscapes/',
                   reload='sample_policy.model')


### validation
ckpt = torch.load("checkpoint/Cityscapes/post_train_30000iter.model")
# print(ckpt["state_dict"].keys())
mtlmodel.load_state_dict(ckpt["state_dict"])
trainer.validate('mtl', hard=True) 