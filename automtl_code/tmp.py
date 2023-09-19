from scipy.special import softmax
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from framework.mtl_model import MTLModel
from framework.trainer import Trainer
from data.heads.pixel2pixel import ASPPHeadNode

from data.dataloader.cityscapes_dataloader import CityScapes
from data.metrics.pixel2pixel_loss import CityScapesCriterions
from data.metrics.pixel2pixel_metrics import CityScapesMetrics

# -------------------------------------------

dataroot = 'datasets/cityscapes/'
tasks = ['segment_semantic', 'depth_zbuffer']
task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1}

headsDict = nn.ModuleDict()
trainDataloaderDict = {task: [] for task in tasks}
valDataloaderDict = {}
criterionDict = {}
metricDict = {}

for task in tasks:
    headsDict[task] = ASPPHeadNode(512, task_cls_num[task])

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


prototxt = 'models/deeplab_resnet34_adashare.prototxt'
# prototxt = 'models/mobilenetv2.prototxt' # the input dim of heads should be changed to 1280
mtlmodel = MTLModel(prototxt, headsDict)
mtlmodel = mtlmodel.cuda()

checkpoint = 'checkpoint/'
trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict, 
                  print_iters=100, val_iters=500, save_num=1, policy_update_iters=100)

# print(mtlmodel)
### pre-train
trainer.pre_train(iters=10000, lr=0.0001, savePath=checkpoint+'Cityscapes/')

# ### alter_train
# loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1, 'policy':0.0005}
# trainer.alter_train_with_reg(iters=20000, policy_network_iters=(100,400), policy_lr=0.01, network_lr=0.0001, 
#                              loss_lambda=loss_lambda,
#                              savePath=checkpoint+'Cityscapes/')

# ### sample policy from trained policy distribution and save
# policy_list = {'segment_semantic': [], 'depth_zbuffer': []}
# name_list = {'segment_semantic': [], 'depth_zbuffer': []}

# for name, param in mtlmodel.named_parameters():
#     if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
#         if 'segment_semantic' in name:
#             policy_list['segment_semantic'].append(param.data.cpu().detach().numpy())
#             name_list['segment_semantic'].append(name)
#         elif 'depth_zbuffer' in name:
#             policy_list['depth_zbuffer'].append(param.data.cpu().detach().numpy())
#             name_list['depth_zbuffer'].append(name)

# sample_policy_dict = OrderedDict()
# for task in tasks:
#     for name, policy in zip(name_list[task], policy_list[task]):
#         distribution = softmax(policy, axis=-1)
#         distribution /= sum(distribution)
#         choice = np.random.choice((0,1,2), p=distribution)
#         if choice == 0:
#             sample_policy_dict[name] = torch.tensor([1.0,0.0,0.0]).cuda()
#         elif choice == 1:
#             sample_policy_dict[name] = torch.tensor([0.0,1.0,0.0]).cuda()
#         elif choice == 2:
#             sample_policy_dict[name] = torch.tensor([0.0,0.0,1.0]).cuda()

# sample_state = {'state_dict': sample_policy_dict}
# torch.save(sample_state, checkpoint+'Cityscapes/' + 'sample_policy.model')

# ### post train from scratch
# loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1}
# trainer.post_train(iters=30000, lr=0.001, 
#                    decay_lr_freq=4000, decay_lr_rate=0.5,
#                    loss_lambda=loss_lambda,
#                    savePath=checkpoint+'Cityscapes/', reload='sample_policy.model')

# ### validation
# mtlmodel.load_state_dict(torch.load('CityScapes.model'))
# trainer.validate('mtl', hard=True) 