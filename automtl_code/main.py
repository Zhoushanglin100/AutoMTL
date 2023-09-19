import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import random, os, sys
import argparse

from collections import OrderedDict
from scipy.special import softmax

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import spatial
from graphviz import Digraph
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from data.heads.pixel2pixel import ASPPHeadNode

from data.dataloader.cityscapes_dataloader import CityScapes
from data.metrics.pixel2pixel_loss import CityScapesCriterions
from data.metrics.pixel2pixel_metrics import CityScapesMetrics

from data.dataloader.nyuv2_dataloader import NYU_v2
from data.metrics.pixel2pixel_loss import NYUCriterions
from data.metrics.pixel2pixel_metrics import NYUMetrics

from framework.mtl_model import MTLModel

from APIs.mobilenetv2 import mobilenet_v2
# from APIs.mtl_pytorch.trainer import Trainer
from APIs.mtl_pytorch.trainer_v2 import Trainer
import APIs.mtl_pytorch.slr.admm_v4 as admm
import APIs.mtl_pytorch.slr.test_sparsity as sparsity

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

# -----------------------------------

def main(args):

    print("---------------")
    print(args)
    print("---------------")

    torch.backends.cudnn.benchmark = True
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': False}

    if args.arch == "mbnet":
        input_dim = 1280
    elif args.arch == "adashare":
        input_dim = 512
    else:
        raise AttributeError("Model not defined")
    
    ### load data and head
    if args.data == "cityscape":
        dataroot = '/data/AutoMTL_data/cityscapes/'
        tasks = ['segment_semantic', 'depth_zbuffer']
        task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1}

        headsDict = nn.ModuleDict()
        trainDataloaderDict = {task: [] for task in tasks}
        valDataloaderDict = {}
        criterionDict = {}
        metricDict = {}

        for task in tasks:
            headsDict[task] = ASPPHeadNode(input_dim, task_cls_num[task])

            dataset = CityScapes(dataroot, 'train', task, crop_h=224, crop_w=224)   # for model trainer
            trainDataloaderDict[task].append(DataLoader(dataset, 16, shuffle=True))
            dataset1 = CityScapes(dataroot, 'train1', task, crop_h=224, crop_w=224)
            trainDataloaderDict[task].append(DataLoader(dataset1, 16, shuffle=True)) # for network param training
            dataset2 = CityScapes(dataroot, 'train2', task, crop_h=224, crop_w=224)
            trainDataloaderDict[task].append(DataLoader(dataset2, 16, shuffle=True)) # for policy param training

            dataset = CityScapes(dataroot, 'test', task)
            valDataloaderDict[task] = DataLoader(dataset, 8, shuffle=True)
            criterionDict[task] = CityScapesCriterions(task)
            metricDict[task] = CityScapesMetrics(task)

    elif args.data == "nyu":
        dataroot = '/data/AutoMTL_data/NYUv2/'
        tasks = ['segment_semantic', 'normal', 'depth_zbuffer']
        task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

        headsDict = nn.ModuleDict()
        trainDataloaderDict = {task: [] for task in tasks}
        valDataloaderDict = {}
        criterionDict = {}
        metricDict = {}

        for task in tasks:
            headsDict[task] = ASPPHeadNode(input_dim, task_cls_num[task])

            dataset = NYU_v2(dataroot, 'train', task, crop_h=321, crop_w=321)       # for model trainer
            trainDataloaderDict[task].append(DataLoader(dataset, 16, shuffle=True))
            dataset1 = NYU_v2(dataroot, 'train1', task, crop_h=321, crop_w=321)
            trainDataloaderDict[task].append(DataLoader(dataset1, 16, shuffle=True)) # for network param training
            dataset2 = NYU_v2(dataroot, 'train2', task, crop_h=321, crop_w=321) 
            trainDataloaderDict[task].append(DataLoader(dataset2, 16, shuffle=True)) # for policy param training

            dataset = NYU_v2(dataroot, 'test', task, crop_h=321, crop_w=321)
            valDataloaderDict[task] = DataLoader(dataset, 8, shuffle=True)
            criterionDict[task] = NYUCriterions(task)
            metricDict[task] = NYUMetrics(task)
    
    ### Define MTL model
    if args.arch == "mbnet":
        mtlmodel = mobilenet_v2(False, heads_dict = headsDict)
    elif args.arch == "adashare":
        prototxt = 'models/deeplab_resnet34_adashare.prototxt'
        # mtlmodel = MTLModel(prototxt, headsDict)
        mtlmodel = MTLModel(prototxt, headsDict, BNsp=True)
    else:
        raise AttributeError("Model not defined")
    mtlmodel = mtlmodel.to(device)

    # # ----------------
    # ### Get model structure
    # for i, (name, W) in enumerate(mtlmodel.named_parameters()):
    #     # if ("policy" not in name) and (len(W.shape) > 1) and \
    #     #     any(task in name for task in tasks):
    #     #     print(name)

    #     if ("policy" not in name) and (len(W.shape) > 1):
    #         print(name)

    # sys.exit(0)
    # # ----------------

    ### Define training framework
    trainer = Trainer(mtlmodel, 
                      trainDataloaderDict, valDataloaderDict, 
                      criterionDict, metricDict, 
                      print_iters=100, val_iters=500, 
                      save_num=1, 
                      policy_update_iters=100)

    # ==================================
    if args.evaluate:
        if "config" in args.evaluate:
            ckpt = torch.load(args.evaluate)["state_dict"]
        else:
            ckpt = torch.load(args.evaluate)

        mtlmodel.load_state_dict(ckpt, strict=False)
        mtlmodel.to(device)

        policy_params = ckpt.copy()
        for key, value in ckpt.items():
            if ('policy' not in key):
                del policy_params[key]
            else:
                if torch.eq(value, torch.tensor([0., 0., 0.]).cuda()).all():
                    del policy_params[key]
        # print(policy_params)
        # for k in policy_params:
        #     print(k)

        sd = mtlmodel.state_dict()
        # sd["net.25.basicOp.weight"] = sd["net.25.taskOp.depth_zbuffer.weight"]
        # sd["net.25.taskOp.segment_semantic.weight"] =torch.zeros_like(sd["net.25.taskOp.segment_semantic.weight"], device=device)
        # sd["net.25.taskOp.normal.weight"] = torch.zeros_like(sd["net.25.taskOp.depth_zbuffer.weight"], device=device)
        # sd["net.25.taskOp.depth_zbuffer.weight"] = torch.zeros_like(sd["net.25.taskOp.depth_zbuffer.weight"], device=device)
        # mtlmodel.load_state_dict(sd, strict=False)

        # # -----------------
        # for name, W in mtlmodel.named_parameters():
        #     print(name, W.shape)
        # for k, v in sd.items():
        #     print(k)
        # for k, v in policy_params.items():
        #     print(k, v)
        # sys.exit(0)
        # # -----------------

        trainer.validate('mtl', hard=True)

        # # ----
        # ### for slr-trained model
        # admm.hard_prune(args, tasks, mtlmodel, args.config_file)
        # trainer.validate('mtl', hard=True)
        # # ----

        sparsity.test_sparsity(args, mtlmodel)
        # sparsity.test_overall_sparsity(mtlmodel, policy_params)
        # sparsity.test_specific_sparsity(tasks, mtlmodel)
        sparsity.test_specific_div_sparsity(tasks, mtlmodel)
        sparsity.test_specific_policy_sparsity(tasks, mtlmodel, policy_params)


        # ---------------------------
        ## policy visualization
        if args.visualize:
            
            name = args.evaluate.split("/")[-1].split(".")[0]
            vis_savepath = f"result/{name}"
            if not os.path.exists(vis_savepath):
                os.makedirs(vis_savepath)
            print(f"All visualization save to {vis_savepath}")
            policy_list = {}
            for task in tasks:
                policy_list[task] = []
            # policy_list = {"segment_semantic": [], "depth_zbuffer": []}
            for name, param in mtlmodel.named_parameters():
                if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
                    policy = param.data.cpu().detach().numpy()
                    distribution = softmax(policy, axis=-1)
                    for task in tasks:
                        if task in name:
                            policy_list[task].append(distribution)
            print(policy_list)

            spectrum_list = []
            ylabels = {}
            for task in tasks:
                ylabels[task] = task
            # ylabels = {'segment_semantic': 'segment_semantic',
            #            'depth_zbuffer': "depth_zbuffer"} 
            tickSize = 15
            labelSize = 16
            for task in tasks:
                policies = policy_list[task]    
                spectrum = np.stack([policy for policy in policies])
                spectrum = np.repeat(spectrum[np.newaxis,:,:],1,axis=0)
                spectrum_list.append(spectrum)
                
                plt.figure(figsize=(10,5))
                plt.xlabel('Layer No.', fontsize=labelSize)
                plt.xticks(fontsize=tickSize)
                plt.ylabel(ylabels[task], fontsize=labelSize)
                plt.yticks(fontsize=tickSize)
                
                ax = plt.subplot()
                im = ax.imshow(spectrum.T)
                ax.set_yticks(np.arange(3))
                ax.set_yticklabels(['shared', 'specific', 'skip'])

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.05)

                cb = plt.colorbar(im, cax=cax)
                cb.ax.tick_params(labelsize=tickSize)
                plt.savefig(f"{vis_savepath}/spect_{task}.png")
                plt.close()

            ### plot task correlation
            for name, param in mtlmodel.named_parameters():
                if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
                    policy = param.data.cpu().detach().numpy()
                    distribution = softmax(policy, axis=-1)
                    for task in tasks:
                        if task in name:
                            policy_list[task].append(distribution)
            tmp_array = []
            for task in tasks:
                tmp_array.append(np.array(policy_list[task]).ravel())
            policy_array = np.array(tmp_array)
            sim = np.zeros((len(tasks),len(tasks)))
            for i in range(len(tasks)):
                for j in range(len(tasks)):
                    sim[i,j] = 1 - spatial.distance.cosine(policy_array[i,:], policy_array[j,:])

            mpl.rc('image', cmap='Blues')
            tickSize = 15
            plt.figure(figsize=(10,10))
            plt.xticks(fontsize=tickSize, rotation='vertical')
            plt.yticks(fontsize=tickSize)
            ax = plt.subplot()
            im = ax.imshow(sim)
            ax.set_xticks(np.arange(len(tasks)))
            ax.set_yticks(np.arange(len(tasks)))
            ax.set_xticklabels(tasks)
            ax.set_yticklabels(tasks)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            cb = plt.colorbar(im, cax=cax,ticks=[1,0.61])
            cb.ax.set_yticklabels(['high', 'low']) 
            cb.ax.tick_params(labelsize=tickSize)
            plt.savefig(f"{vis_savepath}/task_cor")
            plt.close()

            ### Show Policy (for test)
            dot = Digraph(comment='Policy')
            # make nodes
            layer_num = len(policy_list[tasks[0]])
            for i in range(layer_num):
                with dot.subgraph(name='cluster_L'+str(i),node_attr={'rank':'same'}) as c:
                    c.attr(rankdir='LR')
                    c.node('L'+str(i)+'B0', 'Shared')
                    c.node('L'+str(i)+'B1', 'Specific')
                    c.node('L'+str(i)+'B2', 'Skip')
            
            # make edges
            colors = {}
            color_list = list(mcolors.TABLEAU_COLORS.values())
            for i, task in enumerate(tasks):
                colors[task] = color_list[i]
            # colors = {'segment_semantic': 'blue', 'depth_zbuffer': 'red'}
            for task in tasks:
                for i in range(layer_num-1):
                    prev = np.argmax(policy_list[task][i])
                    nxt = np.argmax(policy_list[task][i+1])
                    dot.edge('L'+str(i)+'B'+str(prev), 'L'+str(i+1)+'B'+str(nxt), color=colors[task])
            # dot.render('Best.gv', view=True)  
            dot.render(f'{vis_savepath}/Best', view=False)  

        return
    # ==================================

    ### Train
    checkpoint = 'checkpoint/'
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    savepath = checkpoint+args.save_dir+"/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    print(f"All ckpts save to {savepath}")

    # ----------------
    ### Step 1: pre-train
    if args.pretrain:
        print(">>>>>>>> pre-train <<<<<<<<<<")
        trainer.pre_train(iters=args.pretrain_iters, lr=args.lr, 
                          savePath=savepath, writerPath=savepath)

    # ----------------
    ### Step 2: alter-train
    if args.alter_train:
        print(">>>>>>>> alter-train <<<<<<<<<<")
        if args.data == "cityscape":
            loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1, 'policy':0.0005}
        elif args.data == "nyu":
            loss_lambda = {'segment_semantic': 3, 'normal':20, 'depth_zbuffer': 3, 'policy':0.003}
        trainer.alter_train_with_reg(iters=args.alter_iters, 
                                     policy_network_iters=(100,400), 
                                     policy_lr=0.01, network_lr=0.0001, 
                                     loss_lambda=loss_lambda,
                                     savePath=savepath, writerPath=savepath,
                                     reload=args.pretrain_model)

    # ----------------
    if args.post_train:
        ### Step 3: sample policy from trained policy distribution and save
        if not args.reload_policy:
            print(">>>>>>>> Sample Policy <<<<<<<<<<")
            # policy_list = {'segment_semantic': [], 'depth_zbuffer': []}
            # name_list = {'segment_semantic': [], 'depth_zbuffer': []}

            policy_list = {}
            for task in tasks:
                policy_list[task] = []
            name_list = policy_list.copy()

            if args.alter_model != None:
                print("!! Load alter-train model")
                state = torch.load(savepath + args.alter_model)
                mtlmodel.load_state_dict(state['state_dict'])
            else:
                print("Random Generate Policy!!")

            for name, param in mtlmodel.named_parameters():
                if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
                    for task in tasks:
                        if task in name:
                            policy_list[task].append(param.data.cpu().detach().numpy())
                            name_list[task].append(name)

            shared = args.shared
            sample_policy_dict = OrderedDict()
            for task in tasks:
                count = 0
                for name, policy in zip(name_list[task], policy_list[task]):
                    if count < shared:
                        sample_policy_dict[name] = torch.tensor([1.0, 0.0, 0.0]).cuda()
                    else:
                        distribution = softmax(policy, axis=-1)
                        distribution /= sum(distribution)
                        choice = np.random.choice((0, 1, 2), p=distribution)
                        if choice == 0:
                            sample_policy_dict[name] = torch.tensor([1.0, 0.0, 0.0]).cuda()
                        elif choice == 1:
                            sample_policy_dict[name] = torch.tensor([0.0, 1.0, 0.0]).cuda()
                        elif choice == 2:
                            sample_policy_dict[name] = torch.tensor([0.0, 0.0, 1.0]).cuda()
                    count += 1

            sample_path = savepath
            sample_state = {'state_dict': sample_policy_dict}
            reload_policy = f'sample_policy{args.ext}.model'
            torch.save(sample_state, sample_path + reload_policy)
        else:
            print("Load Existing Policy.")
            reload_policy = args.reload_policy
        
        print("Policy file:", reload_policy)

        # ----------------
        ### Step 4: post train from scratch
        print(">>>>>>>> Post-train <<<<<<<<<<")
        if args.data == "cityscape":
            loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1}
        elif args.data == "nyu":
            loss_lambda = {'segment_semantic': 1, 'normal':1, 'depth_zbuffer': 1}
        trainer.post_train(iters=args.post_iters, lr=args.post_lr,
                            decay_lr_freq=args.decay_lr_freq, decay_lr_rate=0.5,
                            loss_lambda=loss_lambda,
                            savePath=savepath, writerPath=savepath,
                            reload=reload_policy, ext=args.ext)

    # ----------------
    if args.slr_prune:
        ### Step 5: prune after post train
        print(">>>>>>>> Post-train with Pruning <<<<<<<<<<")
        args.has_wandb = has_wandb
        savepath_prune = savepath+f"/prune_{args.data}/"
        if not os.path.exists(savepath_prune):
            os.makedirs(savepath_prune)

        if args.data == "cityscape":
            loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1}
        elif args.data == "nyu":
            loss_lambda = {'segment_semantic': 1, 'normal':1, 'depth_zbuffer': 1}
        print(loss_lambda)

        trainer.post_train_prune(args, loss_lambda=loss_lambda,
                                 lr=args.prune_lr, 
                                 decay_lr_freq=args.decay_lr_freq, decay_lr_rate=0.5,
                                 savePath=savepath_prune, writerPath=savepath_prune)

# --------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # -------------------- Base ---------------------------------
    parser.add_argument('--arch', type=str, default='mbnet', 
                        help="model architecture")
    parser.add_argument('--data', type=str, default='cityscape', 
                        help="dataset")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256, 
                        help="cifar10: train: 50k; test: 10k")

    parser.add_argument('--save-dir', type=str, default='multi', 
                        help="save the model")
    parser.add_argument('--evaluate', type=str, 
                        help="Model path for evalation")
    parser.add_argument('--visualize', action='store_true', default=False, 
                        help="visualize result when evalation")

    parser.add_argument('--pretrain', action='store_true', default=False, 
                        help='whether to run pre-train part')
    parser.add_argument('--pretrain-iters', type=int, default=10000, 
                        help='#iterations for pre-training, default: [8k: bz=128]')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='pre-train learning rate')

    parser.add_argument('--alter-train', action='store_true', default=False, 
                        help='whether to run alter-trian part')
    parser.add_argument('--pretrain-model', type=str, default=None, 
                        help="pretrain model in alter-train")
    parser.add_argument('--alter-iters', type=int, default=20000, 
                        help='#iterations for alter-train, default: 20000')

    parser.add_argument('--post-train', action='store_true', default=False, 
                        help='whether to run post-train part')
    parser.add_argument('--reload-policy', type=str, default=None, 
                        help='load existing policy')
    parser.add_argument('--alter-model', type=str, default=None, 
                        help="alter-train model in post-train")
    parser.add_argument('--post-iters', type=int, default=30000, 
                        help='#iterations for post-train, default: 30000')
    parser.add_argument('--post-lr', type=float, default=0.001, 
                        help='post-train learning rate')
    parser.add_argument('--decay-lr-freq', type=float, default=4000, 
                        help='post-train learning rate decay frequency')
    parser.add_argument('--shared', type=float, default=0, 
                        help='number of layers force to share during sample policy')

    # -------------------- SLR Parameter ---------------------------------
    parser.add_argument('--slr-prune', action='store_true', default=False, 
                        help='whether to run prune the model after post-train')

    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--max-step', type=int, default=6000, metavar='N',
                        help='number of max step to train (default: 6000)')
    parser.add_argument('--prune-lr', type=float, default=0.001, 
                        help='post-train learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='default',
                        help="[default, cosine, step]")
    
    parser.add_argument('--admm-train', action='store_true', default=False,
                        help='Choose admm quantization training')
    parser.add_argument('--masked-retrain', action='store_true', default=False,
                        help='whether to masked training for admm quantization')
    parser.add_argument('--optimization', type=str, default='savlr',
                        help='optimization type: [savlr, admm]')
    parser.add_argument('--admm-epoch', type=int, default=1, metavar='N',
                        help='number of interval epochs to update admm (default: 10)')
    parser.add_argument('--retrain-epochs', type=int, default=20, metavar='N',
                        help='for retraining')
    parser.add_argument('--combine-progressive', action='store_true', default=False,
                        help='for filter pruning after column pruning')
    parser.add_argument('--baseline', type=str, 
                        help='Load baseline model for pruning')
    parser.add_argument('--slr-model', type=str, 
                        help='Load slr-trained model for retrain')
    
    parser.add_argument('--M', type=int, default=300, metavar='N',
                        help='SLR parameter M ')
    parser.add_argument('--r', type=float, default=0.1, metavar='N',
                        help='SLR parameter r ')
    parser.add_argument('--initial-s', type=float, default=0.01, metavar='N',
                        help='SLR parameter initial stepsize')
    parser.add_argument('--rho', type=float, default=0.1, 
                        help="define rho for ADMM")
    parser.add_argument('--rho-num', type=int, default=1, 
                        help="define how many rohs for ADMM training")
    
    parser.add_argument('--config-file', type=str, default='config_0.5', 
                        help="prune config file")
    parser.add_argument('--sparsity-type', type=str, default='irregular',
                        help='sparsity type: [irregular,column,channel,filter,pattern,random-pattern]')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 10)')

    # -------------------- Others ---------------------------------
    parser.add_argument('--enable-wandb', action='store_true', default=False,
                        help='whether to use wandb to log')
    parser.add_argument('--ext', type=str, default='', 
                        help="extension for save the model")

    args_ = parser.parse_args()

    main(args_)