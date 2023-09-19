import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import random, os
import argparse

from collections import OrderedDict
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import spatial
from graphviz import Digraph

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# from torch.utils.tensorboard import SummaryWriter

# from mtl_pytorch.trainer import Trainer
from mtl_pytorch.trainer_v2 import Trainer
from mtl_pytorch.mobilenetv2 import mobilenet_v2
from mtl_pytorch.head import ClassificationHead
from utils.utils import *
# import slr.admm_v2 as admm
import slr.admm_v3 as admm

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

    ### load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                            download=True, transform=transform_test)
        
    # classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    #              'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    x_train, x_test = trainset.data, testset.data
    y_train, y_test = trainset.targets, testset.targets

    animal_trainset = DatasetMaker([get_class_i(x_train, y_train, [2,3,4,5,6,7]),
                                    get_class_i(x_train, y_train, [0,1,8,9])], 
                                   transform_with_aug)
    animal_testset = DatasetMaker([get_class_i(x_test, y_test, [2,3,4,5,6,7]),
                                   get_class_i(x_test, y_test, [0,1,8,9])], 
                                  transform_no_aug)

    tasks = ['multiclass', 'binary']
    task_cls_num = {'multiclass': 10, 'binary': 2}
    train_dataset = {"multiclass": trainset, 
                     "binary": animal_trainset}
    val_dataset = {"multiclass": testset, 
                   "binary": animal_testset}
    criterion = nn.CrossEntropyLoss()

    headsDict = nn.ModuleDict()
    trainDataloaderDict = {task: [] for task in tasks}
    valDataloaderDict = {}
    criterionDict = {}
    metricDict = {}

    for task in tasks:
        headsDict[task] = ClassificationHead(task_cls_num[task])

        trainDataloaderDict[task] = DataLoader(train_dataset[task], 
                                                batch_size=args.batch_size, 
                                                shuffle=True, **kwargs)
        valDataloaderDict[task] = DataLoader(val_dataset[task], batch_size=100, 
                                                shuffle=False, **kwargs)
        criterionDict[task] = criterion
        metricDict[task] = []
    
    ### Define MTL model
    mtlmodel = mobilenet_v2(False, heads_dict=headsDict)
    mtlmodel = mtlmodel.to(device)

    # for i, (name, W) in enumerate(mtlmodel.named_parameters()):
    #     if ("bias" not in name) and ("binary" in name) and ("taskOp" in name) and \
    #     W.requires_grad and (len(W.shape) >= 2):
    #         print(name)
    # print("=============")
    # for i, (name, W) in enumerate(mtlmodel.named_parameters()):
    #     if ("bias" not in name) and ("multiclass" in name) and ("taskOp" in name) and \
    #     W.requires_grad and (len(W.shape) >= 2):            
    #         print(name)
    # # ----------------------------
    # for i, (name, W) in enumerate(mtlmodel.named_parameters()):
    #     if ("bias" not in name) and \
    #         W.requires_grad and (len(W.shape) >= 2):
    #         print(name)
    # sys.exit(0)
    # # ----------------------------
    # for i, (name, W) in enumerate(mtlmodel.named_parameters()):
    #     if (("multiclass" in name) or ("binary" in name)) and ("policy" not in name): # and (len(W.shape) >= 2):            
    #         print(name)
    # sys.exit(0)

    ### Define training framework
    trainer = Trainer(mtlmodel, 
                      trainDataloaderDict, valDataloaderDict, 
                      criterionDict, metricDict, 
                      print_iters=10, val_iters=100, 
                      save_iters=100, save_num=1, 
                      policy_update_iters=100)

    # ==================================
    if args.evaluate:
        ckpt = torch.load(args.evaluate)
        mtlmodel.load_state_dict(ckpt['state_dict'], strict=False)
        sd = mtlmodel.state_dict()

        # for name, W in sd.items():
        #     if ("policy" in name) or ((tasks[0] not in name) and (tasks[1] not in name)):
        #         continue
        #     print(name)
        # sys.exit(0)
        # mtlmodel.load_state_dict(sd)

        policy_params = ckpt['state_dict'].copy()
        for key in ckpt['state_dict']:
            if 'policy' not in key:
                del policy_params[key]
        # print(policy_params)

        # for k, v in policy_params.items():
        #     print(k, v)
        # sys.exit(0)

        trainer.validate('mtl', hard=True)

        # admm.hard_prune(args, tasks, mtlmodel, 
        #                 f"profile/{args.config_file}.yml", 
        #                 policy_params)
        # trainer.validate('mtl', hard=True)
        admm.test_overall_sparsity(mtlmodel, policy_params)
        admm.test_specific_sparsity(tasks, mtlmodel, f"profile/{args.config_file}.yml")
        admm.test_specific_policy_sparsity(tasks, mtlmodel, policy_params)
        # admm.test_sparsity(args, tasks, mtlmodel, policy_params)

        # ---------------------------
        ## policy visualization
        if args.visualize:
            
            name = args.evaluate.split("/")[-1].split(".")[0]
            vis_savepath = f"result/{name}"
            if not os.path.exists(vis_savepath):
                os.makedirs(vis_savepath)
            print(f"All visualization save to {vis_savepath}")

            policy_list = {"multiclass": [], "binary": []}
            for name, param in mtlmodel.named_parameters():
                if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
                    policy = param.data.cpu().detach().numpy()
                    distribution = softmax(policy, axis=-1)
                    if 'multiclass' in name:
                        policy_list['multiclass'].append(distribution)
                    elif 'binary' in name:
                        policy_list['binary'].append(distribution)
            print(policy_list)

            spectrum_list = []
            ylabels = {'multiclass': 'multi-class',
                       'binary': "binary"} 
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
            policy_list = {"multiclass": [], "binary": []}
            for name, param in mtlmodel.named_parameters():
                if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
                    policy = param.data.cpu().detach().numpy()
                    distribution = softmax(policy, axis=-1)
                    if 'multiclass' in name:
                        policy_list['multiclass'].append(distribution)
                    elif 'binary' in name:
                        policy_list['binary'].append(distribution)
            policy_array = np.array([np.array(policy_list['multiclass']).ravel(), 
                                     np.array(policy_list['binary']).ravel()])
            sim = np.zeros((4,4))
            for i in range(len(tasks)):
                for j in range(len(tasks)):
                    sim[i,j] = 1 - spatial.distance.cosine(policy_array[i,:], policy_array[j,:])

            # mpl.rc('image', cmap='Blues')
            # tickSize = 15
            # plt.figure(figsize=(10,10))
            # plt.xticks(fontsize=tickSize, rotation='vertical')
            # plt.yticks(fontsize=tickSize)
            # ax = plt.subplot()
            # im = ax.imshow(sim)
            # ax.set_xticks(np.arange(2))
            # ax.set_yticks(np.arange(2))
            # ax.set_xticklabels(['multiclass', 'binary'])
            # ax.set_yticklabels(['multiclass', 'binary'])
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="4%", pad=0.05)
            # cb = plt.colorbar(im, cax=cax,ticks=[1,0.61])
            # cb.ax.set_yticklabels(['high', 'low']) 
            # cb.ax.tick_params(labelsize=tickSize)
            # plt.savefig(f"{vis_savepath}/task_cor")
            # plt.close()

            ### Show Policy (for test)
            dot = Digraph(comment='Policy')
            # make nodes
            layer_num = len(policy_list['multiclass'])
            for i in range(layer_num):
                with dot.subgraph(name='cluster_L'+str(i),node_attr={'rank':'same'}) as c:
                    c.attr(rankdir='LR')
                    c.node('L'+str(i)+'B0', 'Shared')
                    c.node('L'+str(i)+'B1', 'Specific')
                    c.node('L'+str(i)+'B2', 'Skip')
            
            # make edges
            colors = {'multiclass': 'blue', 'binary': 'red'}
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
        loss_lambda = {'multiclass': 1, 'binary': 1, 'policy': 0.0005}
        trainer.alter_train_with_reg(iters=args.alter_iters, 
                                     policy_network_iters=(50,200), 
                                     policy_lr=0.01, network_lr=0.001,
                                     loss_lambda=loss_lambda,
                                     savePath=savepath, writerPath=savepath,
                                     reload=args.pretrain_model)

    # ----------------
    if args.post_train:
        ### Step 3: sample policy from trained policy distribution and save
        print(">>>>>>>> Sample Policy <<<<<<<<<<")
        policy_list = {'multiclass': [], 'binary': []}
        name_list = {'multiclass': [], 'binary': []}

        if args.alter_model != None:
            print("!! Load alter-train model")
            state = torch.load(savepath + args.alter_model)
            mtlmodel.load_state_dict(state['state_dict'])

        for name, param in mtlmodel.named_parameters():
            if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
                print(name)
                if 'multiclass' in name:
                    policy_list['multiclass'].append(param.data.cpu().detach().numpy())
                    name_list['multiclass'].append(name)
                elif 'binary' in name:
                    policy_list['binary'].append(param.data.cpu().detach().numpy())
                    name_list['binary'].append(name)

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
        torch.save(sample_state, sample_path + f'sample_policy{args.ext}.model')

        # ----------------
        ### Step 4: post train from scratch
        print(">>>>>>>> Post-train <<<<<<<<<<")
        loss_lambda = {'multiclass': 2, 'binary': 1}
        print(loss_lambda)
        trainer.post_train(iters=args.post_iters, lr=args.post_lr,
                            decay_lr_freq=args.decay_lr_freq, decay_lr_rate=0.5,
                            loss_lambda=loss_lambda,
                            savePath=savepath, writerPath=savepath,
                            reload=f'sample_policy{args.ext}.model',
                            ext=args.ext)

    # ----------------
    if args.slr_prune:
        args.has_wandb = has_wandb

        ### Step 5: prune after post train
        savepath_prune = savepath+"/prune/"
        if not os.path.exists(savepath_prune):
            os.makedirs(savepath_prune)

        trainer.post_train_prune(args, 
                                 lr=args.prune_lr,
                                 decay_lr_freq=args.decay_lr_freq, decay_lr_rate=0.5,
                                 savePath=savepath_prune, writerPath=savepath_prune,
                                 ext=args.ext)

# --------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### DONN parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256, 
                        help="cifar10: train: 50k; test: 10k")

    parser.add_argument('--save-dir', type=str, default='multi', help="save the model")
    parser.add_argument('--evaluate', type=str, help="Model path for evalation")
    parser.add_argument('--visualize', action='store_true', default=False, 
                        help="visualize result when evalation")

    parser.add_argument('--pretrain', action='store_true', default=False, 
                        help='whether to run pre-train part')
    parser.add_argument('--pretrain-iters', type=int, default=8000, 
                        help='#iterations for pre-training, default: [8k: bz=128]')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='pre-train learning rate')

    parser.add_argument('--alter-train', action='store_true', default=False, 
                        help='whether to run alter-trian part')
    parser.add_argument('--pretrain-model', type=str, default=None, 
                        help="pretrain model in alter-train")
    parser.add_argument('--alter-iters', type=int, default=8000, 
                        help='#iterations for alter-train, default: 20000')

    parser.add_argument('--post-train', action='store_true', default=False, 
                        help='whether to run post-train part')
    parser.add_argument('--alter-model', type=str, default=None, 
                        help="alter-train model in post-train")
    parser.add_argument('--post-iters', type=int, default=30000, 
                        help='#iterations for post-train, default: 30000')
    parser.add_argument('--post-lr', type=float, default=0.01, 
                        help='post-train learning rate')
    parser.add_argument('--decay-lr-freq', type=float, default=2000, 
                        help='post-train learning rate decay frequency')
    parser.add_argument('--shared', type=float, default=5, 
                        help='number of layers force to share during sample policy')

    parser.add_argument('--slr-prune', action='store_true', default=False, 
                        help='whether to run prune the model after post-train')

    # -------------------- SLR Parameter ---------------------------------

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