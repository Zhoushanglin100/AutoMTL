import numpy as np
import os, sys, pickle

import torch
from torch.utils.tensorboard import SummaryWriter
import slr.admm as admm
import wandb

class Trainer():
    def __init__(self, model, 
                 train_dataloader_dict, val_dataloader_dict, 
                 criterion_dict, metric_dict, 
                 print_iters=50, val_iters=200, 
                 save_iters=200, save_num=5, 
                 policy_update_iters=100):
        
        super(Trainer, self).__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataloader_dict = train_dataloader_dict
        self.val_dataloader_dict = val_dataloader_dict
        self.criterion_dict = criterion_dict
        self.metric_dict = metric_dict
        
        self.tasks = list(self.train_dataloader_dict.keys())
        self.train_iter_dict = {}
        self.train_network_iter_dict = {}
        self.train_policy_iter_dict = {}
        self.loss_list = {}
        self.mixed_loss_list = {}

        self.set_train_loss_data_iter()
        
        self.print_iters = print_iters
        self.val_iters = val_iters
        self.save_iters = save_iters
        self.save_num = save_num
        self.policy_update_iters = policy_update_iters

    def pre_train(self, iters, lr=0.001, task_iters=None, writerPath=None, savePath=None, reload=None):
        self.model.train()
        if writerPath != None:
            writer = SummaryWriter(log_dir=writerPath+'pre_train_all/')
        else:
            writer = None
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        
        start = 0
        if reload is not None and savePath is not None:
            state = torch.load(savePath + reload)
            self.model.load_state_dict(state['state_dict'])
            # optimizer.load_state_dict(state['optimizer'])
            start = state['iter'] + 1
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)

        for i in range(start, iters):
            ### Pre-train all weights
            if task_iters is None:
                self.train_step('pre_train_all', optimizer)
            else:
                task_idx = self.which_task(i, task_iters)
                self.train_step_task('pre_train_all', self.tasks[task_idx], optimizer)

            if (i+1) % self.print_iters == 0:
                self.print_train_loss(i, writer)
                self.reset_train_loss()
            if (i+1) % self.val_iters == 0:
                self.validate('pre_train_all', i, writer=writer)
                self.model.train()
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    state = {'iter': i,
                            'state_dict': self.model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                    self.save_model(state, 'pre_train_all', savePath)
         
        # Reset loss list and the data iters
        self.set_train_loss_data_iter()
        return
    
    def alter_train(self, iters, policy_network_iters=(100,400), policy_lr=0.01, network_lr=0.001, tau=5, writerPath=None, savePath=None, reload=None):
        self.model.train()
        # Key point: set two optimizers, one for the model, one for the policy
        if writerPath != None:
            writer = SummaryWriter(log_dir=writerPath+'alter_train/')
        else:
            writer = None
        
        # Step 1: Get the two optimizers for network and policy respectively
        self.freeze_policy()
        policy_op = torch.optim.Adam(filter(lambda p: p.requires_grad==False, self.model.parameters()), lr=policy_lr, weight_decay=5*1e-4)
        network_op = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=network_lr, momentum=0.9, weight_decay=1e-4)
        self.unfreeze_all_weights()
        start = 0
        if reload is not None and savePath is not None:
            state = torch.load(savePath + reload)
            self.model.load_state_dict(state['state_dict'])
            if 'alter_train' in reload:
                policy_op.load_state_dict(state['policy_op'])
                network_op.load_state_dict(state['network_op'])
                tau = state['tau']
                start = state['iter'] + 1       
        
        # Step 2: Train network and policy alternatively
        policy_count = 0
        for i in range(start, iters):
            # Step 2-1: Train policy when the current iter is in the first part of policy_network_iters
            if i % (policy_network_iters[0] + policy_network_iters[1]) in range(policy_network_iters[0]):
                self.train_step('mtl', policy_op, alter_train='policy', tau=tau)
                policy_count += 1
            # Step 2-2: Train network when the current iter is in the second part of policy_network_iters
            else:
                self.train_step('mtl', network_op,alter_train='network', tau=tau)

            # Step 3: Update tau in policy every self.policy_update_iters
            if policy_count > self.policy_update_iters and tau > 1e-6:
                tau = tau * 0.965
                print('tau: ' + str(tau))
                policy_count = 0

            # Step 4: Print loss
            if (i+1) % self.print_iters == 0:
                self.print_train_loss(i, writer)
                self.reset_train_loss()
            
            # Step 5: Validation
            if (i+1) % self.val_iters == 0:
                self.validate('mtl', i, tau=tau, writer=writer)
                self.model.train()
                
            # Step 6: Save model
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    state = {'iter': i,
                            'state_dict': self.model.state_dict(),
                            'policy_op': policy_op.state_dict(),
                            'network_op': network_op.state_dict(),
                            'tau': tau}
                    self.save_model(state, 'alter_train', savePath)
                    
            ################ delete if not needed ############
            if (i+1) % 10000 == 0:
                for g in policy_op.param_groups:
                    g['lr'] *= 0.3
                    print('lr changed')
            #################################################
                    
        # Reset loss list and the data iters
        self.set_train_loss_data_iter()
        return
    
    def alter_train_with_reg(self, iters, policy_network_iters=(100,400), policy_lr=0.01, network_lr=0.0001, 
                             tau=5, tau_ratio=0.965,
                             policy_scale=1, loss_lambda=1.0,
                             share_num=0,
                             writerPath=None, savePath=None, reload=None):
        self.model.train()
        # Key point: set two optimizers, one for the model, one for the policy
        if writerPath != None:
            writer = SummaryWriter(log_dir=writerPath+'alter_train_with_reg_bottom_' + str(share_num)+'/')
        else:
            writer = None
        
        # Step 1: Get the two optimizers for network and policy respectively
        self.freeze_policy()
        policy_op = torch.optim.Adam(filter(lambda p: p.requires_grad==False, self.model.parameters()), lr=policy_lr, weight_decay=5*1e-4)
        network_op = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=network_lr, momentum=0.9, weight_decay=1e-4)
        self.unfreeze_all_weights()
        start = 0
        if reload is not None and savePath is not None:
            state = torch.load(savePath + reload)
            self.model.load_state_dict(state['state_dict'])
            if 'alter_train_with_reg' in reload:
                policy_op.load_state_dict(state['policy_op'])
                network_op.load_state_dict(state['network_op'])
                tau = state['tau']
                start = state['iter'] + 1       
        
        # Share and freeze some bottom layers 
        self.freeze_and_share_bottom_policy(share_num)
        
        # Step 2: Train network and policy alternatively
        policy_count = 0
        for i in range(start, iters):
            # Step 2-1: Train policy when the current iter is in the first part of policy_network_iters
            if i % (policy_network_iters[0] + policy_network_iters[1]) in range(policy_network_iters[0]):
                self.train_step_with_reg('mtl', policy_op, alter_train='policy', tau=tau,loss_lambda=loss_lambda)
                policy_count += 1
            # Step 2-2: Train network when the current iter is in the second part of policy_network_iters
            else:
                self.train_step_with_reg('mtl', network_op, alter_train='network', tau=tau, scale=policy_scale, loss_lambda=loss_lambda)

            # Step 3: Update tau in policy every self.policy_update_iters
            if policy_count > self.policy_update_iters and tau > 1e-6:
                tau = tau * tau_ratio
                print('tau: ' + str(tau))
                policy_count = 0

            # Step 4: Print loss
            if (i+1) % self.print_iters == 0:
                self.print_train_loss_with_reg(i, writer)
                self.reset_train_loss()
            
            # Step 5: Validation
            if (i+1) % self.val_iters == 0:
                self.validate('mtl', i, tau=tau, hard=True, writer=writer)
                self.model.train()
                
            # Step 6: Save model
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    state = {'iter': i,
                            'state_dict': self.model.state_dict(),
                            'policy_op': policy_op.state_dict(),
                            'network_op': network_op.state_dict(),
                            'tau': tau}
                    if isinstance(loss_lambda, dict):
                        modelName = 'alter_train_with_reg_' + str(loss_lambda['policy']).split('.')[1]
                    elif isinstance(loss_lambda, float):
                        modelName = 'alter_train_with_reg_' + str(loss_lambda).split('.')[1]
                    self.save_model(state, modelName, savePath)

            ################ delete if not needed ############
            if (i+1) % 10000 == 0:
                for g in policy_op.param_groups:
                    g['lr'] = 0.001
                    print('lr changed')
            #################################################
                    
        # Reset loss list and the data iters
        self.set_train_loss_data_iter()
        return
    
    def post_train(self, iters, lr=0.001, task_iters=None, loss_lambda=None, 
                  decay_lr_freq=4000, decay_lr_rate=0.5, 
                  writerPath=None, savePath=None, reload=None,
                  ext=""):
        self.model.train()
        # Key point: set grad of parameters of policy to be false
        if writerPath != None:
            writer = SummaryWriter(log_dir=writerPath+'post_train/')
        else:
            writer = None
        
        start = 0
        if reload is not None and savePath is not None:
            state = torch.load(savePath + reload)
            if 'post_train' in reload:
                start = state['iter'] + 1
                self.model.load_state_dict(state['state_dict'])
            else:
                # Only load the policy parmas
                policy_params = state['state_dict'].copy()
                for key in state['state_dict']:
                    if 'policy' not in key:
                        del policy_params[key]
                print('Task Policy:')
                print(policy_params, flush=True)
                self.model.load_state_dict(policy_params, strict=False)
                
        # Step 1: Freeze policy parameters
        self.freeze_policy()
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)   
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4)
        if reload is not None and savePath is not None:
            if 'post_train' in reload:
                optimizer.load_state_dict(state['optimizer'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)

        for i in range(start, iters):
            # Step 2: Train the network with policy
            if task_iters is None:
                self.train_step('mtl', optimizer, scheduler, hard=True, loss_lambda=loss_lambda)
            else:
                task_idx = self.which_task(i, task_iters)
                self.train_step_task('mtl', self.tasks[task_idx], optimizer, scheduler, hard=True)
            if (i+1) % self.print_iters == 0:
                for param_group in optimizer.param_groups:
                    print("Current lr: ", param_group['lr'])
                self.print_train_loss(i, writer)
                self.reset_train_loss()
            if (i+1) % self.val_iters == 0:
                self.validate('mtl', i, hard=True, writer=writer)
                self.model.train()
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    state = {'iter': i,
                            'state_dict': self.model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                    self.save_model(state, f'post_train{ext}', savePath)
            
        # Reset loss list and the data iters
        self.set_train_loss_data_iter()
        self.unfreeze_all_weights()
        return
    

    # Helper Functions - Train and Validation
    def train_step(self, stage, optimizer, scheduler=None, alter_train=None, tau=1, hard=False, policy_idx=None, loss_lambda=None):
        # Function: Train one iter for each task 
        for task in self.tasks: 
            if alter_train is None:
                try:
                    data = next(self.train_iter_dict[task])
                except StopIteration:
                    self.train_iter_dict[task] = iter(self.train_dataloader_dict[task])
                    continue
                except:
                    continue
            elif alter_train == 'network':
                try:
                    data = next(self.train_network_iter_dict[task])
                except StopIteration:
                    self.train_network_iter_dict[task] = iter(self.train_dataloader_dict[task])
                    continue
                except:
                    continue
            elif alter_train == 'policy':
                try:
                    data = next(self.train_policy_iter_dict[task])
                except StopIteration:
                    self.train_policy_iter_dict[task] = iter(self.train_dataloader_dict[task])
                    continue
                except:
                    continue
            
            x = data[0].to(self.device)
            y = data[1].to(self.device)

            optimizer.zero_grad()
            output = self.model(x, stage, task, tau, hard, policy_idx)
            if 'mask' in data:
                loss = self.criterion_dict[task](output, y, data['mask'].to(self.device))
            else:
                loss = self.criterion_dict[task](output, y)
                
            if isinstance(loss_lambda, dict):
                loss = loss_lambda[task] * loss 
            elif loss_lambda is None:
                pass
            else:
                sys.exit('Loss weights (lambda) should be in the type of dictionary.')
            
            loss.backward()
            optimizer.step()
            self.loss_list[task].append(loss.item())  
            
        if scheduler is not None:
            scheduler.step()
        return
    
    def train_step_with_reg(self, stage, optimizer, scheduler=None, alter_train=None, 
                            tau=1, hard=False, 
                            policy_idx=None, scale=6, loss_lambda=1.0):
        # Function: Train one iter for each task 
        for task in self.tasks:
            if alter_train is None:
                try:
                    data = next(self.train_iter_dict[task])
                except StopIteration:
                    self.train_iter_dict[task] = iter(self.train_dataloader_dict[task])
                    continue
                except:
                    continue
            elif alter_train == 'network':
                try:
                    data = next(self.train_network_iter_dict[task])
                except StopIteration:
                    self.train_network_iter_dict[task] = iter(self.train_dataloader_dict[task])
                    continue
                except:
                    continue
            elif alter_train == 'policy':
                try:
                    data = next(self.train_policy_iter_dict[task])
                except StopIteration:
                    self.train_policy_iter_dict[task] = iter(self.train_dataloader_dict[task])
                    continue
                except:
                    continue
                
            x = data[0].to(self.device)
            y = data[1].to(self.device)

            optimizer.zero_grad()
            output = self.model(x, stage, task, tau, hard, policy_idx)
            if 'mask' in data:
                tloss = self.criterion_dict[task](output, y, data['mask'].to(self.device))
            else:
                tloss = self.criterion_dict[task](output, y)
            
            regloss = self.model.policy_reg(task, policy_idx, tau, scale)
            if isinstance(loss_lambda, dict):
                loss = loss_lambda[task] * tloss + loss_lambda['policy'] * regloss
            elif isinstance(loss_lambda, float):
                loss = tloss + loss_lambda * regloss
            else:
                sys.exit('Loss weights (lambda) should be in the type of dictionary or float.')
            
            loss.backward()
            optimizer.step()
            self.loss_list[task].append((tloss.item(), regloss.item(), loss.item()))  
            
        if scheduler is not None:
            scheduler.step()
        return
    
    def train_step_task(self, stage, task, optimizer, scheduler=None, alter_train=None, tau=1, hard=False, policy_idx=None):
        # Function: Train one iter for one task 
        if alter_train is None:
            try:
                data = next(self.train_iter_dict[task])
            except StopIteration:
                self.train_iter_dict[task] = iter(self.train_dataloader_dict[task][0])
                return
            except:
                return
        elif alter_train == 'network':
            try:
                data = next(self.train_network_iter_dict[task])
            except StopIteration:
                self.train_network_iter_dict[task] = iter(self.train_dataloader_dict[task][1])
                return
            except:
                return
        elif alter_train == 'policy':
            try:
                data = next(self.train_policy_iter_dict[task])
            except StopIteration:
                self.train_policy_iter_dict[task] = iter(self.train_dataloader_dict[task][2])
                return
            except:
                return

        x = data[0].to(self.device)
        y = data[1].to(self.device)

        optimizer.zero_grad()
        output = self.model(x, stage, task, tau, hard, policy_idx)
        if 'mask' in data:
            loss = self.criterion_dict[task](output, y, data['mask'].to(self.device))
        else:
            loss = self.criterion_dict[task](output, y)
        loss.backward()
        optimizer.step()
        self.loss_list[task].append(loss.item())  
            
        if scheduler is not None:
            scheduler.step()
        return
    
    def validate(self, stage, it=0, tau=1, hard=False, writer=None, policy_idx=None):
        self.model.eval()
        for task in self.tasks:
            loss_list = []
            val_running_counter = 0
            for i, data in enumerate(self.val_dataloader_dict[task]):
                x = data[0].to(self.device)
                y = data[1].to(self.device)

                output = self.model(x, stage, task, tau, hard, policy_idx)

                if 'mask' in data:
                    loss = self.criterion_dict[task](output, y, data['mask'].to(self.device))
                    self.metric_dict[task](output, y, data['mask'].to(self.device))
                else:
                    loss = self.criterion_dict[task](output, y)
                    # self.metric_dict[task](output, y)
                    # val_counter_ = torch.eq(torch.argmax(y, dim=1), 
                    #                         torch.argmax(output, dim=1)).float().sum()
                    _, predicted = output.max(1)
                    val_counter_ = predicted.eq(y).sum()

                loss_list.append(loss.item())
                val_running_counter += val_counter_
            
            avg_loss = np.mean(loss_list)
            # val_results = self.metric_dict[task].val_metrics()
            val_results = 100. * val_running_counter / len(self.val_dataloader_dict[task].dataset)

            if writer != None:
                writer.add_scalar('Loss/val/' + task, avg_loss, it)
            print('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task, avg_loss), flush=True)
            print(val_results.item(), flush=True)
        print('======================================================================', flush=True)
        return
    
    def validate_task(self, stage, it, task, tau=1, hard=False, writer=None, policy_idx=None):
        self.model.eval()
        loss_list = []
        for i, data in enumerate(self.val_dataloader_dict[task]):
            x = data[0].to(self.device)
            y = data[1].to(self.device)

            output = self.model(x, stage, task, tau, hard, policy_idx)

            if 'mask' in data:
                loss = self.criterion_dict[task](output, y, data['mask'].to(self.device))
                self.metric_dict[task](output, y, data['mask'].to(self.device))
            else:
                loss = self.criterion_dict[task](output, y)
                self.metric_dict[task](output, y)

            loss_list.append(loss.item())

        avg_loss = np.mean(loss_list)
        val_results = self.metric_dict[task].val_metrics()
        if writer != None:
            writer.add_scalar('Loss/val/' + task, avg_loss, it)
        print('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task, avg_loss), flush=True)
        print(val_results, flush=True)
        print('======================================================================', flush=True)
        return
    
    def print_train_loss(self, it, writer=None):
        # Function: Print loss for each task
        total_loss = 0
        task_num = 0
        for task in self.tasks:
            if self.loss_list[task]:
                avg_loss = np.mean(self.loss_list[task])
            else:
                continue
            total_loss += avg_loss
            task_num += 1
            if writer != None:
                writer.add_scalar('Loss/train/' + task, avg_loss, it)
            print('[Iter {} Task {}] Train Loss: {:.4f}'.format((it+1), task[:4], avg_loss), flush=True)
        print('[Iter {} Total] Train Loss: {:.4f}'.format((it+1), total_loss/task_num), flush=True)
        print('======================================================================', flush=True)
        return
    
    def print_train_loss_with_reg(self, it, writer=None):
        # Function: Print loss for each task
        total_loss = 0
        task_num = 0
        
        for task in self.tasks:
            if self.loss_list[task]:
                avg_tloss = np.mean([x[0] for x in self.loss_list[task]])
                avg_regloss = np.mean([x[1] for x in self.loss_list[task]])
                avg_loss = np.mean([x[2] for x in self.loss_list[task]])
            else:
                continue
            total_loss += avg_loss
            task_num += 1
            if writer != None:
                writer.add_scalar('Loss/train/' + task, avg_loss, it)
            print('[Iter {} Task {}] Task Loss: {:.4f} Reg Loss: {:.4f} Train Loss: {:.4f}'.format((it+1), task, avg_tloss, avg_regloss, avg_loss), flush=True)
        print('[Iter {} Total] Train Loss: {:.4f}'.format((it+1), total_loss/task_num), flush=True)
        print('======================================================================', flush=True)
        return
    
    def save_model(self, state, stage, savePath):
        i = state['iter']
        torch.save(state, savePath + stage + '_' + str(i+1) + 'iter.model')
        if os.path.exists(savePath + stage + '_' + str((i+1)-(self.save_num*self.save_iters))+'iter.model'):
              os.remove(savePath + stage + '_' + str((i+1)-(self.save_num*self.save_iters))+'iter.model')
        return
    
    def load_model(self, savePath, reload):
        state = torch.load(savePath + reload)
        self.model.load_state_dict(state['state_dict'])
        return
    
    # Helper Functions - Utils
    def which_task(self, it, task_iters):
        sum_time = sum(task_iters)
        it_mod = it % sum_time
        for i in range(len(task_iters)):
            lb = sum(task_iters[:i])
            ub = sum(task_iters[:i+1])
            if it_mod in range(lb,ub):
                return i
    
    def reset_train_loss(self):
        for task in self.tasks:
            self.loss_list[task] = []
            self.mixed_loss_list[task] = []
        return
    
    def set_train_loss_data_iter(self):
        for task in self.tasks:
#             self.train_iter_dict[task] = iter(self.cycle(self.train_dataloader_dict[task][0]))
            self.train_iter_dict[task] = iter(self.train_dataloader_dict[task])
            self.train_network_iter_dict[task] = iter(self.train_dataloader_dict[task])
            self.train_policy_iter_dict[task] = iter(self.train_dataloader_dict[task])
            self.loss_list[task] = []
            self.mixed_loss_list[task] = []
        return
    
    def cycle(self, iterable):
        while True:
            for x in iterable:
                yield x
                
    def freeze_policy(self):
        for name, param in self.model.named_parameters():
            if 'policy' in name:
                param.requires_grad = False
        return
    
    def freeze_and_share_bottom_policy(self, share_num):
        self.model.share_bottom_policy(share_num)    
        return
      
    def unfreeze_all_weights(self):
        for name, param in self.model.named_parameters():
             param.requires_grad = True
        return


    # ================================================

    def post_train_prune(self, args, 
                         lr=0.001, decay_lr_freq=4000, decay_lr_rate=0.5, 
                         writerPath=None, savePath=None,
                         ext=""):
        
        self.model.train()
        if writerPath != None:
            writer = SummaryWriter(log_dir=writerPath+'post_train_prune/')
        else:
            writer = None

        self.freeze_policy()
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)   
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)

        """=============="""
        """  ADMM Train  """
        """=============="""

        initial_rho = args.rho
        if args.admm_train:
            print("\n!!!!!!!!!!!!!!!!!!! ADMM TRAIN PHASE !!!!!!!!!!!!!!!!!!!")

            if (not args.has_wandb) or (not args.enable_wandb):
                condition_d = {}
                mixed_losses = []
                ce_loss = []

            admm_dir = savePath+"/slr_train/"+args.config_file+"/"
            if not os.path.exists(admm_dir):
                os.makedirs(admm_dir)
            hp_dir = savePath+"/slr_hp/"+args.config_file+"/"
            if not os.path.exists(hp_dir):
                os.makedirs(hp_dir)

            for i in range(args.rho_num):
                current_rho = initial_rho * 10 ** i
                if i == 0:
                    baseline_path = savePath + args.baseline
                    print("Baseline Model: ", baseline_path)
                    state = torch.load(baseline_path)
                    self.model.load_state_dict(state['state_dict'])
                    self.validate('mtl', 0, hard=True, writer=writer)

                ADMM = admm.ADMM(args, self.model, "profile/" + args.config_file + ".yml", rho=current_rho)
                admm.admm_initialization(args, ADMM, self.model)  # intialize Z and U variables
            
                # ----------------------------

                for epoch in range(1, args.epochs+1):

                    if args.lr_scheduler == 'default':
                        admm.admm_adjust_learning_rate(args, optimizer, epoch)
                    elif args.lr_scheduler in ['step']:
                        scheduler.step()
                    print("current rho: {}".format(current_rho))

                    mixed_loss, loss = self.slr_train(args, ADMM, optimizer, epoch, writer)
                    
                    for param_group in optimizer.param_groups:
                        print("Current lr: ", param_group['lr'])
                    # self.print_train_loss(epoch, writer)
                    self.reset_train_loss()

                    self.validate('mtl', epoch, hard=True, writer=writer)

                    if args.has_wandb and args.enable_wandb:
                        wandb.log({"train/ce_loss": loss})
                        wandb.log({"train/mixed_losses": mixed_loss})
                    else:
                        ce_loss.append(loss)
                        mixed_losses.append(mixed_loss)

                    ### save model
                    state = {'iter': epoch,
                             'state_dict': self.model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    self.save_model(state, f'{args.config_file}_{args.sparsity_type}{args.ext}', admm_dir)

                    print("Condition 1")
                    print(ADMM.condition1)
                    print("Condition 2")
                    print(ADMM.condition2)
                    
                    if (not args.has_wandb) or (not args.enable_wandb):
                        condition_d["Condition1"] = condition_d.get("Condition1", [])+ADMM.condition1
                        condition_d["Condition2"] = condition_d.get("Condition2", [])+ADMM.condition2

                print("----------------> Accuracy after hard-pruning ...")
                model_forhard = self.model
                admm.hard_prune(args, ADMM, model_forhard)
                admm.test_sparsity(args, ADMM, model_forhard)
                self.validate('mtl', epoch, hard=True, writer=writer)

                ### save model
                state = {'iter': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                self.save_model(state, f'{args.config_file}_{args.sparsity_type}{args.ext}', hp_dir)

                ### save result
                if (not args.has_wandb) or (not args.enable_wandb):
                    if not os.path.exists(savePath+"/results"):
                        os.makedirs(savePath+"/results")
                    f = open(savePath+"/results/mixed_losses{}.pkl".format(current_rho),"wb")
                    pickle.dump(mixed_losses,f)
                    f.close()
                    f = open(savePath+"/results/ce_loss{}.pkl".format(current_rho),"wb")
                    pickle.dump(ce_loss,f)
                    f.close()
                    f = open(savePath+"/results/condition.pkl", "wb")
                    pickle.dump(condition_d, f)
                    f.close()

        """================"""
        """End ADMM retrain"""
        """================"""

        """================"""
        """ Masked retrain """
        """================"""
        if args.masked_retrain:

            if (not args.has_wandb) or (not args.enable_wandb):
                epoch_train_loss= []

            print("\n!!!!!!!!!!!!!!!!!!! RETRAIN PHASE !!!!!!!!!!!!!!!!!!!")
            retrain_dir = savePath+"/slr_retrain/"
            if not os.path.exists(retrain_dir):
                os.makedirs(retrain_dir)

            ### load admm trained model
            print("\n---------------> Loading slr trained file...")
            admm_dir = savePath+"/slr_train/"
            state = torch.load(admm_dir + args.slr_model)
            self.model.load_state_dict(state['state_dict'])

            print("\n---------------> Accuracy before hardpruning")
            self.validate('mtl', 0, hard=True, writer=writer)

            ADMM = admm.ADMM(args, self.model, file_name="profile/" + args.config_file + ".yml", rho=initial_rho)
            admm.hard_prune(args, ADMM, self.model)
            admm.test_sparsity(args, ADMM, self.model)

            print("\n---------------> Accuracy after hard-pruning")
            self.validate('mtl', 0, hard=True, writer=writer)

            # ------------------

            for epoch in range(1, args.retrain_epoch+1):

                epoch_loss = []
                scheduler.step()

                avg_loss = self.masked_retrain(args, ADMM, optimizer, epoch)
                for param_group in optimizer.param_groups:
                    print("Current lr: ", param_group['lr'])
                # self.print_train_loss(epoch, writer)
                self.reset_train_loss()
                prec_rt = self.validate('mtl', epoch, hard=True, writer=writer)

                if args.has_wandb and args.enable_wandb:
                    wandb.log({"retrain/retrain_train_loss": avg_loss})
                else:
                    epoch_train_loss.append(epoch_loss)

                ### save model
                state = {'iter': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                self.save_model(state, f'{args.config_file}_{args.sparsity_type}{args.ext}', retrain_dir)

            print("---------------> After retraining")
            self.validate('mtl', epoch, hard=True, writer=writer)
            admm.test_sparsity(args, ADMM, self.model)

        """=================="""
        """End masked retrain"""
        """=================="""

        # Reset loss list and the data iters
        self.set_train_loss_data_iter()
        self.unfreeze_all_weights()
        return    

    # -------------------------------------------------------

    def slr_train(self, args, ADMM, optimizer, epoch, writer):

        ce_loss, mixed_loss = None, None
        ctr=0
        total_ce = 0
        total_loss, total_mix_loss, task_num = 0, 0, 0

        self.model.train()

        for task in self.tasks: 

            train_running_counter = 0
            mixed_loss_list, ce_loss_list = [], []

            train_loader = self.train_dataloader_dict[task]
            for batch_idx, data in enumerate(train_loader):
                ctr += 1

                x = data[0].to(self.device)
                y = data[1].to(self.device)

                optimizer.zero_grad()

                output = self.model(x, "mtl", task, 1, True, None)
                ce_loss = self.criterion_dict[task](output, y)
                total_ce = total_ce + float(ce_loss.item())

                _, predicted = output.max(1)
                train_counter_ = predicted.eq(y).sum()
                
                admm.z_u_update(args, ADMM, self.model, self.device, 
                                optimizer, epoch, data, batch_idx, writer)  # update Z and U variables
                
                ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, self.model, ce_loss)  # append admm losss

                mixed_loss.backward(retain_graph=True)
                optimizer.step()

                self.loss_list[task].append(ce_loss.item())
                self.mixed_loss_list[task].append(mixed_loss.item())

                ce_loss_list.append(ce_loss.item())
                mixed_loss_list.append(mixed_loss.item())
                train_running_counter += train_counter_
            
            ce_avg_loss = np.mean(ce_loss_list)
            mix_avg_loss = np.mean(mixed_loss_list)
            train_results = 100. * train_running_counter / len(train_loader.dataset)

            if args.has_wandb and args.enable_wandb:
                wandb.log({f"train/{task}_train_acc": train_results})
                wandb.log({f"train/{task}_ce_loss": ce_avg_loss})
                wandb.log({f"train/{task}_mixed_losses": mix_avg_loss})
            print('[Epoch {}/{} Task {}] Cross_entropy Loss: {:.4f} Mixed_loss: {:.4f}'.format(epoch, args.epochs, task, 
                                                                                               ce_avg_loss, mix_avg_loss), 
                                                                            flush=True)
            print(train_results.item(), flush=True)
                    
        ADMM.ce_prev = ADMM.ce
        ADMM.ce = total_ce / ctr
        
        for task in self.tasks:
            if self.loss_list[task]:
                ce_avg_loss = np.mean(self.loss_list[task])
                mix_avg_loss = np.mean(self.mixed_loss_list[task])
            else:
                continue
            total_loss += ce_avg_loss
            total_mix_loss += mix_avg_loss
            task_num += 1
        print('[Epoch {} Total] Train CE Loss: {:.4f} MIX Loss: {:.4f}'.format(epoch, 
                                                                               total_loss/task_num, 
                                                                               total_mix_loss/task_num), 
                                                                        flush=True)
        print('======================================================================', flush=True)

        return total_mix_loss/task_num, total_loss/task_num


    def masked_retrain(self, args, ADMM, optimizer, epoch):

        total_loss, task_num = 0, 0
        masks = {}

        self.model.train()

        for task in self.tasks: 

            for i, (name, W) in enumerate(self.model.named_parameters()):
                if name not in ADMM.prune_ratios:
                    continue
                above_threshold, W = admm.weight_pruning(args, W, ADMM.prune_ratios[name])
                W.data = W
                masks[name] = above_threshold

            train_running_counter = 0
            loss_list = []

            train_loader = self.train_dataloader_dict[task]
            for batch_idx, data in enumerate(train_loader):

                x = data[0].to(self.device)
                y = data[1].to(self.device)
                
                optimizer.zero_grad()
                output = self.model(x, "mtl", task, 1, True, None)

                _, predicted = output.max(1)
                train_counter_ = predicted.eq(y).sum()

                loss = self.criterion_dict[task](output, y)
                loss.backward()

                # for i, (name, W) in enumerate(self.model.named_parameters()):
                #     if name in masks:
                #         W.grad *= masks[name]

                for i, (name, W) in enumerate(self.model.named_parameters()):
                    if name in masks:
                        if ("multiclass" not in name) and ("binary" not in name):
                            W.grad *= masks[name]

                optimizer.step()
                self.loss_list[task].append(loss.item())

                loss_list.append(loss.item())
                train_running_counter += train_counter_

            avg_loss = np.mean(loss_list)
            train_results = 100. * train_running_counter / len(train_loader.dataset)

            if args.has_wandb and args.enable_wandb:
                wandb.log({f"retrain/{task}_retrain_train_acc": train_results})
                wandb.log({f"retrain/{task}_loss": avg_loss})

            print('[Retrain Epoch {}/{} Task {}] Cross_entropy Loss: {:.4f}'.format(epoch, args.epochs, task, avg_loss), flush=True)
            print(train_results.item(), flush=True)
        
        for task in self.tasks:
            if self.loss_list[task]:
                avg_loss = np.mean(self.loss_list[task])
            else:
                continue
            total_loss += avg_loss
            task_num += 1

        print('[Retrain Epoch {} Total] Train Loss: {:.4f}'.format(epoch, total_loss/task_num), flush=True)
        print('======================================================================', flush=True)

        return avg_loss
    # =================================================