from __future__ import print_function
import torch
import numpy as np
from numpy import linalg as LA

# ---------------------

def test_sparsity(args, model):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    print("\n|||||||||||||||||||||||||||||")

    total_zeros = 0
    total_nonzeros = 0
    compression = 0

    if args.sparsity_type == "irregular":
        for i, (name, W) in enumerate(model.named_parameters()):
            if ("policy" in name) or ("bias" in name):
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            total_nonzeros += nonzeros
            print("Sparsity at layer {} | Weights: {:.0f}, Weights after pruning: {:.0f}, %: {:.3f}, sparsity: {:.3f}".format(name, 
                                                                float(zeros + nonzeros), float(nonzeros), 
                                                                float(nonzeros) / (float(zeros + nonzeros)),
                                                                float(zeros) / (float(zeros + nonzeros))))
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(float(total_weight_number) / float(total_nonzeros)))
        compression = float(total_weight_number) / float(total_nonzeros)
        print("!!!!!!!!!!!! Compression Total| total weights: {:.0f}, total nonzeros: {:.0f}".format(total_weight_number, total_nonzeros))

    elif args.sparsity_type == "column":
        for i, (name, W) in enumerate(model.named_parameters()):
            if ("policy" in name) or ("bias" in name):
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            column_l2_norm = LA.norm(W2d, 2, axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("column sparsity of layer {} is {}".format(name, zero_column / (zero_column + nonzero_column)))
        print('only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
        compression = (total_zeros + total_nonzeros) / total_nonzeros
    
    elif args.sparsity_type == "channel":
        for i, (name, W) in enumerate(model.named_parameters()):
            if ("policy" in name) or ("bias" in name):
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W3d = W.reshape(shape[0], shape[1], -1)
            channel_l2_norm = LA.norm(W3d, 2, axis=(0,2))
            zero_row = np.sum(channel_l2_norm == 0)
            nonzero_row = np.sum(channel_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("channel sparsity of layer {} is {}".format(name, zero_row / (zero_row + nonzero_row)))
        
        print('only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
        compression = (total_zeros + total_nonzeros) / total_nonzeros

    elif args.sparsity_type == "filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if ("policy" in name) or ("bias" in name):
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            row_l2_norm = LA.norm(W2d, 2, axis=1)
            zero_row = np.sum(row_l2_norm == 0)
            nonzero_row = np.sum(row_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("filter sparsity of layer {} is {}".format(name, zero_row / (zero_row + nonzero_row)))
        print('only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
        compression = (total_zeros + total_nonzeros) / total_nonzeros

    return compression


def test_overall_sparsity(model, policy_params):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    print("\n|||||||||||||| OVERALL |||||||||||||||")

    sd = model.state_dict()

    total_zeros, total_nonzeros = 0, 0
    compression = 0

    d_tmp = {}
    for name, policy in policy_params.items():
        # print("|||", name)
        shared = torch.equal(policy, torch.tensor([1.0, 0.0, 0.0]).cuda())
        specific = torch.equal(policy, torch.tensor([0.0, 1.0, 0.0]).cuda())
        skip = torch.equal(policy, torch.tensor([0.0, 0.0, 1.0]).cuda())

        if shared:
            feature_name = name.split("policy")[0]+"basicOp.weight"
            if feature_name not in sd:
                continue
            if feature_name not in d_tmp:
                d_tmp[feature_name] = 1
                W = sd[feature_name]
            else:
                continue
        if specific: # or shared:
            feature_name = name.split("policy")[0]+"taskOp."+name.split(".")[-1]+".weight"
            if feature_name not in sd:
                continue
            W = sd[feature_name]
        elif skip:
            feature_name = name.split("policy")[0]+"dsOp."+name.split(".")[-1]+".0.weight"
            if feature_name not in sd:
                continue
            W = sd[feature_name]

        W = W.cpu().detach().numpy()
        zeros = np.sum(W == 0)
        total_zeros += zeros
        nonzeros = np.sum(W != 0)
        total_nonzeros += nonzeros
        print("Sparsity at {} | Weights: {:.0f}, After pruning: {:.0f}, %: {:.3f}, sparsity: {:.3f}".format(feature_name, 
                                                            float(zeros + nonzeros), float(nonzeros), 
                                                            float(nonzeros) / (float(zeros + nonzeros)),
                                                            float(zeros) / (float(zeros + nonzeros))))
    total_weight_number = total_zeros + total_nonzeros
    print('overal compression rate is {}'.format(float(total_weight_number) / float(total_nonzeros)))
    compression = float(total_weight_number) / float(total_nonzeros)
    print("!!!!!!!!!!!! Compression Total| total weights: {:.0f}, total nonzeros: {:.0f}".format(total_weight_number, total_nonzeros))

    return compression

# ---------------------
def test_specific_sparsity(tasks, model):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    print("\n|||||||||||||| SPECIFIC |||||||||||||||")

    compression_d = {}
    for task in tasks:
        print("\n-------------------->>>>", task)
        total_zeros = 0
        total_nonzeros = 0
        compression = 0

        for i, (name, W) in enumerate(model.named_parameters()):
            if ('bias' in name) or (task not in name) or ("policy" in name):
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            total_nonzeros += nonzeros
            print("Sparsity at {} | Weights: {:.0f}, After pruning: {:.0f}, %: {:.3f}, sparsity: {:.3f}".format(name, 
                                                                float(zeros + nonzeros), float(nonzeros), 
                                                                float(nonzeros) / (float(zeros + nonzeros)),
                                                                float(zeros) / (float(zeros + nonzeros))))
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(float(total_weight_number) / float(total_nonzeros)))
        compression = float(total_weight_number) / float(total_nonzeros)
        print("!!!!!!!!!!!! Compression Total| total weights: {:.0f}, total nonzeros: {:.0f}".format(total_weight_number, total_nonzeros))
        compression_d[task] = compression
    
    print("\n ||||| Summary:", compression_d)
    print("======================================================================== \n")

    return compression_d


def test_specific_policy_sparsity(tasks, model, policy_params):
    """
    test sparsity for every involved layer that included in the POLICY 
    and the overall compression rate
    """
    print("\n|||||||||||||| specific policy |||||||||||||||")

    sd = model.state_dict()

    compression_d = {}

    for task in tasks:
        print("\n-------------------->>>>", task)
        total_zeros, total_nonzeros = 0, 0
        compression = 0

        for name, policy in policy_params.items():
            if task not in name:
                continue

            shared = torch.equal(policy, torch.tensor([1.0, 0.0, 0.0]).cuda())
            specific = torch.equal(policy, torch.tensor([0.0, 1.0, 0.0]).cuda())
            skip = torch.equal(policy, torch.tensor([0.0, 0.0, 1.0]).cuda())

            if shared:
                feature_name = name.split("policy")[0]+"basicOp.weight"
                if feature_name not in sd:
                    continue
                W = sd[feature_name]
            if specific: # or shared:
                feature_name = name.split("policy")[0]+"taskOp."+name.split(".")[-1]+".weight"
                W = sd[feature_name]
            elif skip:
                feature_name = name.split("policy")[0]+"dsOp."+name.split(".")[-1]+".0.weight"
                if feature_name not in sd:
                    continue
                W = sd[feature_name]

            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            total_nonzeros += nonzeros
            print("Sparsity at {} | Weights: {:.0f}, After pruning: {:.0f}, %: {:.3f}, sparsity: {:.3f}".format(feature_name, 
                                                                float(zeros + nonzeros), float(nonzeros), 
                                                                float(nonzeros) / (float(zeros + nonzeros)),
                                                                float(zeros) / (float(zeros + nonzeros))))
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(float(total_weight_number) / float(total_nonzeros)))
        compression = float(total_weight_number) / float(total_nonzeros)
        print("!!!!!!!!!!!! Compression Total| total weights: {:.0f}, total nonzeros: {:.0f}".format(total_weight_number, total_nonzeros))
        compression_d[task] = compression

    print("\n ||||| Summary:", compression_d)
    print("======================================================================== \n")

    return compression_d



def test_specific_div_sparsity(tasks, model):
    """
    test sparsity for every involved layer and the overall compression rate
    Shared layer use the 'basicOp' layer to calculate
    """
    print("\n|||||||||||||| SPECIFIC (div) |||||||||||||||")

    compression_d = {}
    sd = model.state_dict()
    for task in tasks:
        print("\n-------------------->>>>", task)
        total_zeros = 0
        total_nonzeros = 0
        compression = 0

        for i, (name, W) in enumerate(model.named_parameters()):
            if ('bias' in name) or (task not in name) or ("policy" in name):
                continue
            
            W_tmp = W
            policy_name = name.split("taskOp")[0]+"policy."+task
            if (policy_name in sd) and torch.equal(sd[policy_name], torch.tensor([1.0, 0.0, 0.0]).cuda()):
                name_tmp = name.split("taskOp")[0]+"basicOp.weight"
                W_tmp = sd[name_tmp]
                name = name_tmp

            W = W_tmp.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            total_nonzeros += nonzeros
            print("Sparsity at {} | Weights: {:.0f}, After pruning: {:.0f}, %: {:.3f}, sparsity: {:.3f}".format(name, 
                                                                float(zeros + nonzeros), float(nonzeros), 
                                                                float(nonzeros) / (float(zeros + nonzeros)),
                                                                float(zeros) / (float(zeros + nonzeros))))
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(float(total_weight_number) / float(total_nonzeros)))
        compression = float(total_weight_number) / float(total_nonzeros)
        print("!!!!!!!!!!!! Compression Total| total weights: {:.0f}, total nonzeros: {:.0f}".format(total_weight_number, total_nonzeros))
        compression_d[task] = compression
    
    print("\n ||||| Summary:", compression_d)
    print("======================================================================== \n")

    return compression_d

