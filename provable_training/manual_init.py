import torch
import torch.nn as nn
import math
import numpy as np

# Code from FastIBP:
# (https://github.com/shizhouxing/Fast-Certified-Robust-Training/blob/50560c6c88db774ccee7caedbe6e5021daa29321/manual_init.py)

def get_params(model):
    weights = []
    biases = []
    for p in model.named_parameters():
        if 'weight' in p[0]:
            weights.append(p)
        elif 'bias' in p[0]:
            biases.append(p)
        else:
            print('Skipping parameter {}'.format(p[0]))
    return weights, biases

def ibp_init(model):
    weights, biases = get_params(model)
    for i in range(len(weights)-1):
        if weights[i][1].ndim == 1:
            continue
        weight = weights[i][1]
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2 * math.pi / (fan_in**2))     
        std_before = weight.std().item()
        torch.nn.init.normal_(weight, mean=0, std=std)
        print(f'Reinitialize {weights[i][0]}, std before {std_before:.5f}, std now {weight.std():.5f}')
    # doesn't work for resnets

def kaiming_normal_init(model):
    for p in model.named_parameters():
        if p[0].find('.weight') != -1:
            if p[0].find('bn') != -1 or p[1].ndim == 1:
                continue
            torch.nn.init.kaiming_normal_(p[1].data)

def xavier_normal_init(model):
    for p in model.named_parameters():
        if p[0].find('.weight') != -1:
            if p[0].find('bn') != -1 or p[1].ndim == 1:
                continue
            torch.nn.init.xavier_normal_(p[1].data)

def orthogonal_init(model):
    params = []
    bns = []
    for p in model.named_parameters():
        if p[0].find('.weight') != -1:
            if p[0].find('bn') != -1 or p[1].ndim == 1:
                bns.append(p)
            else:
                params.append(p)
    for p in params[:-1]: 
        std_before = p[1].std().item()
        print('before mean abs', p[1].abs().mean())
        torch.nn.init.orthogonal_(p[1])
        print('Reinitialize {} with orthogonal matrix, std before {:.5f}, std now {:.5f}'.format(
            p[0], std_before, p[1].std()))
        print('after mean abs', p[1].abs().mean())

def manual_init(model, init_method):
    if init_method == 'ibp':
        ibp_init(model)
    elif init_method == 'kaiming-normal': 
        kaiming_normal_init(model)
    elif init_method == 'xavier-normal': 
        xavier_normal_init(model)
    elif init_method == 'orthogonal': 
        orthogonal_init(model)      
    elif init_method == 'default':
        pass
    else:
        raise ValueError(init_method)
    print(f'####### Initialized with: {init_method}')