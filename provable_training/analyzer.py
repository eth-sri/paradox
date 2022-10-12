# from gurobipy import GRB, Model, LinExpr
from provable_training.networks import Normalization
from provable_training.deeppoly import dp_linear, dp_normalize, dp_relu, dp_conv
from provable_training.hybridzono import HybridZonotope
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
    Certification methods that analyze the network on given inputs
"""


# DeepPoly/CROWN backsubstitution
def backward_deeppoly(net, layer_idx, expr_coeff, layer_sizes, layer_bounds, inputs, eps, data_range, skip_ub=False, lambda_ls=None, crown_variant=None, soft_slope_gamma=None):
    # Init coeff matrices: [batch_size, k, *layer_j_shape]
    # (after backsubstituting from curr_layer up to layer j)
    # (l_coeff[:, i, *:] is the lower bound on neuron i value, represented as 1 coeff. per layer-j neuron)
    l_coeff = expr_coeff
    u_coeff = expr_coeff if not skip_ub else None

    # Init bias, now 0 but will be of shape [batch_size, k]
    # Aggregate all per-neuron bias values that appear while backsubstituting
    l_sum_bias = 0
    u_sum_bias = 0 if not skip_ub else None
    
    # Backsubstitute through layer j
    # Go from pre-layer-(j+1) coeffs to pre-layer(j) coeffs
    # And aggregate bias values that appear at this layer
    # (If skip_ub we will just pass around nones)
    for j in range(layer_idx, -1, -1):
        layer = net.layers[j]
        l_bias, u_bias = None, None

        if isinstance(layer, nn.Linear):
            l_coeff, l_bias, u_coeff, u_bias = dp_linear(l_coeff, u_coeff, layer.weight, layer.bias)
        elif isinstance(layer, nn.Flatten):
            l_coeff = l_coeff.view(*l_coeff.size()[:2], *layer_sizes[j][1:])
            if not skip_ub:
                u_coeff = u_coeff.view(*u_coeff.size()[:2], *layer_sizes[j][1:])
        elif isinstance(layer, Normalization):
            l_coeff, l_bias, u_coeff, u_bias = dp_normalize(l_coeff, u_coeff, layer.mean.view(-1), layer.sigma.view(-1))
        elif isinstance(layer, nn.ReLU):
            lambda_l = None if lambda_ls is None else lambda_ls[j]
            l_coeff, l_bias, u_coeff, u_bias = dp_relu(l_coeff, u_coeff, layer_bounds[j], lambda_l=lambda_l, variant=crown_variant, soft_slope_gamma=soft_slope_gamma)
        elif isinstance(layer, nn.Conv2d):	
            l_coeff, l_bias, u_coeff, u_bias = dp_conv(
                layer_sizes[j][2:], l_coeff, u_coeff, layer.weight, layer.bias, layer.stride, layer.padding, layer.groups, layer.dilation)
        else:
            raise RuntimeError(f'Unknown layer type: {type(layer)}')

        if l_bias is not None:
            l_sum_bias = l_bias + l_sum_bias
        if u_bias is not None:
            u_sum_bias = u_bias + u_sum_bias
    
    # After obtaining the coefficients in terms of the input shape, plug in min/max from the L_inf ball and sum up to get the final bounds
    min_inputs = torch.clamp(inputs - eps, min=data_range[0]).unsqueeze(1)
    max_inputs = torch.clamp(inputs + eps, max=data_range[1]).unsqueeze(1)

    neg_l_coeff, pos_l_coeff = l_coeff.clamp(max=0), l_coeff.clamp(min=0)
    l_bias = (pos_l_coeff * min_inputs + neg_l_coeff * max_inputs).view(*l_coeff.size()[:2], -1).sum(2)
    ret_l = l_bias + l_sum_bias

    if not skip_ub:
        neg_u_coeff, pos_u_coeff = u_coeff.clamp(max=0), u_coeff.clamp(min=0)
        u_bias = (pos_u_coeff * max_inputs + neg_u_coeff * min_inputs).view(*u_coeff.size()[:2], -1).sum(2)
        ret_u = u_bias + u_sum_bias
    else:
        ret_u = None

    # [batch_size, k]
    return ret_l, ret_u


# Constructs the specification matrix for logit differences C
# ([b, nb_classes-1, nb_classes] matrix, each row is true_class - class_i)
# (!) Different than "C" in papers (class_i - true_class)
def get_diffs_C(targets, nb_classes=10):
    # [b, 1, nb_classes], one-hot encoding of true labels
    c1 = torch.eye(nb_classes)[targets].unsqueeze(1)     
    # [1, nb_classes, nb_classes]
    c2 = torch.eye(nb_classes).unsqueeze(0) 
    # [batch_size, nb_classes, nb_classes]
    C = c1 - c2

    # remove specifications to self, [batch_size, nb_classes] binary mask
    # used to remove all-zero self-class rows
    # final c: [batch_size, nb_classes-1, nb_classes]
    I = (~(targets.data.unsqueeze(1) == torch.arange(nb_classes).type_as(targets.data).unsqueeze(0)))
    C = (C[I].view(targets.shape[0], nb_classes-1, nb_classes))
    return C.to(targets.device)


# Constructs the identity specification matrix for the current layer
# ([b, k, *layer_size[1:]]
def get_eye_C(targets, layer_size):
    # (used in training)
    k = np.prod(layer_size[1:]) # num_elements without the batch
    iden = torch.eye(k, k).view(-1, *layer_size[1:]).unsqueeze(0).to(targets.device)
    expr_coeff = torch.cat(layer_size[0] * [iden])
    return expr_coeff.to(targets.device)


# loss landscape matters ("iclr2021"): do pgd iterations to learn lambda_ls before the final backsubstitution
def learn_lambda_ls(net, inputs, targets, eps, data_range, layer_bounds, layer_sizes, final_size):
    # Prepare lambda_ls for each ReLU as uniform
    lambda_ls = {}
    for i, layer in enumerate(net.layers):
        if isinstance(layer, nn.ReLU):
            lambda_ls[i] = torch.rand(layer_sizes[i], device=targets.device)
            
    # Params (move to args later)
    pgd_iters = 1
    pgd_eta = 1 

    # Only propagate grads through lambda_ls
    for p in net.parameters(): p.requires_grad = False

    for _ in range(pgd_iters):
        # Require grad for the next iteration
        for i, lambda_l in lambda_ls.items():
            lambda_ls[i].requires_grad = True 

        # Backsubstitute to get logit bounds
        diffs_C = get_diffs_C(targets)
        lb, _ = backward_deeppoly(net, len(net.layers)-1, diffs_C, layer_sizes, layer_bounds, inputs, eps, data_range, skip_ub=True, lambda_ls=lambda_ls) # no soft slope gamma here

        # Build a box, get loss, and backpropagate to lambda_ls
        abs_outs = HybridZonotope(lb, torch.zeros_like(lb), None, 'box')
        abs_loss = F.cross_entropy(abs_outs.get_logits(targets), targets)
        abs_loss.backward()

        # Update - PGD step
        for i, lambda_l in lambda_ls.items():
            # Could also do lambda_l.data = ... but (.data) is deprecated 
            lambda_ls[i] = torch.clamp(lambda_l - pgd_eta * lambda_l.grad.sign(), 0, 1).detach()
    
    # Require grads on the net again
    for p in net.parameters(): p.requires_grad = True

    return lambda_ls


def forward_deeppoly(net, inputs, targets, eps, data_range, bounds=None, crown_variant=None, is_iclr2021=False, C='diffs', soft_slope_gamma=None):
    x = inputs
    layer_sizes, layer_bounds = {}, {}
    for i, layer in enumerate(net.layers):
        layer_sizes[i] = x.size()
        if isinstance(layer, nn.Linear):
            x = layer(x)
        elif isinstance(layer, nn.ReLU):
            if bounds is not None:
                # Use previously supplied pre-bounds (CROWN-IBP)
                # Skip backsubstitution
                layer_bounds[i] = bounds[i]
            else:
                # Init the lb/ub coeff matrix for backsubstitution
                # inner_sz = layer_sizes[i][1:], k = prod(inner_sz)
                # [batch_sz, k, *inner_sz]
                eye_C = get_eye_C(targets, layer_sizes[i])

                # Backsubstitution: backpropagate DeepPoly inequalities to get pre-bounds for the current layer
                # [batch_sz, k]
                lb, ub = backward_deeppoly(net, i - 1, eye_C, layer_sizes, layer_bounds, inputs, eps, data_range, crown_variant=crown_variant, soft_slope_gamma=soft_slope_gamma)

                # Reshape flat bounds back to layer shape, [batch_sz, inner_sz]
                layer_bounds[i] = (lb.view(layer_sizes[i]), ub.view(layer_sizes[i]))

            x = layer(x)
        elif isinstance(layer, Normalization):
            x = layer(x)
        elif isinstance(layer, nn.Flatten):
            x = x.view((x.size()[0], -1))
        elif isinstance(layer, nn.Conv2d):	
            x = layer(x)
        else:
            raise RuntimeError(f'Unknown layer type: {type(layer)}')
    final_size = x.size()

    # ICLR 2021 loss landscape matters: learn lambda_ls for the final backsubstitution
    if is_iclr2021:
        assert bounds is not None  # Only in CROWN-IBP mode
        
        layer_bounds_detached = {}
        for k, v in layer_bounds.items():
            layer_bounds_detached[k] = (v[0].detach(), v[1].detach())

        lambda_ls = learn_lambda_ls(net, inputs, targets, eps, data_range, layer_bounds_detached, layer_sizes, final_size)
    else:
        lambda_ls = None

    if C == 'eye':
        eye_C = get_eye_C(targets, final_size)
        lb, ub = backward_deeppoly(net, len(net.layers)-1, eye_C, layer_sizes, layer_bounds, inputs, eps, data_range, lambda_ls=lambda_ls, crown_variant=crown_variant, soft_slope_gamma=soft_slope_gamma)
        return HybridZonotope(0.5 * (ub + lb), 0.5 * (ub - lb), None, 'box')
        
    # Final backsubstitution (always done, even in CROWN-IBP)
    # Use diffs C!
    diffs_C = get_diffs_C(targets)
    
    lb, _ = backward_deeppoly(net, len(net.layers)-1, diffs_C, layer_sizes, layer_bounds, inputs, eps, data_range, skip_ub=True, lambda_ls=lambda_ls, crown_variant=crown_variant, soft_slope_gamma=soft_slope_gamma)

    """
    # Return a box that describes the final bounds
    # If logits invoke .verify(), if differences simply check if lb>0
    #return HybridZonotope(0.5 * (ub + lb), 0.5 * (ub - lb), None, 'box')
    """
    # Since we now use differences for logits as well, just always return only the lb (dummy box)
    return HybridZonotope(lb, torch.zeros_like(lb), None, 'box') #, nb_unstable/nb_total


"""
    Executes the forward pass in the abstract HybridZono domain by iterating over
    layers in the network and applying the corresponding transformer
    to the input zonotope (x)

    Uses given bounds if set, in the end returns all used bounds
    
    Instances: box, deepz, hbox etc.
"""

def forward_hybridzono(net, x, bounds=None, C=None, soft_slope_gamma=None, zono_kappa=None, loosebox_round=None, loosebox_widen=None, dcs_per_one=None):
    if bounds is None:
        bounds = dict() # Doing this in the fn args is wrong (!) 

    for i, layer in enumerate(net.layers):
        if isinstance(layer, nn.ReLU):
            # round/widen before relu!

            if loosebox_round is not None:
                lb, ub = x.concretize()

                assert dcs_per_one is not None

                omega = loosebox_round
                W = dcs_per_one

                # simplified:
                # lb = lb - (ceil - lb) * round 
                # ub = ub + (ub - floor) * round

                #lb_ceil = torch.ceil(lb)
                #lb = lb_ceil - (lb_ceil - lb) * (loosebox_round + 1)

                lb = lb - omega * (torch.ceil(lb*W) - W*lb)


                #ub_floor = torch.floor(ub)
                #ub = ub_floor + (ub - ub_floor) * (loosebox_round + 1)

                ub = ub + omega * (W*ub - torch.floor(W*ub))

                mid = 0.5 * (ub + lb)
                halfrange = 0.5 * (ub - lb)
                x = HybridZonotope(mid, halfrange, None, 'box')
            
            if loosebox_widen is not None:
                lb, ub = x.concretize()
                lb = lb - loosebox_widen
                ub = ub + loosebox_widen
                mid = 0.5 * (ub + lb)
                halfrange = 0.5 * (ub - lb)
                x = HybridZonotope(mid, halfrange, None, 'box')
            
            if i not in bounds:
                bounds[i] = x.concretize()
            #lb, ub = bounds[i]
            #nb_unstable += ((ub > 0) & (lb < 0)).float().sum().item()
            #nb_total += torch.numel(ub)
            
            x = x.relu(bounds[i], soft_slope_gamma=soft_slope_gamma, zono_kappa=zono_kappa, loosebox_round=None, loosebox_widen=None)
        elif isinstance(layer, nn.ReLU6):
            x = x.relu6()
        elif isinstance(layer, nn.Conv2d):
            x = x.conv2d(layer.weight, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
        elif isinstance(layer, Normalization):
            x = layer(x)
        elif isinstance(layer, nn.Linear):
            if i == len(net.layers) - 1 and C is not None:
                # Final layer, pass C (could be eye or diffs)
                x = x.linear(layer.weight, layer.bias, C=C)
            else:
                x = x.linear(layer.weight, layer.bias)
        elif isinstance(layer, nn.Flatten):
            x = x.view((x.head.size()[0], -1))
        else:
            raise RuntimeError(f'Unknown layer type: {type(layer)}')

    # Do not detach the bounds (they should carry the gradient for the weights)
    return x, bounds #, nb_unstable/nb_total



"""
    Cauchy approximation of Zonotopes
"""
#def compute_bounds_approx(eps, blocks, layer_idx, inputs, k=50):
def forward_cauchy(domain, net, box_inputs, k=50):

    curr_head, beta = box_inputs.head, box_inputs.beta
    device = beta.device

    curr_cauchy = beta.unsqueeze(0) * torch.clamp(torch.FloatTensor(k, *beta.size()).to(device).cauchy_(), -1e10, 1e10)
    n_cauchy = 1 
    bounds_approx = {}

    batch_size = beta.shape[0]
    
    for i, layer in enumerate(net.layers):
            
        if isinstance(layer, nn.ReLU):
            lb, ub = bounds_approx[i]
            is_cross = (lb < 0) & (ub > 0)

            if domain == 'cauchy-hbox':
                relu_lambda = torch.where(is_cross, ub * 0.5, (lb >= 0).float())
                relu_mu = torch.zeros(lb.size()).to(device)
            elif domain == 'cauchy-zono':
                D = 1e-6
                relu_lambda = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).float())
                relu_mu = torch.where(is_cross, -0.5*ub*lb/(ub-lb+D), torch.zeros(lb.size()).to(device))
            else:
                raise RuntimeError(f'Unknown cauchy domain: {domain}')

            curr_head = curr_head * relu_lambda + relu_mu
            curr_cauchy = curr_cauchy * relu_lambda.unsqueeze(0)
            new_cauchy = relu_mu.unsqueeze(0) * torch.clamp(torch.FloatTensor(k, *curr_head.size()).to(device).cauchy_(), -1e10, 1e10)
            curr_cauchy = torch.cat([curr_cauchy, new_cauchy], dim=0)
            n_cauchy += 1
        elif isinstance(layer, nn.Conv2d):
            curr_head = layer(curr_head)
            tmp_cauchy = curr_cauchy.view(-1, *curr_cauchy.size()[2:])
            tmp_cauchy = F.conv2d(tmp_cauchy, layer.weight, None, layer.stride, layer.padding, layer.dilation, layer.groups)
            curr_cauchy = tmp_cauchy.view(-1, batch_size, *tmp_cauchy.size()[1:])
        elif isinstance(layer, Normalization):
            curr_head = (curr_head - layer.mean) / layer.sigma
            curr_cauchy /= layer.sigma.unsqueeze(0)
        elif isinstance(layer, nn.Linear):
            curr_head = layer(curr_head)
            curr_cauchy = torch.matmul(curr_cauchy, layer.weight.t())
        elif isinstance(layer, nn.Flatten):
            curr_head = curr_head.view(batch_size, -1)
            curr_cauchy = curr_cauchy.view(curr_cauchy.size()[0], batch_size, -1)
        else:
            raise RuntimeError(f'Unknown layer type: {type(layer)}')

        # same for the final?
        if i+1 == len(net.layers) or isinstance(net.layers[i+1], nn.ReLU):
            l1_approx = 0
            for j in range(n_cauchy):
                l1_approx += torch.median(curr_cauchy[j*k:(j+1)*k].abs(), dim=0)[0]
            lb = curr_head - l1_approx
            ub = curr_head + l1_approx
            bounds_approx[i+1] = (lb, ub)

    # Return a final box
    lb, ub = bounds_approx[len(net.layers)]
    return HybridZonotope(0.5 * (ub + lb), 0.5 * (ub - lb), None, 'box')

""" 
    LP certification using gurobi (can't be used in training, no grads)
"""
def lp_add_relu_constraints(model, in_lb, in_ub, in_neuron, out_neuron, relaxation):
    if in_ub <= 0:
        out_neuron.lb = 0
        out_neuron.ub = 0
    elif in_lb >= 0:
        model.addConstr(in_neuron, GRB.EQUAL, out_neuron)
    else:
        model.addConstr(out_neuron >= 0)
        model.addConstr(out_neuron >= in_neuron)
        if relaxation == 'triangle':
            model.addConstr(-in_ub * in_neuron + (in_ub - in_lb) * out_neuron, GRB.LESS_EQUAL, -in_lb * in_ub)
        elif relaxation == 'parallelogram':
            model.addConstr(out_neuron, GRB.LESS_EQUAL, in_ub)
            model.addConstr(out_neuron - in_neuron, GRB.LESS_EQUAL, -in_lb)
        else:
            raise RuntimeError(f'Unknown LP relaxation: {relaxation}')


def analyze_lp(eps, net, inputs, targets, relaxation):
    model = Model("LP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 10)

    x = inputs
    x_flat = inputs.view(-1).cpu().numpy()

    neurons = {}
    neurons[-1] = []
    for j in range(x_flat.shape[0]):
        lb = max(0, x_flat[j] - eps)
        ub = min(1, x_flat[j] + eps)
        neurons[-1].append(model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name='input_{}'.format(j)))
    model.update()

    for i, layer in enumerate(net.layers):
        x = layer(x)
        neurons[i] = []
        if isinstance(layer, Normalization):
            mean, sigma = 0.1307, 0.3081
            for j in range(x_flat.shape[0]):
                lb = (neurons[i-1][j].lb - mean) / sigma
                ub = (neurons[i-1][j].ub - mean) / sigma
                neurons[i] += [model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name='hidden_{}_{}'.format(i, j))]
                model.addConstr(neurons[i][j] == (neurons[i-1][j] - mean) / sigma)
        elif isinstance(layer, nn.Flatten):
            neurons[i] = neurons[i-1]
        elif isinstance(layer, nn.Linear):
            for j in range(layer.weight.size()[0]):
                expr = LinExpr()
                expr += layer.bias[j].item()
                coeffs = layer.weight[j].cpu().numpy()
                expr += LinExpr(coeffs, neurons[i-1])

                model.setObjective(expr, GRB.MINIMIZE)
                model.update()
                model.optimize()
                lb = model.objVal

                model.setObjective(expr, GRB.MAXIMIZE)
                model.update()
                model.optimize()
                ub = model.objVal

                neurons[i] += [model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name='hidden_{}_{}'.format(i, j))]
                model.addConstr(neurons[i][j] == expr)
        elif isinstance(layer, nn.ReLU):
            model.update()
            for j in range(len(neurons[i-1])):
                neurons[i] += [model.addVar(lb=0, ub=max(0, neurons[i-1][j].ub), name='hidden_{}_{}'.format(i, j))]
                lp_add_relu_constraints(model, neurons[i-1][j].lb, neurons[i-1][j].ub, neurons[i-1][j], neurons[i][j], relaxation)
        else:
            assert False

    true_label = targets[0].item()
    min_diff = None
    for j in range(10):
        if true_label == j:
            continue
        
        model.setObjective(neurons[i][true_label] - neurons[i][j], GRB.MINIMIZE)
        model.update()
        model.optimize()
        lb = model.objVal

        if min_diff is None or lb < min_diff:
            min_diff = lb
        if lb < 0:
            return 0.0, min_diff
    return 1.0, min_diff