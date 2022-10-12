import torch
import torch.nn.functional as F

# Backsubstitution utils


def dp_linear(l_coeff, u_coeff, weight, bias):
    new_l_coeff = l_coeff.matmul(weight)
    new_l_bias = l_coeff.matmul(bias)
    new_u_coeff = None if u_coeff is None else u_coeff.matmul(weight)
    new_u_bias = None if u_coeff is None else u_coeff.matmul(bias)
    return new_l_coeff, new_l_bias, new_u_coeff, new_u_bias


def dp_normalize(l_coeff, u_coeff, mean, sigma):
    # There are 1 mean & sigma per channel
    n_channels = mean.shape[0]
    assert l_coeff.shape[2] == n_channels

    # Divide each coefficient by sigma, add coeff * (-mean/sigma) to bias
    div_shape = torch.ones(l_coeff.ndim, dtype=torch.int)
    div_shape[2] = n_channels
    coeff_divisor = sigma.view(div_shape.tolist())
    bias_factor = (-mean / sigma)

    new_l_coeff = l_coeff / coeff_divisor
    new_l_bias = (l_coeff.view(*l_coeff.size()[:3], -1).sum(3) * bias_factor).sum(2)

    new_u_coeff = None if u_coeff is None else u_coeff / coeff_divisor
    new_u_bias = None if u_coeff is None else (u_coeff.view(*u_coeff.size()[:3], -1).sum(3) * bias_factor).sum(2)
    
    return new_l_coeff, new_l_bias, new_u_coeff, new_u_bias


def dp_relu(l_coeff, u_coeff, bounds, lambda_l=None, variant=None, soft_slope_gamma=None):
    # lambda_l*x + mu_l (<=) y (<=) lambda_u*x + mu_u
    # we could have bounds (CROWN-IBP)
    # we could have a predetermined lambda_l for the cross case (ICLR2021)
    lb, ub = bounds
    
    """
        [Variants]
        ['0','1', '0-c', '1-c', 'lower-tria', 'upper-tria', 'lower-tria-c', 'upper-tria-c']

        standard:   (0, 0)    (0/x, Kx-Kl)    (x, x)
        
        0:          (0, 0)    (0, Kx-Kl)      (x, x)
        0-c:        (0, 0)    (0, Kx-Kl)      (l, x)

        1:          (0, 0)    (x, Kx-Kl)      (x, x)
        1-c:        (x, 0)    (x, Kx-Kl)      (x, x)

        lower-tria: (0, 0)    (0, x-l)        (x, x)
        upper-tria: (0, 0)    (x, u)          (x, x)
        lower-tria-c: (0, x-l)  (0, x-l)        (l, x)
        upper-tria-c: (x, 0)    (x, u)          (x, u)
    """

    #print(f'Variant: {variant}')

    if variant is None: 
        # CROWN/CROWN-IBP/ICLR2021
        if lambda_l is None: # so not ICLR2021 (supplied: random / random+pgd)
            if soft_slope_gamma is not None:
                # Soft slope!
                D = 1e-6
                lambda_l = torch.sigmoid(soft_slope_gamma * (lb/(ub+D) - ub/(lb-D)))
                #closest = (lambda_l - 0.5).abs().min().item()
                #print(f'using soft slope! min={lambda_l.min().item()} max={lambda_l.max().item()} closest={closest}')
            else:
                # Heuristic
                lambda_l = torch.where(ub < -lb, torch.zeros(lb.size()).to(lb.device), torch.ones(lb.size()).to(lb.device))
        lambda_l = torch.where(ub < 0, torch.zeros(lb.size()).to(lb.device), lambda_l)
        lambda_l = torch.where(lb > 0, torch.ones(lb.size()).to(lb.device), lambda_l)

        lambda_u = ub / (ub - lb + 1e-9)
        lambda_u = torch.where(ub < 0, torch.zeros(ub.size()).to(ub.device), lambda_u)
        lambda_u = torch.where(lb > 0, torch.ones(ub.size()).to(ub.device), lambda_u)
        
        mu_l = torch.zeros(lb.size()).to(lb.device)
        mu_u = torch.where((lb < 0) & (ub > 0), -ub * lb / (ub - lb + 1e-9), torch.zeros(lb.size()).to(lb.device))
    elif variant in ['0', '0-c']:
        # 0:          (0, 0)    (0, Kx-Kl)      (x, x)
        # 0-c:        (0, 0)    (0, Kx-Kl)      (l, x)
        lambda_l = torch.zeros(lb.size()).to(lb.device)
        if variant == '0':
            lambda_l = torch.where(lb > 0, torch.ones(lb.size()).to(lb.device), lambda_l)
        
        lambda_u = ub / (ub - lb + 1e-9)
        lambda_u = torch.where(ub < 0, torch.zeros(ub.size()).to(ub.device), lambda_u)
        lambda_u = torch.where(lb > 0, torch.ones(ub.size()).to(ub.device), lambda_u)

        mu_l = torch.zeros(lb.size()).to(lb.device)
        if variant == '0-c':
            mu_l = torch.where(lb > 0, lb, mu_l)
    
        mu_u = torch.where((lb < 0) & (ub > 0), -ub * lb / (ub - lb + 1e-9), torch.zeros(lb.size()).to(lb.device))
    elif variant in ['1', '1-c']:
        # 1:          (0, 0)    (x, Kx-Kl)      (x, x)
        # 1-c:        (x, 0)    (x, Kx-Kl)      (x, x)
        lambda_l = torch.ones(lb.size()).to(lb.device)
        if variant == '1':
            lambda_l = torch.where(ub < 0, torch.zeros(lb.size()).to(lb.device), lambda_l)
        
        lambda_u = ub / (ub - lb + 1e-9)
        lambda_u = torch.where(ub < 0, torch.zeros(ub.size()).to(ub.device), lambda_u)
        lambda_u = torch.where(lb > 0, torch.ones(ub.size()).to(ub.device), lambda_u)

        mu_l = torch.zeros(lb.size()).to(lb.device)
        mu_u = torch.where((lb < 0) & (ub > 0), -ub * lb / (ub - lb + 1e-9), torch.zeros(lb.size()).to(lb.device))
    elif variant in ['lower-tria', 'lower-tria-c']:
        # lower-tria: (0, 0)    (0, x-l)        (x, x)
        # lower-tria-c: (0, x-l)  (0, x-l)        (l, x)
        lambda_l = torch.zeros(lb.size()).to(lb.device)
        lambda_u = torch.ones(lb.size()).to(lb.device)
        if variant == 'lower-tria':
            lambda_l = torch.where(lb > 0, torch.ones(lb.size()).to(lb.device), lambda_l)
            lambda_u = torch.where(ub < 0, torch.zeros(lb.size()).to(lb.device), lambda_u)
        
        mu_l = torch.zeros(lb.size()).to(lb.device)
        mu_u = torch.where((lb < 0) & (ub > 0), -lb, torch.zeros(lb.size()).to(lb.device))
        if variant == 'lower-tria-c':
            mu_l = torch.where(lb > 0, lb, mu_l)
            mu_u = torch.where(ub < 0, -lb, mu_u)
    elif variant in ['upper-tria', 'upper-tria-c']:
        # upper-tria: (0, 0)    (x, u)          (x, x)
        # upper-tria-c: (x, 0)    (x, u)          (x, u)
        lambda_l = torch.ones(lb.size()).to(lb.device)
        lambda_u = torch.zeros(lb.size()).to(lb.device)
        if variant == 'upper-tria':
            lambda_l = torch.where(ub < 0, torch.zeros(lb.size()).to(lb.device), lambda_l)
            lambda_u = torch.where(lb > 0, torch.ones(lb.size()).to(lb.device), lambda_u)

        mu_l = torch.zeros(lb.size()).to(lb.device)
        mu_u = torch.where((lb < 0) & (ub > 0), ub, torch.zeros(lb.size()).to(lb.device))
        if variant == 'upper-tria-c':
            mu_u = torch.where(lb > 0, ub, mu_u)
    else:
        raise RuntimeError(f'Unknown variant: {variant}')

    ################################################################################
    ################################################################################

    lambda_l, lambda_u = lambda_l.unsqueeze(1), lambda_u.unsqueeze(1)
    mu_l, mu_u = mu_l.unsqueeze(1), mu_u.unsqueeze(1)

    neg_l_coeff, pos_l_coeff = l_coeff.clamp(max=0), l_coeff.clamp(min=0)
    new_l_coeff = pos_l_coeff * lambda_l + neg_l_coeff * lambda_u	
    new_l_bias = pos_l_coeff * mu_l + neg_l_coeff * mu_u	

    if u_coeff is not None:
        neg_u_coeff, pos_u_coeff = u_coeff.clamp(max=0), u_coeff.clamp(min=0)
        new_u_coeff = pos_u_coeff * lambda_u + neg_u_coeff * lambda_l	
        new_u_bias = pos_u_coeff * mu_u + neg_u_coeff * mu_l	
    else:
        new_u_coeff = None
            
    if len(new_l_bias.size()) == 3:	
        new_l_bias = new_l_bias.sum(2)	
        new_u_bias = None if u_coeff is None else new_u_bias.sum(2)	
    else:	
        new_l_bias = new_l_bias.sum((2, 3, 4))	
        new_u_bias = None if u_coeff is None else new_u_bias.sum((2, 3, 4))	

    return new_l_coeff, new_l_bias, new_u_coeff, new_u_bias

	
def dp_conv(preconv_wh, l_coeff, u_coeff, weight, bias, stride, padding, groups, dilation):	
    # The output padding is used to reconstruct the original pre-conv shape, 
    # as for stride>1 this might be ambiguous ([0, stride-1] rows/cols might be 
    # completely ignored on the right/bottom of the pre-conv tensor). If we know
    # the pre-conv shape we can calculate the needed output_padding.
    kernel_wh = weight.shape[-2:]
    w_padding = (preconv_wh[0] + 2*padding[0] - kernel_wh[0]) % stride[0]
    h_padding = (preconv_wh[1] + 2*padding[1] - kernel_wh[1]) % stride[1]
    output_padding = (w_padding, h_padding)

    sz = l_coeff.shape	

    new_l_coeff = F.conv_transpose2d(l_coeff.view((sz[0]*sz[1], *sz[2:])), weight, None, stride, padding, output_padding, groups, dilation)	
    new_l_coeff = new_l_coeff.view((sz[0], sz[1], *new_l_coeff.shape[1:]))
    new_l_bias = (l_coeff.sum((3,4)) * bias).sum(2)	


    if u_coeff is not None:
        new_u_coeff = F.conv_transpose2d(u_coeff.view((sz[0]*sz[1], *sz[2:])), weight, None, stride, padding, output_padding, groups, dilation)	
        new_u_coeff = new_u_coeff.view((sz[0], sz[1], *new_u_coeff.shape[1:]))	
        new_u_bias = (u_coeff.sum((3,4)) * bias).sum(2)	
    else:
        new_u_coeff, new_u_bias = None, None

    return new_l_coeff, new_l_bias, new_u_coeff, new_u_bias	
