import numpy as np
import torch
import torch.nn.functional as F

"""
    The implementation of box, hbox and zonotopes.
"""

def linf_clamped(x, eps, clamp_min, clamp_max):
    # clamp to given range
    min_x = torch.clamp(x-eps, min=clamp_min)
    max_x = torch.clamp(x+eps, max=clamp_max)
    x_center = 0.5 * (max_x + min_x)
    x_beta = 0.5 * (max_x - min_x)
    return x_center, x_beta


def get_new_errs(is_cross, new_head, halfrange):
    # for each neuron where is_cross appears in at least one example in batch 
    # add a new error term with corresponding halfrange[] as coefficient 
    # new_head used only for shape 
    new_err_pos = (is_cross.sum(dim=0) > 0).nonzero(as_tuple=False)
    num_new_errs = new_err_pos.size()[0]

    if num_new_errs == 0:
        return None 

    nnz = is_cross.nonzero(as_tuple=False)

    if len(new_head.size()) == 2:
        batch_size, n = new_head.size()[0], new_head.size()[1]
        ids_mat = torch.zeros(n, dtype=torch.long).to(new_head.device)
        ids_mat[new_err_pos[:, 0]] = torch.arange(num_new_errs).to(new_head.device)
        mu_values = halfrange[nnz[:, 0], nnz[:, 1]]
        new_errs = torch.zeros((num_new_errs, batch_size, n)).to(new_head.device, dtype=new_head.dtype)
        err_ids = ids_mat[nnz[:, 1]]
        new_errs[err_ids, nnz[:, 0], nnz[:, 1]] = mu_values
    else:
        batch_size, n_channels, img_dim = new_head.size()[0], new_head.size()[1], new_head.size()[2]
        ids_mat = torch.zeros((n_channels, img_dim, img_dim), dtype=torch.long).to(new_head.device)
        ids_mat[new_err_pos[:, 0], new_err_pos[:, 1], new_err_pos[:, 2]] = torch.arange(num_new_errs).to(new_head.device)
        mu_values = halfrange[nnz[:, 0], nnz[:, 1], nnz[:, 2], nnz[:, 3]]
        new_errs = torch.zeros((num_new_errs, batch_size, n_channels, img_dim, img_dim)).to(new_head.device, dtype=new_head.dtype)
        err_ids = ids_mat[nnz[:, 1], nnz[:, 2], nnz[:, 3]]
        new_errs[err_ids, nnz[:, 0], nnz[:, 1], nnz[:, 2], nnz[:, 3]] = mu_values
    return new_errs


class HybridZonotope:
    """
    Representation based on HybridZonotope from https://github.com/eth-sri/diffai/blob/master/ai.py
    Described in https://files.sri.inf.ethz.ch/website/papers/icml18-diffai.pdf (h_C + diag(h_B)*\beta + h_e*e)
    Here: head + diag(beta)*\beta + errors*e (batched!)
    """

    box_domains = ['box']
    #zono_domains = ['zono', 'zbox', 'zdiag', 'zswitch']
    #hybrid_domains = ['hzono', 'hbox', 'hdiag', 'hswitch']
    zono_domains = ['zono', 'zbox', 'zdiag', 'zdiag-c', 'zswitch', 'zono-ibp', 'cauchy-zono']
    hybrid_domains = ['hzono', 'hbox', 'hdiag', 'hdiag-c', 'hswitch', 'cauchy-hbox']

    supported_domains = box_domains + zono_domains + hybrid_domains

    def __init__(self, head, beta, errors, domain):
        self.head = head  # shape: [batch_size, *d]
        self.beta = beta  # shape: [batch_size, *d]
        self.errors = errors  # shape: [num_error_terms, batch_size, *d]
        if domain not in HybridZonotope.supported_domains:
            raise RuntimeError('Unsupported HybridZonotope domain: {}'.format(domain))
        self.domain = domain
        self.device = self.head.device
        if domain == 'box':
            assert self.errors is None
        elif domain in HybridZonotope.zono_domains:
            assert self.beta is None 
        assert not torch.isnan(self.head).any()
        assert self.beta is None or (not torch.isnan(self.beta).any())
        assert self.errors is None or (not torch.isnan(self.errors).any())

    @staticmethod
    def construct_from_noise(x, eps, domain, data_range):
        # Clamp to data_range
        x_center, x_beta = linf_clamped(x, eps, data_range[0], data_range[1])
        x_center, x_beta = x_center.to(dtype=torch.float32), x_beta.to(dtype=torch.float32)
        if domain == 'box':
            return HybridZonotope(x_center, x_beta, None, domain)
        elif domain in HybridZonotope.supported_domains:
            batch_size = x.size()[0]
            n_elements = x[0].numel()
            ei = torch.eye(n_elements).expand(batch_size, n_elements, n_elements).permute(1, 0, 2).to(x.device)
            if len(x.size()) > 2:
                ei = ei.contiguous().view(n_elements, *x.size())

            new_beta = None if domain in HybridZonotope.zono_domains else torch.zeros(x_beta.shape).to(device=x_beta.device, dtype=torch.float32)
            
            return HybridZonotope(x_center, new_beta, ei * x_beta.unsqueeze(0), domain)
        else:
            raise RuntimeError('Unsupported HybridZonotope domain: {}'.format(domain))

    def size(self):
        return self.head.size()

    def view(self, size):
        return HybridZonotope(self.head.view(*size),
                              None if self.beta is None else self.beta.view(size),
                              None if self.errors is None else self.errors.view(self.errors.size()[0], *size),
                              self.domain)

    def normalize(self, mean, sigma):
        return (self - mean) / sigma

    def __sub__(self, other):
        assert isinstance(other, torch.Tensor)
        return HybridZonotope(self.head - other, self.beta, self.errors, self.domain)

    def __add__(self, other):
        assert isinstance(other, torch.Tensor)
        return HybridZonotope(self.head + other, self.beta, self.errors, self.domain)

    def __truediv__(self, other):
        assert isinstance(other, torch.Tensor)
        return HybridZonotope(self.head / other,
                              None if self.beta is None else self.beta / abs(other),
                              None if self.errors is None else self.errors / other,
                              self.domain)

    def clone(self):
        return HybridZonotope(self.head.clone(),
                              None if self.beta is None else self.beta.clone(),
                              None if self.errors is None else self.errors.clone(),
                              self.domain)

    def detach(self):
        return HybridZonotope(self.head.detach(),
                              None if self.beta is None else self.beta.detach(),
                              None if self.errors is None else self.errors.detach(),
                              self.domain)

    # Conv2D transformer
    def conv2d(self, weight, bias, stride, padding, dilation, groups):
        new_head = F.conv2d(self.head, weight, bias, stride, padding, dilation, groups)
        
        if self.beta is not None:
            new_beta = F.conv2d(self.beta, weight.abs(), None, stride, padding, dilation, groups)
        else:
            new_beta = None 

        if self.errors is not None:
            # treat k * batch_size as the batch size, conv over all
            errors_resized = self.errors.view(-1, *self.errors.size()[2:])
            new_errors = F.conv2d(errors_resized, weight, None, stride, padding, dilation, groups)
            new_errors = new_errors.view(self.errors.size()[0], self.errors.size()[1], *new_errors.size()[1:])
        else:
            new_errors = None
            
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    # Linear layer transformer
    def linear(self, weight, bias, C=None):
        if C is not None:
            # Merge the logit diff with the last layer
            # C shape: [b, 9, d'=10] (9 times: true class - class i)
            weight = C.matmul(weight) 
            bias = C.matmul(bias)
            # weight: [b, 9, d], bias: [b, 9]
        else:
            bias = bias.unsqueeze(0)
            # weight: [d', d], bias: [1, d']
        return self.matmul(weight.transpose(-1, -2)) + bias

    def matmul(self, other):
        # print(self.head.isnan().any())
        # print(self.errors.isnan().any())
        # print(other.isnan().any())
        # head/beta: [b, d], errors: [k, b, d]
        if len(other.shape) == 3:
            # other: [b, d, 9], return [b, 9] and [k, b, 9]
            head = self.head.unsqueeze(-2).matmul(other).squeeze(-2)
            if self.beta is not None:
                beta = self.beta.unsqueeze(-2).matmul(other.abs()).squeeze(-2)
            else:
                beta = None
            if self.errors is not None:
                errors = self.errors.unsqueeze(-2).matmul(other).squeeze(-2)
            else:
                errors = None
        else:
            # other: [d, d'], return: [b, d'] and [k, b, d']
            head = self.head.matmul(other)
            if self.beta is not None:
                beta = self.beta.matmul(other.abs())
            else:
                beta = None
            if self.errors is not None:
                errors = self.errors.matmul(other)
            else:
                errors = None
            # print('-> ', head.isnan().any())
            # print('-> ', errors.isnan().any())
            
        return HybridZonotope(head, beta, errors, self.domain)

    # ReLU transformer
    def relu(self, bounds, soft_slope_gamma=None, zono_kappa=None, loosebox_round=None, loosebox_widen=None):
        # External (IBP or concretized) bounds=(lb, ub) need to be given
        assert bounds is not None 
        lb, ub = bounds
        
        if self.domain == 'box':
            min_relu, max_relu = F.relu(lb), F.relu(ub)

            mid = 0.5 * (max_relu + min_relu)
            halfrange = 0.5 * (max_relu - min_relu)
            return HybridZonotope(mid, halfrange, None, self.domain)
        elif self.domain == 'legacy-hbox':  # Keep this here for now
            # Split three cases
            is_under = (ub <= 0)
            is_above = (ub > 0) & (lb >= 0) 
            is_cross = (ub > 0) & (lb < 0)

            # Faster implementation:

            new_head = self.head.clone() 
            new_beta = self.beta.clone()
            new_errors = self.errors.clone() 

            ubhalf = ub/2

            new_head[is_under] = 0
            new_head[is_cross] = ubhalf[is_cross]

            new_beta[is_under] = 0
            new_beta[is_cross] = ubhalf[is_cross]

            new_errors[:, ~is_above] = 0

            return HybridZonotope(new_head, new_beta, new_errors, self.domain)
        else:
            # [zono domains]
            # Once we have lambda and d=-l*lambda
            # mu = d/2 for cross, 0 otherwise [exception: for zbox/hbox we have mu = u/2]
            # In all cases: (head * lambda + mu), (old_errs * lambda) + (new_errs = diag(mu)?)
            #
            # Under: (head * 0 + 0), (old_errs * 0)
            # Cross (head * lambda + mu), (old_errs * lambda) + (new_errs = diag(mu))
            # Above: (head * 1 + 0), (old_errs * 1)
            #
            # zono: lambda = u/(u-l), zdiag: lambda = 1, zbox: lambda = 0, zswitch: lambda = 0/1
            #
            # [hybrid domains]: instead of adding new errors aggregate in beta
            #
            # In all cases: (head * lambda + mu), (beta * lambda + mu), (old_errs * lambda)
            #
            # Under: (head * 0 + 0), (beta * 0 + 0), (old_errs * 0)
            # Cross (head * lambda + mu), (beta * lambda + mu), (old_errs * lambda)
            # Above: (head * 1 + 0), (beta * 1 + 0), (old_errs * 1)
            # 
            # hzono: lambda = u/(u-l), hdiag: lambda = 1, hbox: lambda = 0, hswitch: lambda = 0/1
            # 

            # Find cross cases and set opt slope
            is_cross = (lb < 0) & (ub > 0)
            D = 1e-6
            if self.domain in ['zono', 'hzono']:
                lambda_opt = ub/(ub-lb+D)
                if soft_slope_gamma is not None:
                    # soft zono, works only with zono 
                    cross_lambda = torch.sigmoid(soft_slope_gamma * (lb/(ub+D) - ub/(lb-D)))

                    cross_mu_small = (ub - ub * cross_lambda) * 0.5
                    cross_mu_big = (-lb * cross_lambda) * 0.5
                    cross_mu = torch.where(cross_lambda > lambda_opt, cross_mu_big, cross_mu_small)
                elif zono_kappa is not None:
                    # slanted zono, works only with zono
                    cross_lambda = ub/(ub-lb+D) * zono_kappa

                    cross_mu_small = (ub - ub * cross_lambda) * 0.5
                    cross_mu_big = (-lb * cross_lambda) * 0.5
                    cross_mu = torch.where(cross_lambda > lambda_opt, cross_mu_big, cross_mu_small)
                else:
                    cross_lambda = lambda_opt 
                    cross_mu = (-lb * lambda_opt) * 0.5
            elif self.domain in ['zdiag', 'hdiag', 'zdiag-c', 'hdiag-c']:
                cross_lambda = torch.ones(lb.size()).to(device=self.device, dtype=self.head.dtype)
                cross_mu = (-lb) * 0.5
            elif self.domain in ['zbox', 'hbox']:
                cross_lambda = torch.zeros(lb.size()).to(device=self.device, dtype=self.head.dtype)
                cross_mu = ub * 0.5
            elif self.domain in ['zswitch', 'hswitch']:
                # box if |lb| > |ub|, diag if |lb| <= |ub|
                cond = ((-lb) <= ub)
                cross_lambda = cond.to(device=self.device, dtype=self.head.dtype)
                cross_mu = torch.where(cond, (-lb) * 0.5, ub * 0.5)
            else:
                raise RuntimeError(f'Unknown zono/hybrid domain: {self.domain}')
            
            # Set 0/1 for other cases
            relu_lambda = torch.where(is_cross, cross_lambda, (lb >= 0).to(dtype=self.head.dtype))

            # Set mu to d/2 for cross cases
            zeros = torch.zeros(lb.size()).to(device=self.device, dtype=self.head.dtype)
            relu_mu = torch.where(is_cross, cross_mu, zeros)

            if self.domain in ['zdiag-c', 'hdiag-c']:
                # adjust negative case to make it continuous!
                below_lambda = torch.ones(lb.size()).to(device=self.device, dtype=self.head.dtype)
                below_mu = (-lb) * 0.5
                relu_lambda = torch.where((ub < 0), below_lambda, relu_lambda)
                relu_mu = torch.where((ub < 0), below_mu, relu_mu)

            if self.domain in HybridZonotope.zono_domains:
                # Head gets shifted by mu
                new_head = self.head * relu_lambda + relu_mu

                # Old errors get multiplied by the slope, new errors
                old_errs = self.errors * relu_lambda
                new_errs = get_new_errs(is_cross, new_head, relu_mu)

                if new_errs is not None:
                    new_errors = torch.cat([old_errs, new_errs], dim=0)
                else:
                    new_errors = old_errs

                return HybridZonotope(new_head, None, new_errors, self.domain)
            elif self.domain in HybridZonotope.hybrid_domains:
                # Head gets shifted by mu
                new_head = self.head * relu_lambda + relu_mu

                # Beta gets shifted by mu as well (no new error terms!)
                new_beta = self.beta * relu_lambda + relu_mu

                # Old errors get multiplied by the slope
                new_errors = self.errors * relu_lambda
                return HybridZonotope(new_head, new_beta, new_errors, self.domain)
            else:
                raise RuntimeError(f'Unknown zono/hybrid domain: {self.domain}')

    # ReLU6 transformer
    def relu6(self, deepz_lambda=None, use_all=False, it=0):
        lb, ub = self.concretize()
        if self.domain == 'box':
            min_relu, max_relu = F.relu6(lb), F.relu6(ub)
            return HybridZonotope(0.5 * (max_relu + min_relu), 0.5 * (max_relu - min_relu), None, self.domain)
        elif self.domain == 'zono':
            # Once we have slope=lambda and d1 and d2
            # center = (d1 + d2) / 2
            # halfrange = (d2 - d1) / 2
            # In all cases: (head * slope + center) + (old_errs * slope) + (new_errs = diag(halfrange))
            #
            # Far left: (head * 0 + 0) + (old_errs * 0)
            # Far right: (head * 0 + 6) + (old_errs * 0)
            # Only middle: (head * 1 + 0) + (old_errs * 1)
            #
            # Left cross (u <= 6): lambda = U / (U - L), d1 = 0, d2 = -L*lambda
            # Right cross (l >= 0): lambda = (6 - L) / (U - L), d1 = L-L*lambda, d2=6-6*lambda
            # Both cross (l < 0 and u > 6):
            #           lambda_L = 6 / (6 - L), d1_L = 0, d2_L = -lambda_L*L
            #           lambda_R = 6 / U, d1_R = 0, d2_R = 6 - lambda*6
            # lambda = min(lambda_L, lambda_R)

            # Find cross cases and set opt slope
            one_left = (ub <= 0)
            one_right = (lb >= 6)
            one_middle = (lb >= 0) & (ub <= 6)

            cross_left = (lb < 0) & ( (ub > 0) & (ub <= 6) )
            cross_right = ( (lb >= 0) & (lb < 6) ) & (ub > 6)
            cross_both = (lb < 0) & (ub > 6)

            D = 1e-6

            slope = torch.zeros(lb.size()).to(device=self.device, dtype=self.head.dtype)
            center = torch.zeros(lb.size()).to(device=self.device, dtype=self.head.dtype)
            halfrange = torch.zeros(lb.size()).to(device=self.device, dtype=self.head.dtype)

            slope[one_middle] = 1
            center[one_right] = 6

            L, U = lb, ub

            # Prep
            slope_l = U / (U - L + D)
            d2_l = -L * slope_l

            slope_r = (6 - L) / (U - L + D)
            d1_r = L - L*slope_r
            d2_r = 6 - 6*slope_r

            slope_both_l = 6 / (6 - L + D)
            d2_both_l = -slope_both_l * L

            slope_both_r = 6 / (U + D)
            d2_both_r = 6 - slope_both_r*6

            # Set
            slope[cross_left] = slope_l[cross_left]
            center[cross_left] = 0.5 * d2_l[cross_left]
            halfrange[cross_left] = 0.5 * d2_l[cross_left]

            slope[cross_right] = slope_r[cross_right]
            center[cross_right] = 0.5 * (d1_r[cross_right] + d2_r[cross_right])
            halfrange[cross_right] = 0.5 * (d2_r[cross_right] - d1_r[cross_right])

            l_smaller = slope_both_l < slope_both_r
            
            mask1 = cross_both & l_smaller
            slope[mask1] = slope_both_l[mask1]
            center[mask1] = 0.5 * d2_both_l[mask1]
            halfrange[mask1] = 0.5 * d2_both_l[mask1]

            mask2 = cross_both & (~l_smaller)
            slope[mask2] = slope_both_r[mask2]
            center[mask2] = 0.5 * d2_both_r[mask2]
            halfrange[mask2] = 0.5 * d2_both_r[mask2]

            # Apply
            # Head gets shifted by mu
            new_head = self.head * slope + center

            # Old errors get multiplied by the slope
            old_errs = self.errors * slope

            # New errors
            cross = cross_left | cross_right | cross_both

            new_errs = get_new_errs(cross, new_head, halfrange)

            if new_errs is not None:
                new_errors = torch.cat([old_errs, new_errs], dim=0)
            else:
                new_errors = old_errs

            return HybridZonotope(new_head, None, new_errors, self.domain)
        elif self.domain == 'hbox':  #creluBoxy
            # Split three cases
            is_0 = (ub <= 0)
            is_6 = (lb >= 6)
            is_inside = (lb >= 0) & (ub <= 6)
            is_rest = (~is_0) & (~is_6) & (~is_inside)
            # otherwise drop

            # Faster implementation only:

            new_head = self.head.clone() 
            new_beta = self.beta.clone()
            new_errors = self.errors.clone() 

            new_head[is_0] = 0
            new_beta[is_0] = 0

            new_head[is_6] = 6
            new_beta[is_6] = 0

            # For the inside, noop 
            ub[ub > 6] = 6
            lb[lb < 0] = 0

            mid = (ub + lb) * 0.5
            radius = (ub - lb) * 0.5
            new_head[is_rest] = mid[is_rest]
            new_beta[is_rest] = radius[is_rest]

            new_errors[:, ~is_inside] = 0

            return HybridZonotope(new_head, new_beta, new_errors, self.domain)
        else:
            raise RuntimeError('Error applying relu with unknown domain: {}'.format(self.domain))

    def concretize(self):
        delta = 0
        if self.beta is not None:
            delta = delta + self.beta
        if self.errors is not None:
            delta = delta + self.errors.abs().sum(0)
        return self.head - delta, self.head + delta

    # Average tightness of the zonotope, used for regression
    def avg_width(self):
        lb, ub = self.concretize()
        return (ub - lb).mean()

    # Does class i *always* have a greater logit value than class j?
    # delta = lower bound of logit_i - logit_j
    def is_greater(self, i, j):
        # Note: works only if d is a scalar (head/beta are of shape [batch_size, d])
        if self.errors is not None:
            diff_errors = (self.errors[:, :, i] - self.errors[:, :, j]).abs().sum(dim=0)
            diff_head = self.head[:, i] - self.head[:, j]
            delta = diff_head - diff_errors
            if self.beta is not None:
                delta -= self.beta[:, i].abs() + self.beta[:, j].abs()
            return delta, delta > 0
        else:
            diff_head = (self.head[:, i] - self.head[:, j])
            # TODO: why is this different than the previous branch? Shouldn't matter if beta>0
            diff_beta = (self.beta[:, i] + self.beta[:, j]).abs()
            delta = (diff_head - diff_beta)
            return delta, delta > 0

    # Returns a mask of verified examples in a batch
    def verify(self, targets):
        #assert False # deprecated since we use C
        n_class = self.head.size()[1]
        verified = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        verified_corr = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        for i in range(n_class):
            isg = torch.ones(targets.size(), dtype=torch.uint8).to(self.head.device)
            for j in range(n_class):
                if i != j:
                    _, ok = self.is_greater(i, j)
                    isg = isg & ok.byte()
            verified = verified | isg
            verified_corr = verified_corr.byte() | (targets.eq(i).byte() & isg)
        return verified, verified_corr


    # An attempt to generalize verification to regression
    def verify_regression(self, inputs, targets, eps_reg):
        # targets: [batch_size, 1]
        lb, ub = self.concretize()
        dists = torch.max((targets - lb).abs(), (ub - targets).abs()) 
        verified = (dists < eps_reg).int().to(self.head.device)
        return verified 

    # Returns minimum of logit[i] - logit[j]
    def get_min_diff(self, i, j):
        return self.is_greater(i, j)[0]


    # A "logit" vector created by using the upper bound for all classes 
    # and the lower bound for the target class
    def get_wc_logits(self, targets):
        # assert False  # deprecated since we use C
        batch_size = targets.size()[0]
        lb, ub = self.concretize()
        wc_logits = ub
        wc_logits[np.arange(batch_size), targets] = lb[np.arange(batch_size), targets]
        return wc_logits


    # Get logits assuming self bounds 9 (true_class - class_i) differences
    def get_logits(self, targets, nb_classes=10):
        assert(len(self.head.shape) == 2)
        assert(self.head.shape[1] == 9) # nb_classes-1
        batch_size = targets.shape[0]

        # Prepare [b, nb_classes-1] scatter idxs matrix
        idxs_base = torch.zeros((nb_classes, nb_classes - 1), dtype=torch.long, device=targets.device)
        for i in range(idxs_base.shape[0]):
            for j in range(idxs_base.shape[1]):
                idxs_base[i,j] = j if j < i else j+1
        idxs = idxs_base[targets, :]

        # Scatter
        lb, _ = self.concretize()
        dest = torch.zeros(batch_size, nb_classes, device=targets.device)
        logits = dest.scatter(1, idxs, -lb)
        return logits  # [b, nb_classes]


    def ce_loss(self, targets):
        # assert False  # deprecated since we have C
        wc_logits = self.get_wc_logits(targets)
        return F.cross_entropy(wc_logits, targets)


    def mse_loss(self, targets):
        lb, ub = self.concretize()
        furthest = lb

        mask = (ub - targets).abs() > (targets - lb).abs()

        furthest[mask] = ub[mask]
        return F.mse_loss(furthest, targets)

    
    def reg_loss(self, targets):
        lb, ub = self.concretize()
        return torch.mean(ub-lb)
