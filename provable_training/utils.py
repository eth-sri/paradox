import torch 
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
from sklearn import decomposition
from provable_training.hybridzono import HybridZonotope
from provable_training.analyzer import get_diffs_C, get_eye_C, forward_hybridzono, forward_deeppoly, forward_cauchy

"""
    A set of utilities needed for provable training.
"""

# Builds a dataset object based on the dataset name
def get_dataset(do_transform, setting, eps, dataset, nb_train=None):
    if dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()) 
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        input_shape = torch.LongTensor([1, 28, 28])
        data_min = 0
        data_max = 1
    elif dataset == 'fashion-mnist':
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        input_shape = torch.LongTensor([1, 28, 28])
        data_min = 0
        data_max = 1
    elif dataset == 'cifar10':
        if do_transform:
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
            transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor()
            ]))
        else:
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
            transform=transforms.Compose([
                        transforms.ToTensor()
            ]))


        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

        input_shape = torch.LongTensor([3, 32, 32])
        data_min = 0
        data_max = 1
    elif dataset == 'svhn':
        if do_transform:
            train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, 
            transform=transforms.Compose([
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor()
            ]))
        else:
            train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, 
            transform=transforms.Compose([
                        transforms.ToTensor()
            ]))

        test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
        input_shape = torch.LongTensor([3, 32, 32])
        data_min = 0
        data_max = 1
    else:
        raise RuntimeError(f'Unsupported dataset: {dataset}')
    return train_set, test_set, input_shape, (data_min, data_max)


# Creates loaders from the specified dataset
def get_loaders(train_set, train_bs, test_set, test_bs, input_shape, shuffle=True):

    # note: drop_last used to be true
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bs, shuffle=shuffle, num_workers=4, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=4, drop_last=False)

    return len(train_set), train_loader, len(test_set), test_loader, input_shape


# Computes the sum of L1 norms of all weight params in the network (usedfor regularization)
def get_param_norm(net, norm_type):
    ret = 0
    for param_name, param_value in net.named_parameters():
        if 'weight' in param_name:
            if norm_type == 'L1':
                ret += param_value.abs().sum()
            elif norm_type == 'L2':
                ret += param_value.pow(2).sum()
            else:
                pass
    return ret

# Keeps track of the running average of inputed values
# add(acc, cnt) = "accuracy of acc on a batch of cnt examples"
# solves the issue of unequal batches (128 ... 128 80)
class Averager:
    def __init__(self):
        self.sum = 0
        self.cnt = 0
    
    def add(self, val, cnt):
        self.sum += val * cnt 
        self.cnt += cnt 
    
    def avg(self):
        return self.sum / self.cnt

# Schedules a hyperparam from start to end in n_steps (after waiting for warmup steps)
class Scheduler:
    def __init__(self, val_start, val_end, mix_steps, warmup_steps):
        self.val_start = val_start
        self.val_end = val_end 
        self.mix_steps = mix_steps 
        self.warmup_steps = warmup_steps 
        self.curr_steps = 0 

    def advance_time(self, steps_inc):
        self.curr_steps += steps_inc

    def get(self):
        if self.curr_steps < self.warmup_steps:
            return self.val_start
        elif self.curr_steps >= self.warmup_steps + self.mix_steps:
            return self.val_end 
        else:
            progress = (self.curr_steps - self.warmup_steps) / float(self.mix_steps)
        return self.val_start + progress * (self.val_end - self.val_start)


# Runs (untargeted) PGD on a batch of inputs to get adversarial inputs
# Uses args.input_min and args.input_max
def attack_pgd(args, eps, net, inputs, targets):
    delta = torch.FloatTensor(inputs.size()).to(args.device).uniform_(-eps, eps).requires_grad_(True)
    for _ in range(args.pgd_steps):
        net.zero_grad()
        adv_inputs = inputs + delta
        adv_outs = net(adv_inputs)
        adv_loss = F.cross_entropy(adv_outs, targets) if args.setting == 'classification' else F.mse_loss(adv_outs.flatten(), targets)
        adv_loss.backward()
        delta.data = torch.clamp(delta.data + args.pgd_step_size * delta.grad.sign(), -eps, eps)
        delta.data = torch.max(delta.data, args.input_min-inputs)
        delta.data = torch.min(delta.data, args.input_max-inputs)
        delta.grad.zero_()
    return (inputs + delta).detach()

# Constructs and returns the loss function for given params
# loss_fn: (net, inputs, targets, eps, kappa, beta) -> (real-valued loss, acc, aux_acc)
# eps/kappa/beta required but ignored for natural training
# (acc represents the average standard accuracy for this batch)
# (aux_acc represents the average adversarial/provable accuracy for this batch)
def get_loss_fn(args):

    # Define the function and capture all needed args
    def loss_fn(net, inputs, targets, eps, kappa, beta, args=args):
        assert args.setting == 'classification'
        nb_classes = net.layers[-1].weight.shape[0]

        # Propagate
        outs = net(inputs)

        # Prepare natural and regularization loss 
        nat_loss = F.cross_entropy(outs, targets)
        if args.reg is None:
            reg_loss = 0
        else:
            reg_loss = args.reg_lambda * get_param_norm(net, args.reg)

        # Prepare C 
        if args.C == 'eye':
            C = None
        else:
            C = get_diffs_C(targets, nb_classes)

        aux_acc = None
        if args.mode == 'train-provable':
            # If kappa=1 just return regular loss (aux_acc will not be used)
            if kappa == 1:
                loss = nat_loss + reg_loss
                aux_acc = 0  # Ignored
            else:
                # kappa<1, we need both the natural and the abstract loss (no more branching on kappa)

                # If we need bounds and/or beta<1, we need to propagate a box (once!)
                if args.train_domain in ['zono-ibp', 'crown-ibp', 'iclr2021', 'box'] or beta < 1:
                    box_inputs = HybridZonotope.construct_from_noise(inputs, eps, 'box', (args.input_min, args.input_max))
                    box_outs, box_bounds = forward_hybridzono(net, box_inputs, C=C, loosebox_round=args.loosebox_round, loosebox_widen=args.loosebox_widen, dcs_per_one=args.dcs_per_one)
                
                # If beta>0 we need to propagate the abstract element
                if beta > 0:
                    if args.train_domain == 'box':
                        abs_outs = box_outs 
                    elif args.train_domain in ['zono', 'zbox', 'zdiag', 'zdiag-c', 'zswitch', 'hzono', 'hbox', 'hdiag', 'hdiag-c', 'hswitch']:
                        abs_inputs = HybridZonotope.construct_from_noise(inputs, eps, args.train_domain, (args.input_min, args.input_max))
                        abs_outs, _ = forward_hybridzono(net, abs_inputs, C=C, soft_slope_gamma=args.soft_slope_gamma, zono_kappa=args.zono_kappa)
                    elif args.train_domain == 'zono-ibp':
                        abs_inputs = HybridZonotope.construct_from_noise(inputs, eps, 'zono', (args.input_min, args.input_max))
                        abs_outs, _ = forward_hybridzono(net, abs_inputs, bounds=box_bounds, C=C)
                    elif args.train_domain in ['cauchy-zono', 'cauchy-hbox']:
                        box_inputs = HybridZonotope.construct_from_noise(inputs, eps, 'box', (args.input_min, args.input_max))
                        abs_outs = forward_cauchy(args.train_domain, net, box_inputs)

                    elif args.train_domain in ['deeppoly']:
                        # Note: deeppoly gets only the lower bound, as C is merged
                        abs_outs = forward_deeppoly(net, inputs, targets, eps, (args.input_min, args.input_max), crown_variant=args.crown_variant, soft_slope_gamma=args.soft_slope_gamma)
                    elif args.train_domain == 'crown-ibp':
                        abs_outs = forward_deeppoly(net, inputs, targets, eps, (args.input_min, args.input_max), bounds=box_bounds, soft_slope_gamma=args.soft_slope_gamma) 
                    elif args.train_domain == 'iclr2021':
                        abs_outs = forward_deeppoly(net, inputs, targets, eps, (args.input_min, args.input_max), bounds=box_bounds, is_iclr2021=True) 

                # Finally, build the abstract loss and choose the verifier
                # NOTE: when scheduling beta towards 0, the verifier will suddenly become box
                if beta == 0:
                    assert args.C == 'diffs'  # crown-ibp always uses diffs
                    abs_loss = F.cross_entropy(box_outs.get_logits(targets, nb_classes), targets)
                    verifier = box_outs
                elif beta == 1:
                    if args.C == 'eye':
                        abs_loss = abs_outs.ce_loss(targets)
                    else:
                        abs_loss = F.cross_entropy(abs_outs.get_logits(targets, nb_classes), targets)

                    verifier = abs_outs
                else:
                    # CROWN-IBP implementation: combine the logits instead of combining losses
                    assert args.C == 'diffs'  # crown-ibp always uses diffs
                    logits = (1 - beta) * box_outs.get_logits(targets, nb_classes) + beta * abs_outs.get_logits(targets, nb_classes)
                    abs_loss = F.cross_entropy(logits, targets)
                    verifier = abs_outs  # If combining

                # Build the full loss: natural + abstract + regularization
                # (!!!) Flipped the meaning of kappa to be consistent with CROWN-IBP
                loss = (kappa) * nat_loss + (1 - kappa) * abs_loss + reg_loss

                if args.C == 'eye':
                    aux_acc = verifier.verify(targets)[1].float().mean().item() / args.batch_pieces
                else:
                    # The verifier must include C, so we care only about the lower bound
                    verifier_lb, _ = verifier.concretize()
                    aux_acc = (verifier_lb >= 0).all(dim=1).float().mean().item() / args.batch_pieces

                ##### Debug: print values of losses
                # print(f'Loss:{loss}')
                # print(f'ver_acc:{aux_acc}')
        elif args.mode == 'train-pgd':
            adv_inputs = attack_pgd(args, eps, net, inputs, targets)
            adv_outs = net(adv_inputs)
            loss = F.cross_entropy(adv_outs, targets) + reg_loss
            aux_acc = targets.eq(adv_outs.max(dim=1)[1]).float().mean().item() / args.batch_pieces
        elif args.mode == 'train-natural':
            loss = F.cross_entropy(outs, targets) + reg_loss
        else:
            raise RuntimeError(f'Unknown mode for loss fn: {args.mode}')

        acc = targets.eq(outs.max(dim=1)[1]).float().mean() / args.batch_pieces
        return loss, acc, aux_acc
    
    # Return it finally
    return loss_fn
