import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train a neural network with augmentation')

    domains_box = ['box']
    domains_zono = ['zono', 'zbox', 'zdiag', 'zdiag-c', 'zswitch', 'zono-ibp', 'cauchy-zono']
    domains_hybrid = ['hzono', 'hbox', 'hdiag', 'hdiag-c', 'hswitch', 'cauchy-hbox']
    domains_dp = ['deeppoly', 'crown-ibp', 'iclr2021']
    domains_lp = ['triangle']
    domains = domains_box + domains_zono + domains_hybrid + domains_dp + domains_lp

    # Soft deeppoly/crown-ibp slope: sigmoid(gamma * (l/u - u/l)), try [1, 10, 100, 1000]
    parser.add_argument('--soft_slope_gamma', default=None, type=float, help='the steepness of the sigmoid curve')
    parser.add_argument('--zono_kappa', default=None, type=float, help='the steepness of the zono')
    
    parser.add_argument('--loosebox_widen', default=None, type=float, help='[lb, ub] -> [lb-1, ub+1]')
    parser.add_argument('--loosebox_round', default=None, type=float, help='[lb, ub] -> [floor(lb), ceil(ub)]')	
    parser.add_argument('--dcs_per_one', default=1, type=float, help='discontinuities per 1 of loosebox_round (default=1)')

    parser.add_argument('--crown_variant', type=str, default=None, choices=['0','1', '0-c', '1-c', 'lower-tria', 'upper-tria', 'lower-tria-c', 'upper-tria-c'], help='optimizer')
    # Global params
    parser.add_argument('--percent', default=100, type=int, help='percentage of train dataset to use')
    parser.add_argument('--init_method', default='default', type=str, choices=['ibp', 'kaiming-normal', 'xavier-normal', 'orthogonal', 'default'], help='initialization type')
    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune')
    parser.add_argument('--label', required=True, type=str, help='label used for this run - subdirs are named like this')
    parser.add_argument('--seed', default=100, type=int, help='seed for the random number generator')
    parser.add_argument('--device', type=str, default='cuda', help='the device to use (cuda/cpu)')
    parser.add_argument('--mode', type=str, required=True, help='mode of training/testing (eval, train-provable, train-natural, train-pgd)')
    parser.add_argument('--setting', type=str, default='classification', choices=['classification', 'regression'], help='current learning setting, use only classification')
    parser.add_argument('--dataset', required=True, help='the name of the dataset to use')
    parser.add_argument('--net', type=str, required=True, help='neural network architecture name')

    # Path params
    parser.add_argument('--main_dir', type=str, default='.', help='directory where the models should be stored')
    parser.add_argument('--load_model', default=None, type=str, help='the path to the model snapshot to load')
    parser.add_argument('--onnx_file', default=None, type=str, help='path where to save the network in onnx format')

    # High-level training params
    parser.add_argument('--opt', type=str, default='adam', choices=['adam','sgd'], help='the optimizer')
    parser.add_argument('--train_domain', default=None, type=str, choices=domains, help='the domain (relaxation) to use in provable training')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size for training and testing')
    parser.add_argument('--batch_pieces', default=1, type=int, help='in how many pieces to split each batch (approximately)')
    parser.add_argument('--eps_train', type=float, help='the epsilon noise used in training')
    parser.add_argument('--eps_test', nargs='+', type=float, default=[], help='which epsilon(s) to use for evaluation')
    parser.add_argument('--reg', default=None, type=str, choices=['L1', 'L2'], help='regularization type')
    parser.add_argument('--reg_lambda', default=0, type=float, help='L1/L2 regularization weight')
    parser.add_argument('--relu6', default=False, action='store_true', help='if relu6 should be used instead of relu')
    parser.add_argument('--do_transform', default=False, action='store_true', help='if the (cifar/SVHN) dataset should be augmented')
    parser.add_argument('--unnormalized', default=False, action='store_true', help='do not normalize the data')
    parser.add_argument('--C', type=str, choices=['eye', 'diffs'], help='whether to merge logit diffs with the last layer or not (elision)')

    # Training schedule / mixing params
    # [warmup (natural training) ---> rampup/mix (eps, kappa, beta) ---> final param training]
    # eps (l_inf radius): from 0 to args.eps 
    # kappa (natural vs provable tradeoff): from 1 (natural) to 0 (provable)
    # beta (complex relaxation vs IBP tradeoff, beta=1 means no IBP): from args.beta1 to args.beta2 
    parser.add_argument('--n_epochs', default=200, type=int, help='the number of epochs')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='the duration of warmup (pure natural training)')
    parser.add_argument('--mix_epochs', default=50, type=int, help='the duration of mixing (eps/kappa/beta scheduling)')
    parser.add_argument('--lr', default=0.0005, type=float, help='the initial learning rate')
    
    # 4 scheduling choices: "steps" (no extra argument, default StepLR), lr_milestones (MultiStepRL), ibp_scheduling (StepLR with IBP params), tenhalf (halve LR every 10 epochs)
    parser.add_argument('--lr_milestones', default=None, type=str, help='use the provided comma-separated milestones (MultiStepLR) instead of StepLR, if <1 treated as percentage of epochs')
    parser.add_argument('--ibp_scheduling', default=False, action='store_true', help='use IBP scheduling')
    parser.add_argument('--tenhalf', default=False, action='store_true', help='use LR scheduling that halves LR every 10 epochs')

    # Beta mixing ((1-beta)*box + beta*model), defaults to 1->1 (equivalent to not using beta at all)
    parser.add_argument('--beta1', type=float, default=1, help='initial value in beta mixing')
    parser.add_argument('--beta2', type=float, default=1, help='final value in beta mixing')
    parser.add_argument('--kappa1', type=float, default=1, help='initial value in kappa mixing')
    parser.add_argument('--kappa2', type=float, default=0, help='final value in kappa mixing')
    parser.add_argument('--eps1', type=float, default=0, help='initial value in eps mixing')

    # PGD params
    parser.add_argument('--pgd_steps', default=100, type=int, help='number of pgd steps')
    parser.add_argument('--pgd_step_size', default=0.01, type=float, help='step size for pgd')

    # Verification params
    parser.add_argument('--verify_domains', required=True, nargs='+', type=str, choices=domains, help='what domains to use to verify')
    parser.add_argument('--verify_max_tests', type=int, default=None, help='the maximum number of tests used in verification')
    parser.add_argument('--skip', type=int, default=-1, help='the number of epochs to skip (0 = skip first epoch)')

    # Parse and run
    return parser.parse_args()
    

