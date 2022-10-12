import datetime
import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

from args_factory import get_args
from provable_training.eval import evaluate, test
from provable_training.networks import create_net
from provable_training.train import train
from provable_training.utils import get_dataset, get_loaders, Scheduler
from provable_training.manual_init import manual_init


"""
    Entry point for certified (provable) training: loads the args, and invokes the training/eval
"""

def main():
    args = get_args()
    print(f'\nStarted at: {datetime.datetime.now()}\n')

    # Set the label and sort domains
    if args.label is None:
        # If no label provided, use the timestamp
        args.label = str(int(time.time()))
    args.verify_domains.sort()

    # Experimenting, not a real run, automatically disables neptune
    if args.label == 'exp':
        args.neptune = None

    # Load the data
    torch.manual_seed(args.seed)
    train_set, test_set, input_shape, data_range = get_dataset(args.do_transform, args.setting, args.eps_train, args.dataset)
    args.input_min = data_range[0]
    args.input_max = data_range[1]

    # Cut the dataset 
    if args.percent < 100:
        indices = np.arange(len(train_set))
        N = round((args.percent / 100.0) * len(train_set))
        print(f'Plan to use {N}/60000 stratified train examples!')
        idx1, _ = train_test_split(indices, train_size=N, stratify=train_set.targets)
        train_set = Subset(train_set, idx1)
        print(f'Actually using {len(train_set)}/60000 examples, class counts:')
        print(train_set.dataset.targets[train_set.indices].unique(return_counts=True)[1])
    else:
        print(f'Using full dataset!')
    # Create loaders and the net
    num_train, train_loader, num_test, test_loader, input_shape = get_loaders(train_set, args.batch_size, test_set, args.batch_size, input_shape)

    if args.load_model is not None:
        while os.system(f'stat {args.load_model}') != 0:
            print('cant find the model, trying again in 1 minute')
            time.sleep(60)

    net = create_net(args.net, input_shape, args.device, args.load_model, args.relu6, not args.unnormalized, dataset=args.dataset)
    
    # Build the mode string, model name, and set up the results directory
    if args.mode == 'eval':
        model_name = f'{args.net}_{args.label}_eval'
    else:
        mode_str = args.mode 
        if args.mode == 'train-provable':
            mode_str += f'-{args.train_domain}'
            if args.train_domain == 'crown-ibp':
                # Note: remember to change this if using fractional betas
                mode_str += f'-{int(args.beta1)}to{int(args.beta2)}'
            mode_str += f'-{args.eps_train}'
        model_name = f'{args.net}_{mode_str}'
    args.model_name = model_name
    
    results_dir = os.path.join(args.main_dir, 'results/provable_training', args.label)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Check required args for train, but don't make them required always as that complicates eval
    if args.mode == 'train-provable':
        assert args.eps_train is not None
        assert args.eps_test is not None
        assert args.C is not None
    
    # No more args will be added, init neptune here
    if args.neptune:
        import neptune
        neptune.init(api_token=os.getenv('NEPTUNE_API_KEY'), project_qualified_name=args.neptune)
        neptune.create_experiment(args.label, params=vars(args))

    # Set inits 
    manual_init(net, args.init_method)
    # Print args
    print('All args:')
    for k, v in sorted(list(vars(args).items())):
        print(f'{k:15} = {v}')
    print()

    # Print the total number of params of the net
    print('Params:')
    num_params = 0
    for parameter in net.named_parameters():
        print(parameter[0], parameter[1].shape)
        num_params += parameter[1].numel()
    print(f'Total num params: {num_params}\n')

    # Branch on modes
    if args.mode == 'eval':
        print('Starting the evaluation (test set)')
        #evaluate(args, results_dir, net, model_name, train_set, train_loader, 'train set')
        evaluate(args, results_dir, net, model_name, test_set, test_loader, 'test set')
    elif args.mode in ['train-natural', 'train-pgd', 'train-provable']:
        # Prepare the directory where the snapshots will be saved
        model_dir = os.path.join(args.main_dir, 'saved_models/', args.label)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Print the description
        desc = f'Training with mode {args.mode}'
        if args.mode == 'train-provable':
            if args.train_domain is None:
                raise RuntimeError('Train domain must be set for provable training')
            desc += f' and domain {args.train_domain}'
        print(desc)
        print()

        is_tweak = 'tweak' in args.label

        # Set up the optimizer
        if args.opt == 'adam':
            opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
        elif args.opt == 'sgd':
            opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise RuntimeError(f'Unknown optimizer: {args.opt}')

        # Pick one LR scheduler 
        nb_batches = len(train_loader)
        # (StepLR multiplies by gamma every step_size epochs)
        # (MultiStepLR multiplies by gamma when a milestone epoch is reached)
        if args.lr_milestones is not None:
            # MNIST/FashionMNIST: 130,190
            # SVHN: 0.6,0.9
            # CIFAR-10: 2600,3040
            ms = [float(m) for m in args.lr_milestones.split(',')]
            ms = [nb_batches*int(m) if m>=1 else nb_batches*int(m*args.n_epochs) for m in ms]
            lr_sched = optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.1)
        elif args.ibp_scheduling:
            # Always step
            lr_sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[15000, 25000], gamma=0.1)
        else:
            # Step only after mixing is done!
            step = nb_batches * (10 if args.tenhalf else 20)
            lr_sched = optim.lr_scheduler.StepLR(opt, step_size=step, gamma=0.5)

        # Set up eps/kappa/beta schedulers
        # The args.n_epochs of (provable) training are divided in 3 sections:
        # (1) warm up for args.warmup_epochs: regular training, e=0, k=1, b=beta1
        # (2) ramp up (mix) for args.mix_epochs: e -> args.eps_train, k -> 0, b -> beta2
        # (3) final params training: e=args.eps_train, k=0, b=beta2
        # (note: Scheduler changed to conform to this)
        # All count in steps!
        if args.ibp_scheduling:
            mix_steps = 10000
            warmup_steps = 2000
        else:
            mix_steps = nb_batches * args.mix_epochs
            warmup_steps = nb_batches * args.warmup_epochs

        eps_sched = Scheduler(args.eps1, args.eps_train, mix_steps, warmup_steps)
        kappa_sched = Scheduler(args.kappa1, args.kappa2, mix_steps, warmup_steps) # default: 1->0
        beta_sched = Scheduler(args.beta1, args.beta2, mix_steps, warmup_steps) # default: 1->1

        # Save a snapshot before start! 
        model_path = os.path.join(model_dir, f'{model_name}_epoch-1.pt') 
        print(f'Saving model {model_path}\n')
        torch.save(net.state_dict(), model_path)

        test_every = 500 if not is_tweak else 20 # ~Code state
        # Main training loop
        for epoch in range(args.n_epochs):
            # Do one epoch of training
            train(epoch, args, net, eps_sched, kappa_sched, beta_sched, lr_sched, train_loader, opt)

            # One more step (~code state)
            if (args.lr_milestones or epoch >= args.warmup_epochs + args.mix_epochs) and not is_tweak:
                lr_sched.step()

            # Save every 20 epochs and after the last epoch
            if (epoch+1) % 20 == 0 or (epoch+1) == args.n_epochs:
                model_path = os.path.join(model_dir, f'{model_name}_epoch{epoch}.pt')
                print(f'Saving model {model_path}\n')
                torch.save(net.state_dict(), model_path)
                if args.neptune and (epoch+1) == args.n_epochs:
                    neptune.log_artifact(model_path)

                # Test
                if (epoch + 1) % test_every == 0:
                    for eps in args.eps_test:
                        test(args, net, test_loader, eps, do_pgd=True, epoch=epoch) 
        
        # Done with training, evaluate
        print('Evaluating after training (test set)\n')
        #evaluate(args, results_dir, net, model_name, train_set, train_loader, 'train')
        evaluate(args, results_dir, net, model_name, test_set, test_loader, 'test')
    elif args.mode == 'print_onnx':
        print('Printing onnx network to:', args.onnx_file)
        dummy_input = torch.randn((1, 1, 28, 28), device='cuda')
        net.skip_norm = True
        torch.onnx.export(net, dummy_input, args.onnx_file, verbose=True)
    else:
        raise RuntimeError(f'Unknown mode: {args.mode}')

    print(f'Finished at: {datetime.datetime.now()}')


if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False, precision=4)
    main()

