import collections
import json
import os
from tabulate import tabulate
import time
from tqdm import tqdm

from provable_training.analyzer import forward_hybridzono, forward_deeppoly, get_diffs_C, forward_cauchy
from provable_training.hybridzono import HybridZonotope
from provable_training.utils import attack_pgd, Averager


"""
    Evaluates the net: encapsulates testing and verification, prints the result table
    and saves the results to a text file
"""

TQDM_BAR_FORMAT = '{desc}{percentage:3.0f}%|{bar:5}| {n_fmt}/{total_fmt}'

# Tests the net on given data
def test(args, net, test_loader, eps, do_pgd=True, epoch=None):
    acc = Averager()
    adv_acc = Averager()

    #return -1, -1
    assert args.setting == 'classification'
    # NOTE: reports only accuracy

    if not do_pgd:
        print('do_pgd set to false, skipping PGD testing (will report 0)')

    pbar = tqdm(test_loader, bar_format=TQDM_BAR_FORMAT)

    for inputs, targets in pbar:
        curr_examples = inputs.shape[0]
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outs = net(inputs)
        preds = outs.max(dim=1)[1]

        ok = targets.eq(preds)
        curr_acc = ok.float().mean().item()
        acc.add(curr_acc, curr_examples)

        # PGD attack
        if do_pgd:
            adv_inputs = attack_pgd(args, eps, net, inputs, targets)
            adv_outs = net(adv_inputs)
            curr_adv_acc = targets.eq(adv_outs.max(dim=1)[1]).float().mean().item()
            adv_acc.add(curr_adv_acc, curr_examples)

        # Print average batch loss, standard accuracy and adversarial accuracy
        epoch_str = f' epoch={epoch}' if epoch is not None else ''
        adv_acc_avg = adv_acc.avg() if do_pgd else -1
        pbar.set_description('[TEST{}] acc={:.3f}, adv_acc={:.3f}'.format(epoch_str, acc.avg(), adv_acc_avg))
    return acc.avg(), adv_acc_avg


# Runs all verification methods from args.verify_domains on the network
def verify(args, net, loader, eps):
    assert args.setting == 'classification'

    pbar = tqdm(loader, bar_format=TQDM_BAR_FORMAT)
    n_batches = 0
    ver_acc = {domain: Averager() for domain in args.verify_domains}
    distances = {domain: ([], []) for domain in args.verify_domains}
    tot_time = {domain: Averager() for domain in args.verify_domains}
        
    # Go through the data in the given dataloader
    for inputs, targets in pbar:
        curr_examples = inputs.shape[0]

        # Break if we are verifying only a subset of the examples
        test_idx = n_batches*args.batch_size
        if args.verify_max_tests is not None and test_idx >= args.verify_max_tests:
            # Make sure that shuffle=False for test_loader
            # (it shuffles the dataset again on each iteration!)
            break

        # Move to GPU
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # Try verification on this batch for all domains
        for domain in args.verify_domains:
            begtime = time.time()

            # Create an abstract object and propagate it
            if domain == 'deeppoly' or domain == 'crown-ibp' or domain == 'iclr2021':
                # DP family
                bounds = None
                if domain == 'crown-ibp' or domain == 'iclr2021':
                    box_inputs = HybridZonotope.construct_from_noise(inputs, eps, 'box', (args.input_min, args.input_max))
                    _, bounds = forward_hybridzono(net, box_inputs)
                abs_outs = forward_deeppoly(net, inputs, targets, eps, (args.input_min, args.input_max), bounds=bounds, is_iclr2021=(domain == 'iclr2021'), crown_variant=args.crown_variant, soft_slope_gamma=args.soft_slope_gamma)
                lb, _ = abs_outs.concretize()  # C merged, lower bound on (true-other)
                #lb[range(args.batch_size), targets] = 1e9
                verified = (lb > 0).all(dim=1)
                curr_ver_acc = verified.float().mean().item()
                ver_acc[domain].add(curr_ver_acc, curr_examples)
            elif domain in ['cauchy-zono', 'cauchy-hbox']:
                box_inputs = HybridZonotope.construct_from_noise(inputs, eps, 'box', (args.input_min, args.input_max))
                abs_outs = forward_cauchy(domain, net, box_inputs)
                curr_ver_acc = abs_outs.verify(targets)[1].float().mean().item()
                ver_acc[domain].add(curr_ver_acc, curr_examples)
            else:
                # HybridZonotope family
                if domain == 'zono-ibp':
                    box_inputs = HybridZonotope.construct_from_noise(inputs, eps, 'box', (args.input_min, args.input_max))

                    nb_classes = net.layers[-1].weight.shape[0]
                    C = None if (args.C == 'eye') else get_diffs_C(targets, nb_classes)
                    _, bounds = forward_hybridzono(net, box_inputs, C=C)
                    
                    abs_inputs = HybridZonotope.construct_from_noise(inputs, eps, 'zono', (args.input_min, args.input_max))
                else:
                    abs_inputs = HybridZonotope.construct_from_noise(inputs, eps, domain, (args.input_min, args.input_max))
                    bounds = None 

                if args.C == 'eye':
                    abs_outs, _ = forward_hybridzono(net, abs_inputs, C=None, soft_slope_gamma=args.soft_slope_gamma, zono_kappa=args.zono_kappa, loosebox_round=args.loosebox_round, loosebox_widen=args.loosebox_widen, dcs_per_one=args.dcs_per_one, bounds=bounds)
                    curr_ver_acc = abs_outs.verify(targets)[1].float().mean().item()
                    ver_acc[domain].add(curr_ver_acc, curr_examples)
                else:
                    nb_classes = net.layers[-1].weight.shape[0]
                    C = get_diffs_C(targets, nb_classes) 
                    abs_outs, _ = forward_hybridzono(net, abs_inputs, C=C, soft_slope_gamma=args.soft_slope_gamma, zono_kappa=args.zono_kappa, loosebox_round=args.loosebox_round, loosebox_widen=args.loosebox_widen, dcs_per_one=args.dcs_per_one, bounds=bounds)
                    lb, _ = abs_outs.concretize()
                    curr_ver_acc =  (lb > 0).all(dim=1).float().mean().item()
                    ver_acc[domain].add(curr_ver_acc, curr_examples)

            tot_time[domain].add(time.time() - begtime, curr_examples)
        n_batches += 1
        
        # Print accuracy and time per batch for each domain
        desc_domains = []
        for domain in args.verify_domains:
            desc_domains.append('{}(v={:.5f} t={:.2f}s)'.format(domain, ver_acc[domain].avg(), tot_time[domain].avg()))
        pbar.set_description('[VERIFY] ' + ', '.join(desc_domains))

    # Turn averagers into numbers
    for domain in args.verify_domains:
        ver_acc[domain] = ver_acc[domain].avg()
        # ver_acc[domain] = round(float(ver_acc[domain]), 5)
    return ver_acc


def evaluate(args, results_dir, net, model_name, data, loader, dataset_label):
    # Test and verify
    domains_str = ', '.join(args.verify_domains)
    print(f'Evaluating (test+verify) on the {dataset_label} with eps from {args.eps_test} and domains: {domains_str}')
    
    results = dict()
    results['model_name'] = model_name 
    for eps in args.eps_test:
        print(f'\n[Evaluating with eps={eps}]\n')
        acc, adv_acc = test(args, net, loader, eps, do_pgd=False)
        ver_accs = verify(args, net, loader, eps)

        cur_accs = collections.OrderedDict()
        cur_accs['standard'] = round(acc, 3)
        cur_accs['PGD'] = round(adv_acc, 3)
        for k, v in ver_accs.items():
            cur_accs[f'{k}'] = round(v, 3)

        cur_errs = collections.OrderedDict()
        for k, v in cur_accs.items():
            if v == -1:
                err_v = -1 
            else:
                err_v = round((1-v)*100, 2)
            cur_errs[f'{k}'] = err_v
        # Put in the results dict
        results[str(eps)] = {'accs':cur_accs, 'errs':cur_errs}

        # Print the tables
        print(f'\nAcc table (eps={eps}):\n')
        headers = ['ACCS'] + list(cur_accs.keys())
        table = [[model_name] + list(cur_accs.values())]
        print(tabulate(table, headers, tablefmt='github'))

        print(f'\nTest error table (eps={eps}):\n')
        headers = ['ERRORS'] + list(cur_errs.keys())
        table = [[model_name] + list(cur_errs.values())]
        print(tabulate(table, headers, tablefmt='github'))

    # Create a log dict and save it to file
    log = dict(vars(args))
    log['results'] = results 

    filename = os.path.join(results_dir, f'{model_name}.txt')
    with open(filename, 'w') as f:
        json.dump(log, f, sort_keys=True, indent=4)
    #print(log)
    print(f'\nWritten the results log to {filename}')

    # Neptune logging
    if args.neptune:
        print('Logging for neptune')
        import neptune

        # Log everything in case we need it (metric)
        # and also some special ones for the dashboard (text)
        idx = 1
        ver_idx = 1
        for eps, d in results.items():
            if eps == 'model_name':
                continue
            for k, v in d['accs'].items():
                neptune.log_metric(f'{eps}_acc_{k}', v)
                if k == 'standard':
                    if idx == 1:
                        neptune.log_text('acc', f'{v}')
                elif k == 'PGD':
                    suffix = '' if idx == 1 else f'{idx}'
                    neptune.log_text(f'pgd_acc{suffix}', f'{v} ({eps})')
                else:
                    suffix = '' if ver_idx == 1 else f'{ver_idx}'
                    neptune.log_text(f'ver_acc{suffix}', f'{v} ({k} {eps})')
                    ver_idx += 1
            idx += 1

        # Repeat for errors
        idx = 1
        ver_idx = 1
        for eps, d in results.items():
            if eps == 'model_name':
                continue
            for k, v in d['errs'].items():
                neptune.log_metric(f'{eps}_err_{k}', v)
                if k == 'standard':
                    if idx == 1:
                        neptune.log_text('err', f'{v}')
                elif k == 'PGD':
                    suffix = '' if idx == 1 else f'{idx}'
                    neptune.log_text(f'pgd_err{suffix}', f'{v} ({eps})')
                else:
                    suffix = '' if ver_idx == 1 else f'{ver_idx}'
                    neptune.log_text(f'ver_err{suffix}', f'{v} ({k} {eps})')
                    ver_idx += 1
            idx += 1
        print('Done logging for neptune')