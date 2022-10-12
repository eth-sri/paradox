import torch
import torch.nn.functional as F
import time

from provable_training.utils import get_loss_fn, Averager

"""
    Does *one epoch* of training on the net with given args
"""
    
def train(epoch, args, net, eps_sched, kappa_sched, beta_sched, lr_sched, train_loader, opt):
    nb_batches = len(train_loader) # last might be smaller

    begtime = time.time()

    # A hack used when restarting training from some epoch
    if epoch < args.skip:
        eps_sched.advance_time(nb_batches)
        kappa_sched.advance_time(nb_batches)
        beta_sched.advance_time(nb_batches)

        # step LR always for milestones/ibp and after mixing for rest
        if args.lr_milestones or args.ibp_scheduling or epoch >= args.warmup_epochs + args.mix_epochs:
            for _ in range(nb_batches):
                lr_sched.step()
        return []

    # Metrics: total loss + ok/adversarially ok/provably ok examples so far
    # The verified accuracy also gets saved in a csv
    loss, acc, adv_acc, ver_acc = Averager(), Averager(), Averager(), Averager()

    # Get the loss function
    loss_fn = get_loss_fn(args)

    for batch_idx, (inputs_batch, targets_batch) in enumerate(train_loader):
        # Get new values of eps, kappa and beta
        eps, kappa, beta = eps_sched.get(), kappa_sched.get(), beta_sched.get()
        
        # Move the current batch to GPU
        inputs_batch, targets_batch = inputs_batch.to(args.device), targets_batch.to(args.device)

        # Split one batch into several "pieces" so backpropagation is done once per batch
        # but less data is propagated in one forward pass (useful in memory-intensive cases)
        inputs_pieces = torch.split(inputs_batch, args.batch_size // args.batch_pieces, dim=0)
        assert(len(inputs_pieces) == args.batch_pieces)
        targets_pieces = torch.split(targets_batch, args.batch_size // args.batch_pieces, dim=0)
        assert(len(targets_pieces) == args.batch_pieces)

        # Zero grad once and propagate each piece
        opt.zero_grad()
        for inputs, targets in zip(inputs_pieces, targets_pieces):
            curr_examples = inputs.shape[0]

            # Based on the mode get loss, accuracy, and aux (adv/ver) accuracy
            # (All three are "averages")
            if epoch < args.warmup_epochs:
                kappa = 1 # override, needed if we do kappa 0->0 (but it should be 1 during warmup)
                
            curr_loss, curr_acc, curr_aux_acc = loss_fn(net, inputs, targets, eps=eps, kappa=kappa, beta=beta)

            # Sum up loss and standard accuracy
            loss.add(curr_loss, curr_examples)
            acc.add(curr_acc, curr_examples)

            # Log for neptune
            if args.neptune:
                import neptune
                neptune.log_metric('batch_loss', curr_loss)
                neptune.log_metric('batch_acc', curr_acc)
                if args.mode == 'train-provable':
                    neptune.log_metric('batch_ver_acc', curr_aux_acc)
                elif args.mode == 'train-pgd':
                    neptune.log_metric('batch_adv_acc', curr_aux_acc)

            # Sum up
            if args.mode == 'train-provable':
                ver_acc.add(curr_aux_acc, curr_examples)
            elif args.mode == 'train-pgd':
                adv_acc.add(curr_aux_acc, curr_examples)
            
            # Backpropagate the current piece but as loss was averaged we need to divide it by
            # batch_pieces to get the same values regardless of batch_pieces 
            # (when the loss grads accumulate)
            curr_loss /= args.batch_pieces

            curr_loss.backward()

        opt.step()

        # Print average loss, standard accuracy and additional accuracy
        batches_done = batch_idx+1
        if batches_done % 50 == 0 or batches_done == nb_batches:
            curr_lr = lr_sched.get_last_lr()[0]
            
            desc = '[{:d}:{:d}] (e={:.4f} k={:.4f} b={:.4f} | lr={:5}) loss={:.4f}, acc={:.3f}'.format(epoch, batch_idx+1, eps, kappa, beta, curr_lr, loss.avg(), acc.avg())

            if args.mode == 'train-provable' and args.setting == 'classification' and kappa < 1:
                # Ignore ver_acc if kappa=1 (we still just do natural training)
                desc += ', ver={:.3f}'.format(ver_acc.avg())
            elif args.mode == 'train-pgd':
                desc += ', adv={:.3f}'.format(adv_acc.avg())
            epoch_time = time.time() - begtime
            desc += f' (t={epoch_time:.2f}s)'
            print(desc)
        
        # Advance the timers at the end of the batch
        eps_sched.advance_time(1)
        kappa_sched.advance_time(1)
        beta_sched.advance_time(1)

        # step LR always for milestones/ibp and after mixing for rest
        if args.lr_milestones or args.ibp_scheduling or epoch >= args.warmup_epochs + args.mix_epochs:
            lr_sched.step()

    print()