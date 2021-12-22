import torch


def get_scheduler(optimizer, e):
    if e.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, e.scheduler_step_size, e.scheduler_gamma)
    elif e.scheduler == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - (e.num_epochs - e.scheduler_num_epoch_linear_decrease)) \
                   / float(e.scheduler_num_epoch_linear_decrease + 1)
            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        raise Exception('Unknown scheduler: ' + e.scheduler)


    return scheduler
