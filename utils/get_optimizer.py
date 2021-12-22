from torch import optim


def get_optimizer(network_parameters, e):
    if e.optimizer == 'sgd':
        optimizer = optim.SGD(network_parameters, lr=e.learning_rate, momentum=e.momentum)
    elif e.optimizer == 'adam':
        optimizer = optim.Adam(network_parameters, lr=e.learning_rate, betas=(0.5, 0.999))
    else:
        raise Exception('Unknown optimizer: ' + e.optimizer)

    return optimizer

