import torch
import torch.nn.functional as F
import sys

def custom_loss(net_out, target, nickname):

  if nickname == 'MSE':
    loss = F.mse_loss(net_out, target)
  elif nickname == 'MAE':
    loss = F.l1_loss(net_out, target)
  else:
    raise Exception('Unknown loss: ' + nickname)

  return loss
