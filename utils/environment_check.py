import struct
import scipy
import sys
import torch
import torchvision


def environment_check(gpu_index):
  gpu_available = torch.cuda.is_available()
  device = torch.device('cuda:%i' % gpu_index if gpu_available else 'cpu')

  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.benchmark = True
  NUM_WORKERS = 4
  torch.backends.cudnn.enabled = True

  print('Python version: ' + sys.version)
  print('SciPy version: ' + str(scipy.__version__))
  print('PyTorch version: ', torch.__version__)
  print('torchvision version: ', torchvision.__version__)
  print('Bits: ' + str(struct.calcsize('P') * 8))  # 32 or 64 bits
  print('CUDA version: ', torch.version.cuda)
  print('Pytorch CUDA availability: ' + str(gpu_available))
  print('Device: ' + str(device))
  print('----------------------------------------------------------------')

  return gpu_available, device, NUM_WORKERS