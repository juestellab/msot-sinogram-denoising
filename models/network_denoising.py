import torch.nn as nn
from .networks import Unet
import torch


class DenoisingNet(nn.Module):

    def __init__(self, e):
        super(DenoisingNet, self).__init__()

        self.include_timestps_channel = e.include_timestps_channel
        if self.include_timestps_channel:
            input_nc = 2
        else:
            input_nc = 1

        output_nc = 1
        self.model = Unet(input_nc, output_nc, ngf=e.ngf_unet, n_layers=e.num_layers_unet, use_bias=e.use_bias)

        if e.gain_for_normal_weight_init > 0:
            def init_weights(m):
                if hasattr(m, 'weight'):
                    nn.init.normal_(m.weight.data, 0.0, e.gain_for_normal_weight_init)
            self.model.apply(init_weights)

    def forward(self, x):

        if self.include_timestps_channel:
            timestps_channel = torch.div(torch.arange(start=0, end=x.shape[2], step=1.0, dtype=x.dtype, device=x.device), x.shape[2]).reshape(1, 1, -1, 1)
            x = torch.cat([x, timestps_channel.repeat(x.shape[0], 1, 1, x.shape[3])], 1)
        return self.model(x)
