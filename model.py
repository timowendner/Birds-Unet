import torch
from torch import nn
from torch import Tensor


def dual(in_channel, out_channel, kernel=9):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.Dropout1d(p=0.2),
        nn.ReLU(),
        nn.Conv1d(out_channel, out_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


def up(in_channel, out_channel, scale=2, kernel=9, pad=0):
    return nn.Sequential(
        nn.Conv1d(in_channel, in_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.Dropout1d(p=0.2),
        nn.ReLU(),
        nn.Conv1d(in_channel, out_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.ReLU(),
        nn.ConvTranspose1d(out_channel, out_channel,
                           kernel_size=scale, stride=scale, output_padding=pad),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

        # define the pooling layer
        length = config.data_length
        scale = config.model_scale
        kernel = config.model_kernel
        self.pool = nn.MaxPool1d(kernel_size=scale, stride=scale)

        # define the encoder
        last = config.model_in
        pad = []
        self.length = [length]
        self.down = nn.ModuleList([])
        for channel in config.model_layers:
            cur_pad, length = length % scale, length // scale
            self.length.append(length)
            pad.append(cur_pad)
            layer = dual(last, channel, kernel=kernel)
            self.down.append(layer)
            last = channel

        # define the decoder
        self.up = nn.ModuleList([])
        for channel in reversed(config.model_layers):
            layer = up(last, channel, scale=scale,
                       kernel=kernel, pad=pad.pop())
            self.up.append(layer)
            last = channel * 2

        # define the output layer
        output = nn.ModuleList([])
        for channel in config.model_layers_out:
            conv = nn.Conv1d(
                last, channel, kernel_size=kernel, padding=kernel//2)
            output.append(conv)
            output.append(nn.ReLU())
            last = channel
        output.append(
            nn.Conv1d(last, config.model_out, kernel_size=kernel, padding=kernel//2))
        self.output = nn.Sequential(*output)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: Tensor) -> Tensor:
        # apply the encoder
        encoder = []
        for layer in self.down:
            x = layer(x)
            encoder.append(x)
            x = self.pool(x)

        # apply the decoder
        for layer in self.up:
            x = layer(x)
            x = torch.cat([encoder.pop(), x], 1)

        # apply the output
        x = self.output(x)
        print(x.shape)
        x = self.softmax(x)
        return x
