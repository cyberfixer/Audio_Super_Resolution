from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import CONFIG
from dataset import CustomDataset
from torch.utils.data import DataLoader
from performer_pytorch import Performer
# TODO: rename the class of the model. if you change it remember to change the import in main.py


class SSAR(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO:need to add layers

    def forward(self, x):
        pass
        # TODO:need define forward


class TUNet(nn.Module):
    def __init__(self):
        super(TUNet, self).__init__()
        self.encoder = Encoder(max_len=CONFIG.DATA.window_size,
                               kernel_sizes=CONFIG.MODEL.kernel_sizes,
                               strides=CONFIG.MODEL.strides,
                               out_channels=CONFIG.MODEL.out_channels,
                               tfilm=CONFIG.MODEL.tfilm,
                               n_blocks=CONFIG.MODEL.n_blocks)
        bottleneck_size = CONFIG.DATA.window_size // np.array(
            CONFIG.MODEL.strides).prod()

        if CONFIG.MODEL.bottleneck_type == 'performer':
            self.bottleneck = Performer(dim=CONFIG.MODEL.out_channels[2], depth=CONFIG.MODEL.TRANSFORMER.depth,
                                        heads=CONFIG.MODEL.TRANSFORMER.heads, causal=False,
                                        dim_head=CONFIG.MODEL.TRANSFORMER.dim_head, local_window_size=bottleneck_size)
        elif CONFIG.MODEL.bottleneck_type == 'lstm':
            self.bottleneck = nn.LSTM(input_size=CONFIG.MODEL.out_channels[2], hidden_size=CONFIG.MODEL.out_channels[2],
                                      num_layers=CONFIG.MODEL.TRANSFORMER.depth, batch_first=True)

        self.decoder = Decoder(in_len=self.encoder.out_len,
                               kernel_sizes=CONFIG.MODEL.kernel_sizes,
                               strides=CONFIG.MODEL.strides,
                               out_channels=CONFIG.MODEL.out_channels,
                               tfilm=CONFIG.MODEL.tfilm,
                               n_blocks=CONFIG.MODEL.n_blocks)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        if CONFIG.MODEL.bottleneck_type is not None:
            x3 = x3.permute([0, 2, 1])
            if CONFIG.MODEL.bottleneck_type == 'performer':
                bottle_neck = self.bottleneck(x3)
            elif CONFIG.MODEL.bottleneck_type == 'lstm':
                bottle_neck = self.bottleneck(x3)[0].clone()
            else:
                bottle_neck = self.bottleneck(inputs_embeds=x3)[0]
            bottle_neck += x3
            bottle_neck = bottle_neck.permute([0, 2, 1])
        else:
            bottle_neck = x3
        x_dec = self.decoder([x, x1, x2, bottle_neck])
        return x_dec


class Encoder(nn.Module):
    def __init__(self, max_len, kernel_sizes, strides, out_channels, tfilm, n_blocks):
        super(Encoder, self).__init__()
        self.tfilm = tfilm

        n_layers = len(strides)
        paddings = [(kernel_sizes[i] - strides[i]) //
                    2 for i in range(n_layers)]

        if self.tfilm:
            b_size = max_len // (n_blocks * strides[0])
            self.tfilm_d = TFiLM(block_size=b_size, input_dim=out_channels[0])
            b_size //= strides[1]
            self.tfilm_d1 = TFiLM(block_size=b_size, input_dim=out_channels[1])

        self.downconv = nn.Conv1d(in_channels=1, out_channels=out_channels[0], kernel_size=kernel_sizes[0],
                                  stride=strides[0], padding=paddings[0], padding_mode='replicate')
        self.downconv1 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1],
                                   kernel_size=kernel_sizes[1],
                                   stride=strides[1], padding=paddings[1], padding_mode='replicate')
        self.downconv2 = nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2],
                                   kernel_size=kernel_sizes[2],
                                   stride=strides[2], padding=paddings[2], padding_mode='replicate')
        self.out_len = max_len // (strides[0] * strides[1] * strides[2])

    def forward(self, x):
        print("x len: ", len(x))
        x1 = F.leaky_relu(self.downconv(x), 0.2)  # 2048
        if self.tfilm:
            x1 = self.tfilm_d(x1)
        x2 = F.leaky_relu(self.downconv1(x1), 0.2)  # 1024
        if self.tfilm:
            x2 = self.tfilm_d1(x2)
        x3 = F.leaky_relu(self.downconv2(x2), 0.2)  # 512
        return [x1, x2, x3]


class Decoder(nn.Module):
    def __init__(self, in_len, kernel_sizes, strides, out_channels, tfilm, n_blocks):
        super(Decoder, self).__init__()
        self.tfilm = tfilm
        n_layers = len(strides)
        paddings = [(kernel_sizes[i] - strides[i]) //
                    2 for i in range(n_layers)]

        if self.tfilm:
            in_len *= strides[2]
            self.tfilm_u1 = TFiLM(block_size=in_len //
                                  n_blocks, input_dim=out_channels[1])
            in_len *= strides[1]
            self.tfilm_u = TFiLM(block_size=in_len //
                                 n_blocks, input_dim=out_channels[0])

        self.convt3 = nn.ConvTranspose1d(in_channels=out_channels[2], out_channels=out_channels[1], stride=strides[2],
                                         kernel_size=kernel_sizes[2], padding=paddings[2])
        self.convt2 = nn.ConvTranspose1d(in_channels=out_channels[1], out_channels=out_channels[0], stride=strides[1],
                                         kernel_size=kernel_sizes[1], padding=paddings[1])
        self.convt1 = nn.ConvTranspose1d(in_channels=out_channels[0], out_channels=1, stride=strides[0],
                                         kernel_size=kernel_sizes[0], padding=paddings[0])
        self.dropout = nn.Dropout(0.0)

    def forward(self, x_list):
        x, x1, x2, bottle_neck = x_list
        x_dec = self.dropout(F.leaky_relu(self.convt3(bottle_neck), 0.2))
        if self.tfilm:
            x_dec = self.tfilm_u1(x_dec)
        x_dec = x2 + x_dec

        x_dec = self.dropout(F.leaky_relu(self.convt2(x_dec), 0.2))
        if self.tfilm:
            x_dec = self.tfilm_u(x_dec)
        x_dec = x1 + x_dec
        x_dec = x + torch.tanh(self.convt1(x_dec))
        return x_dec


class TFiLM(nn.Module):
    def __init__(self, block_size, input_dim, **kwargs):
        super(TFiLM, self).__init__(**kwargs)
        self.block_size = block_size
        self.max_pool = nn.MaxPool1d(kernel_size=self.block_size)
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True)

    def make_normalizer(self, x_in):
        """ Pools to downsample along 'temporal' dimension and then
            runs LSTM to generate normalization weights.
        """
        x_in_down = self.max_pool(x_in).permute([0, 2, 1])
        x_rnn, _ = self.lstm(x_in_down)
        return x_rnn.permute([0, 2, 1])

    def apply_normalizer(self, x_in, x_norm):
        """
        Applies normalization weights by multiplying them into their respective blocks.
        """
        # channel first
        n_blocks = x_in.shape[2] // self.block_size
        n_filters = x_in.shape[1]

        # reshape input into blocks
        x_norm = torch.reshape(x_norm, shape=(-1, n_filters, n_blocks, 1))
        x_in = torch.reshape(
            x_in, shape=(-1, n_filters, n_blocks, self.block_size))

        # multiply
        x_out = x_norm * x_in

        # return to original shape
        x_out = torch.reshape(
            x_out, shape=(-1, n_filters, n_blocks * self.block_size))

        return x_out

    def forward(self, x):
        assert len(x.shape) == 3, 'Input should be tensor with dimension \
                                   (batch_size, steps, num_features).'
        assert x.shape[2] % self.block_size == 0, 'Number of steps must be a \
                                                   multiple of the block size.'

        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x


def main():
    DATaset = CustomDataset()
    data_loader = DataLoader(DATaset, shuffle=False,
                             batch_size=16, collate_fn=CustomDataset.collate_fn)
    model = TUNet()

    model.train()

    for batch, (X, y) in tqdm(enumerate(data_loader)):
        print(f"\nshape X: {X.shape}")
        # 1. Forward pass
        y_pred = model(X)
        print(f"shape y_pred: {y_pred.shape}")


if __name__ == "__main__":
    main()
