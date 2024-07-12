import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Tuple


class CausalConv3d(nn.Module):
    def __init__(
        self, chan_in, chan_out, kernel_size: Tuple[int, int, int], collapse_space=False, pad_mode="constant", **kwargs
    ):
        super().__init__()
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert height_kernel_size % 2 == 1 and width_kernel_size % 2 == 1

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2 if not collapse_space else 0
        width_pad = width_kernel_size // 2 if not collapse_space else 0

        self.time_pad = time_pad
        self.time_causal_padding = (
            height_pad,
            height_pad,
            width_pad,
            width_pad,
            time_pad,
            0,
        )

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, padding=0, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"
        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


class MultiscaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int = 1024,
        dropout: float = 0.25,
        conv_groups: int = 8,
        window_sizes: list = [1, 5, 9, 11],
        if_padding: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.groups = conv_groups
        self.if_padding = if_padding
        self.window_sizes = window_sizes
        self.embed = []
        self._build()

    def _build(self):
        for window_size in self.window_sizes:
            padding = (window_size // 2, 0, 0)
            if self.if_padding:
                padding = (window_size // 2, 1, 1)
            self.embed.append(
                nn.Sequential(
                    # normal conv3d
                    nn.Conv3d(
                        self.dim,
                        self.output_dim,  # self.output_dim
                        kernel_size=(window_size, 3, 3),
                        stride=(1, 1, 1),
                        padding=padding,
                        groups=self.groups if window_size != 1 else 1,
                    ),
                    nn.LeakyReLU(0.1),  # 0.3164
                    nn.Dropout3d(self.dropout),
                )
            )
        self.mapping = nn.Conv3d(self.output_dim, self.output_dim, 1)
        self.conv1x1 = nn.Conv3d(self.output_dim, self.output_dim, 1)
        self.pool = nn.AvgPool3d((1, 3, 3), stride=1)
        self.embed = nn.ModuleList(self.embed)

    def forward(self, x):
        ret = 0
        for embed in self.embed:
            ret += embed(x)
        ret = self.mapping(ret)
        return ret + self.pool(self.conv1x1(x))


class CausalMultiscaleBlock(MultiscaleBlock):
    def __init__(
        self,
        dim: int,
        output_dim: int = 1024,
        dropout: float = 0.25,
        conv_groups: int = 8,
        window_sizes: list = [1, 5, 9, 11],
        if_padding: bool = False,
    ):
        super().__init__(dim, output_dim, dropout, conv_groups, window_sizes, if_padding)

    def _build(self):
        for window_size in self.window_sizes:
            self.embed.append(
                nn.Sequential(
                    CausalConv3d(
                        self.dim,
                        self.output_dim,
                        kernel_size=(window_size, 3, 3),
                        collapse_space=not self.if_padding,
                        groups=self.groups if window_size != 1 else 1,
                    ),
                    nn.LeakyReLU(0.1),  # 0.33
                    nn.Dropout3d(self.dropout),
                )
            )
        self.mapping = nn.Conv3d(self.output_dim, self.output_dim, 1)
        self.conv1x1 = nn.Conv3d(self.output_dim, self.output_dim, 1)
        self.pool = nn.AvgPool3d((1, 3, 3), stride=(1, 1, 1))
        self.embed = nn.ModuleList(self.embed)

    def forward(self, x):
        ret = 0
        for embed in self.embed:
            ret += embed(x)
        ret = self.mapping(ret)
        return ret + self.pool(self.conv1x1(x))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AdaLN_Zero3D(nn.Module):
    def __init__(self, in_features: int, act_layer: Callable[..., nn.Module] = nn.SiLU):
        super().__init__()
        self.in_features = in_features
        self.act_layer = act_layer
        self.adaLN_modulation = nn.Sequential(
            self.act_layer(),
            nn.Conv3d(self.in_features, 3 * self.in_features, 1, bias=True),
        )
        self.net = nn.Sequential(
            self.act_layer(),
            nn.Conv3d(self.in_features, self.in_features, 1, bias=True),
        )
        self.initialize()

    def initialize(self):
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.normal_(self.net[1].weight, 0, 0.02)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        x = x + gate * self.net(modulate(x, shift, scale))
        return x


class SingleStage3DModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal=False):
        super(SingleStage3DModel, self).__init__()
        self.conv_1x1 = nn.Conv3d(dim, num_f_maps, 1)
        self.conv_type = DilatedResidualCausal3DLayer if causal else DilatedResidual3DLayer
        self.layers = nn.ModuleList([copy.deepcopy(self.conv_type(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv3d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)  # batch x features x time x H x W
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out  # batch x features x time x H x W


class DilatedResidual3DLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, dropout=0.25):
        super(DilatedResidual3DLayer, self).__init__()
        space_kernel_size = 3
        self.conv_dilated = nn.Conv3d(
            in_channels,
            out_channels,
            (3, space_kernel_size, space_kernel_size),
            padding=(dilation, space_kernel_size // 2, space_kernel_size // 2),
            dilation=(dilation, 1, 1),
            groups=out_channels,
        )
        self.norm = nn.BatchNorm3d(in_channels)
        self.conv_1x1 = nn.Conv3d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.norm(self.conv_dilated(x)))  # batch x features x time x H x W
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class DilatedResidualCausal3DLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, dropout=0.25):
        super(DilatedResidualCausal3DLayer, self).__init__()
        space_kernel_size = 3
        self.padding = (
            space_kernel_size // 2,
            space_kernel_size // 2,
            space_kernel_size // 2,
            space_kernel_size // 2,
            2 * dilation,
            0,
        )
        self.conv_dilated = nn.Conv3d(
            in_channels,
            out_channels,
            (3, space_kernel_size, space_kernel_size),
            dilation=(dilation, 1, 1),
            padding=0,
            groups=out_channels,
        )
        self.norm = nn.BatchNorm3d(in_channels)
        self.conv_1x1 = nn.Conv3d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch x features x time x H x W
        out = F.pad(x, self.padding, mode="constant")
        out = F.relu(self.norm(self.conv_dilated(out)))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# Vi-ST
class ViST(nn.Module):
    def __init__(
        self,
        video_layers,
        video_dim,
        dinov2_dim,
        cells,
        grids=16,
        causal=True,
    ):
        super().__init__()
        self.cells = cells
        self.grids = grids
        self.stage1 = SingleStage3DModel(video_layers, video_dim, dinov2_dim, cells, causal=causal)
        multi_scale_block_clz = MultiscaleBlock if not causal else CausalMultiscaleBlock
        self.msb_1 = multi_scale_block_clz(
            dim=cells,
            output_dim=cells,
            dropout=0.25,
            window_sizes=[1, 25],
            conv_groups=cells,
        )
        self.msb_2 = multi_scale_block_clz(
            dim=cells,
            output_dim=cells,
            dropout=0.25,
            window_sizes=[1, 13, 21],
            conv_groups=cells,
        )
        self.msb_3 = multi_scale_block_clz(
            dim=cells,
            output_dim=cells,
            dropout=0.25,
            window_sizes=[1, 7, 9],
            conv_groups=cells,
        )
        self.msb_4 = multi_scale_block_clz(
            dim=cells,
            output_dim=cells,
            dropout=0.25,
            window_sizes=[1, 3, 5],
            conv_groups=cells,
        )
        self.msb_5 = nn.Linear((self.grids - 2 * 4) ** 2 * cells, cells)
        self.adaln = AdaLN_Zero3D(cells)

    def forward(self, x, rf_mask=[], **kwargs):
        # Input x: [batch, times, patches, dims]
        # Input mask: [batch, 1, grids, grids, times]
        # Input rf_mask: [batch, cells, grids, grids]
        x = rearrange(x, "b t (p q) d -> b d t p q", p=16, q=16)
        rf_mask = rf_mask.sum(axis=1, keepdim=True).unsqueeze(2).repeat(1, self.cells, x.size()[2], 1, 1)
        out = self.adaln(self.stage1(x), rf_mask)
        out = self.msb_1(out)
        out = self.msb_2(out)
        out = self.msb_3(out)
        out = self.msb_4(out)
        out = rearrange(out, "b c t p q -> b t (c p q)")
        out = self.msb_5(out).permute(0, 2, 1)
        return out


if __name__ == "__main__":
    import torchinfo

    batch_size = 16
    video_layers = 6
    video_dim = 256
    cells = 90
    dinov2_dim = 1024
    grids = 16
    times = 128
    cells = 90

    model = ViST(video_layers, video_dim, dinov2_dim, cells, causal=True)
    print(model)
    inputs = [
        torch.randn(batch_size, times, grids * grids, dinov2_dim),
        torch.randn(batch_size, cells, grids, grids),
    ]
    torchinfo.summary(
        model,
        input_data=inputs,
        dtypes=[torch.float16, torch.long],
        col_names=(
            "input_size",
            "output_size",
            "num_params",
        ),
        depth=3,
        device="cuda",
    )

# =============================================================================================================================
# Layer (type:depth-idx)                             Input Shape               Output Shape              Param #
# =============================================================================================================================
# ViST                                               [16, 128, 256, 1024]      [16, 90, 128]             --
# ├─SingleStage3DModel: 1-1                          [16, 1024, 128, 16, 16]   [16, 90, 128, 16, 16]     --
# │    └─Conv3d: 2-1                                 [16, 1024, 128, 16, 16]   [16, 256, 128, 16, 16]    262,400
# │    └─ModuleList: 2-2                             --                        --                        --
# │    │    └─DilatedResidualCausal3DLayer: 3-1      [16, 256, 128, 16, 16]    [16, 256, 128, 16, 16]    73,472
# │    │    └─DilatedResidualCausal3DLayer: 3-2      [16, 256, 128, 16, 16]    [16, 256, 128, 16, 16]    73,472
# │    │    └─DilatedResidualCausal3DLayer: 3-3      [16, 256, 128, 16, 16]    [16, 256, 128, 16, 16]    73,472
# │    │    └─DilatedResidualCausal3DLayer: 3-4      [16, 256, 128, 16, 16]    [16, 256, 128, 16, 16]    73,472
# │    │    └─DilatedResidualCausal3DLayer: 3-5      [16, 256, 128, 16, 16]    [16, 256, 128, 16, 16]    73,472
# │    │    └─DilatedResidualCausal3DLayer: 3-6      [16, 256, 128, 16, 16]    [16, 256, 128, 16, 16]    73,472
# │    └─Conv3d: 2-3                                 [16, 256, 128, 16, 16]    [16, 90, 128, 16, 16]     23,130
# ├─AdaLN_Zero3D: 1-2                                [16, 90, 128, 16, 16]     [16, 90, 128, 16, 16]     --
# │    └─Sequential: 2-4                             [16, 90, 128, 16, 16]     [16, 270, 128, 16, 16]    --
# │    │    └─SiLU: 3-7                              [16, 90, 128, 16, 16]     [16, 90, 128, 16, 16]     --
# │    │    └─Conv3d: 3-8                            [16, 90, 128, 16, 16]     [16, 270, 128, 16, 16]    24,570
# │    └─Sequential: 2-5                             [16, 90, 128, 16, 16]     [16, 90, 128, 16, 16]     --
# │    │    └─SiLU: 3-9                              [16, 90, 128, 16, 16]     [16, 90, 128, 16, 16]     --
# │    │    └─Conv3d: 3-10                           [16, 90, 128, 16, 16]     [16, 90, 128, 16, 16]     8,190
# ├─CausalMultiscaleBlock: 1-3                       [16, 90, 128, 16, 16]     [16, 90, 128, 14, 14]     --
# │    └─ModuleList: 2-6                             --                        --                        --
# │    │    └─Sequential: 3-11                       [16, 90, 128, 16, 16]     [16, 90, 128, 14, 14]     72,990
# │    │    └─Sequential: 3-12                       [16, 90, 128, 16, 16]     [16, 90, 128, 14, 14]     20,340
# │    └─Conv3d: 2-7                                 [16, 90, 128, 14, 14]     [16, 90, 128, 14, 14]     8,190
# │    └─Conv3d: 2-8                                 [16, 90, 128, 16, 16]     [16, 90, 128, 16, 16]     8,190
# │    └─AvgPool3d: 2-9                              [16, 90, 128, 16, 16]     [16, 90, 128, 14, 14]     --
# ├─CausalMultiscaleBlock: 1-4                       [16, 90, 128, 14, 14]     [16, 90, 128, 12, 12]     --
# │    └─ModuleList: 2-10                            --                        --                        --
# │    │    └─Sequential: 3-13                       [16, 90, 128, 14, 14]     [16, 90, 128, 12, 12]     72,990
# │    │    └─Sequential: 3-14                       [16, 90, 128, 14, 14]     [16, 90, 128, 12, 12]     10,620
# │    │    └─Sequential: 3-15                       [16, 90, 128, 14, 14]     [16, 90, 128, 12, 12]     17,100
# │    └─Conv3d: 2-11                                [16, 90, 128, 12, 12]     [16, 90, 128, 12, 12]     8,190
# │    └─Conv3d: 2-12                                [16, 90, 128, 14, 14]     [16, 90, 128, 14, 14]     8,190
# │    └─AvgPool3d: 2-13                             [16, 90, 128, 14, 14]     [16, 90, 128, 12, 12]     --
# ├─CausalMultiscaleBlock: 1-5                       [16, 90, 128, 12, 12]     [16, 90, 128, 10, 10]     --
# │    └─ModuleList: 2-14                            --                        --                        --
# │    │    └─Sequential: 3-16                       [16, 90, 128, 12, 12]     [16, 90, 128, 10, 10]     72,990
# │    │    └─Sequential: 3-17                       [16, 90, 128, 12, 12]     [16, 90, 128, 10, 10]     5,760
# │    │    └─Sequential: 3-18                       [16, 90, 128, 12, 12]     [16, 90, 128, 10, 10]     7,380
# │    └─Conv3d: 2-15                                [16, 90, 128, 10, 10]     [16, 90, 128, 10, 10]     8,190
# │    └─Conv3d: 2-16                                [16, 90, 128, 12, 12]     [16, 90, 128, 12, 12]     8,190
# │    └─AvgPool3d: 2-17                             [16, 90, 128, 12, 12]     [16, 90, 128, 10, 10]     --
# ├─CausalMultiscaleBlock: 1-6                       [16, 90, 128, 10, 10]     [16, 90, 128, 8, 8]       --
# │    └─ModuleList: 2-18                            --                        --                        --
# │    │    └─Sequential: 3-19                       [16, 90, 128, 10, 10]     [16, 90, 128, 8, 8]       72,990
# │    │    └─Sequential: 3-20                       [16, 90, 128, 10, 10]     [16, 90, 128, 8, 8]       2,520
# │    │    └─Sequential: 3-21                       [16, 90, 128, 10, 10]     [16, 90, 128, 8, 8]       4,140
# │    └─Conv3d: 2-19                                [16, 90, 128, 8, 8]       [16, 90, 128, 8, 8]       8,190
# │    └─Conv3d: 2-20                                [16, 90, 128, 10, 10]     [16, 90, 128, 10, 10]     8,190
# │    └─AvgPool3d: 2-21                             [16, 90, 128, 10, 10]     [16, 90, 128, 8, 8]       --
# ├─Linear: 1-7                                      [16, 128, 5760]           [16, 128, 90]             518,490
# =============================================================================================================================
# Total params: 1,702,952
# Trainable params: 1,702,952
# Non-trainable params: 0
# Total mult-adds (G): 511.77
# =============================================================================================================================
# Input size (MB): 2148.96
# Forward/backward pass size (MB): 26000.00
# Params size (MB): 6.81
# Estimated Total Size (MB): 28155.77
# =============================================================================================================================
