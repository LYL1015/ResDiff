import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import pytorch_wavelets as pw
from torchvision import transforms


def show_img(x):
    image = (x[0] + 1) / 2
    plt.imshow(image.cpu().detach().numpy().transpose(2, 1, 0))
    plt.show()


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        # encoding shape: [1, 1, dim] (dim=32)
        return encoding


# Integration of x and noise feature
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# ResSE module
class ResSE(nn.Module):
    def __init__(self, ch_in, reduction=2):
        super(ResSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        tmp = x
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + tmp


# FD Info Spliter
class FD_Info_Spliter(nn.Module):
    def __init__(self, dim, in_channels, out_channels, image_size):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.noise_func = nn.Linear(dim, image_size)
        self.noise_resSE = ResSE(in_channels)
        self.sigma_resSE = ResSE(in_channels * 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.HF_guided_resSE = ResSE(in_channels * 2)
        self.channel_transform = nn.Conv2d(6, 3, 1)

    def forward(self, x, noise_embed):
        cnn_x, x = torch.split(x, 3, dim=1)

        assert x.shape == cnn_x.shape
        # Noise image suppression
        b, c, h, w = x.shape
        noise_embed = self.noise_func(noise_embed.view(b, -1))
        noise_embed = noise_embed.unsqueeze(1).unsqueeze(2).repeat(1, 3, self.image_size, 1)
        noise_atten = self.noise_resSE(noise_embed)
        denoise_x = x * noise_atten

        # High and low frequency information separation
        n, m = x.shape[-2:]
        device = x.device

        # create frequency grid
        xx = torch.arange(n, dtype=torch.float, device=device)
        yy = torch.arange(m, dtype=torch.float, device=device)
        u, v = torch.meshgrid(xx, yy)
        u = u - n / 2
        v = v - m / 2

        # convert tensor to complex tensor and apply FFT
        tensor_complex = torch.stack([cnn_x, torch.zeros_like(cnn_x)], dim=-1)
        tensor_complex = torch.view_as_complex(tensor_complex)
        tensor_fft = torch.fft.fftn(tensor_complex)

        # Concat the real and imaginary parts
        x_real, x_imag = torch.real(tensor_fft), torch.imag(tensor_fft)
        x_fd = torch.cat([x_real, x_imag], dim=1)

        # get sigma, numerical stabilization was performed
        sigma_pre = torch.abs(torch.mean(self.avg_pool(self.sigma_resSE(x_fd)), dim=1)) + self.image_size/2
        sigma_min = torch.tensor(self.image_size-10, device=device).view(1, 1, 1).expand_as(sigma_pre)
        sigma = torch.minimum(sigma_pre, sigma_min)

        # calculate Gaussian high-pass filter
        D = torch.sqrt(u ** 2 + v ** 2).to(device)
        H = 1 - torch.exp(-D ** 2 / (2 * sigma ** 2))
        H = H.to(device).unsqueeze(1)
        H = torch.cat([H, H, H], dim=1)

        # apply Gaussian high-pass filter to FFT
        tensor_filtered_fft = tensor_fft * H

        # get Frequency-domain guided attention weight,thus obtain Low-frequency feature map
        x_real_filterd, x_imag_filterd = torch.real(tensor_filtered_fft), torch.imag(tensor_filtered_fft)
        x_fd_filterd = torch.cat([x_real_filterd, x_imag_filterd], dim=1)
        x_hf_guided_atten = self.HF_guided_resSE(x_fd_filterd)

        x_lf_feature = cnn_x * self.channel_transform(x_hf_guided_atten)

        # IFFT，get High-frequency feature map
        tensor_filtered = torch.fft.ifftn(tensor_filtered_fft)
        x_hf_feature = torch.abs(tensor_filtered)

        return torch.cat([x, cnn_x, denoise_x, x_lf_feature, x_hf_feature], dim=1)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


# HF_guided_CA
class HF_guided_CA(nn.Module):
    def __init__(self, in_channel, norm_groups=32):
        super().__init__()

        self.norm = nn.GroupNorm(norm_groups, in_channel).to('cuda')
        self.q = nn.Conv2d(3, in_channel, 1, bias=False)
        self.kv = nn.Conv2d(in_channel, in_channel * 2, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input, quary):
        batch, channel, height, width = input.shape
        head_dim = channel

        norm = self.norm(input)

        kv = self.kv(norm).view(batch, 1, head_dim * 2, height, width)
        key, value = kv.chunk(2, dim=2)  # bhdyx
        quary = self.q(quary).unsqueeze(1)

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", quary, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, 1, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, 1, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class UNet(nn.Module):
    def __init__(
            self,
            in_channel=9,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8,),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        self.fd_spliter = FD_Info_Spliter(dim=inner_channel, in_channels=3, out_channels=3, image_size=image_size)
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]

        self.hf_ca_list = []

        # DWT downsampling of the number of layers, note to preserve equal depth with the unet
        self.J = 4
        for i in range(self.J):
            self.hf_ca_list.append(HF_guided_CA(inner_channel * (2 ** i)))
        self.hf_ca_list = nn.ModuleList(self.hf_ca_list)

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):

        # Images of each layer obtained by DWT
        dwt_x, _ = torch.split(x, 3, dim=1)

        J = self.J
        dwt_img_list = []
        dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
        dwt_f.cuda()
        x_dwt = dwt_f(dwt_x)[1]
        for i in range(J):
            dwt_img_list.append(x_dwt[i][:, :, 0, :, :] + x_dwt[i][:, :, 1, :, :] + x_dwt[i][:, :, 2, :, :])

        # Performing time-step embedding
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        x = self.fd_spliter(x, t)

        feats = []
        idx = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            if len(feats) != 0 and feats[-1].shape[2:] != x.shape[2:]:
                hf_ca = self.hf_ca_list[idx]
                idx += 1
                query = dwt_img_list.pop(0)
                feats.append(hf_ca(x, query))
            else:
                feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


if __name__ == '__main__':
    img = torch.randn(2, 6, 128, 128).to('cuda')
    t = torch.tensor([[0.645], [0.545]]).to('cuda')
    net = UNet().to('cuda')
    y = net(img, t)
    print(y.shape)
