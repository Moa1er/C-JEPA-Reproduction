"""
Minimal shim for the bits of `nerv` that SlotFormer's StoSAVi imports.

Replaces these imports inside savi.py:
    from nerv.training import BaseModel
    from nerv.models import deconv_out_shape, conv_norm_act, deconv_norm_act

This avoids pulling in nerv's full dep tree (open3d, pytorch-lightning,
moviepy, opencv-python, wandb, ...) for inference-only use.

Faithful port of nerv/training/model.py and nerv/models/{utils,modules}.py
at tag v0.1.0 (https://github.com/Wuziyi616/nerv/tree/v0.1.0).
"""

import torch.nn as nn


class BaseModel(nn.Module):
    """Stand-in for nerv.training.BaseModel.

    The original carries training hooks (calc_train_loss, _training_step_*,
    loss_function) — none are called by StoSAVi.forward / .encode / .decode,
    so for inference an empty nn.Module subclass is sufficient. StoSAVi
    overrides the dtype/device properties itself.
    """

    def __init__(self):
        super().__init__()


def deconv_out_shape(in_size, stride, padding, kernel_size, out_padding,
                     dilation=1):
    """Output spatial size of a ConvTranspose layer."""
    return (in_size - 1) * stride - 2 * padding + dilation * (
        kernel_size - 1) + out_padding + 1


_ACTS = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'swish': nn.SiLU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'mish': nn.Mish,
    'gelu': nn.GELU,
}


def _get_normalizer(norm, channels):
    if norm == '':
        return nn.Identity()
    if norm == 'bn':
        return nn.BatchNorm2d(channels)
    if norm == 'gn':
        return nn.GroupNorm(16, channels)
    if norm == 'in':
        return nn.InstanceNorm2d(channels)
    if norm == 'ln':
        return nn.LayerNorm(channels)
    raise ValueError(f'Normalizer {norm!r} not supported')


def _get_act(act):
    if act == '':
        return nn.Identity()
    if act not in _ACTS:
        raise ValueError(f'Activation {act!r} not supported')
    return _ACTS[act]()


def conv_norm_act(in_channels, out_channels, kernel_size, stride=1,
                  dilation=1, groups=1, norm='bn', act='relu'):
    """2D Conv + Norm + Act."""
    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=kernel_size // 2,
        dilation=dilation, groups=groups,
        bias=norm not in ('bn', 'in'),
    )
    return nn.Sequential(conv, _get_normalizer(norm, out_channels), _get_act(act))


def deconv_norm_act(in_channels, out_channels, kernel_size, stride=1,
                    dilation=1, groups=1, norm='bn', act='relu'):
    """2D ConvTranspose + Norm + Act."""
    deconv = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=kernel_size // 2,
        output_padding=stride - 1,
        dilation=dilation, groups=groups,
        bias=norm not in ('bn', 'in'),
    )
    return nn.Sequential(deconv, _get_normalizer(norm, out_channels), _get_act(act))
