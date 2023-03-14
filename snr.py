# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import numpy as np
import torch
from typing import Union

EPS = 1e-8

def si_snr(target: Union[torch.tensor, np.ndarray],
           estimate: Union[torch.tensor, np.ndarray]) -> torch.tensor:
    """Calculates SI-SNR estiamte from target audio and estiamte audio. The
    audio sequene is expected to be a tensor/array of dimension more than 1.
    The last dimension is interpreted as time.

    The implementation is based on the example here:
    https://www.tutorialexample.com/wp-content/uploads/2021/12/SI-SNR-definition.png

    Parameters
    ----------
    target : Union[torch.tensor, np.ndarray]
        Target audio waveform.
    estimate : Union[torch.tensor, np.ndarray]
        Estimate audio waveform.

    Returns
    -------
    torch.tensor
        SI-SNR of each target and estimate pair.
    """
    if not torch.is_tensor(target):
        target: torch.tensor = torch.tensor(target)
    if not torch.is_tensor(estimate):
        estimate: torch.tensor = torch.tensor(estimate)

    # zero mean to ensure scale invariance
    s_target = target - torch.mean(target, dim=-1, keepdim=True)
    s_estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj ** 2,
                              dim=-1) / (torch.sum(e_noise ** 2,
                                                   dim=-1) + EPS)
    return 10 * torch.log10(pair_wise_sdr + EPS)
    

def sdr(estimate, target):
    # x: clean speech target
    # n: noise signal
    # y = x + n: noisy input
    # \hat{x} : estimate signal

    # x_target = <x, \hat{x}> / ||x||^2 * x
    # e_noise = <n, \hat{x}> / ||n||^2 * n
    # e_artif = \hat{x} - x_target - e_noise

    # sdr = 10 log10(||x_target||^2 / ||e_noise + e_artif||^2)
    #     = 10 log10(||x_target||^2 / ||\hat{x} - x_target||^2)
    pass