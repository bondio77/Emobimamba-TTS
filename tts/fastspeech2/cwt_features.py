"""CWT (Continuous Wavelet Transform) feature utilities for pitch modeling."""

import torch
import torch.nn as nn
import numpy as np


class stats_predictor(nn.Module):
    """Predicts mean or std statistics for CWT pitch reconstruction."""

    def __init__(self, d_model, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, D) -> conv expects (B, D, T)
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out = self.linear(out).squeeze(-1)
        return out


def inverse_cwt(cwt_spec, scales, mean, std):
    """Reconstruct pitch from CWT representation.

    Args:
        cwt_spec: CWT spectrogram (B, T, n_scales)
        scales: Wavelet scales
        mean: Mean of original pitch
        std: Std of original pitch

    Returns:
        Reconstructed pitch contour
    """
    if isinstance(scales, np.ndarray):
        scales = torch.from_numpy(scales).float()
    if cwt_spec.is_cuda and not scales.is_cuda:
        scales = scales.to(cwt_spec.device)

    # Inverse CWT via scale summation
    pitch_norm = cwt_spec.sum(dim=-1) / scales.sum()
    pitch = pitch_norm * std + mean
    return pitch
