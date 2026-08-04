#!/usr/bin/env python3
"""Compose TN-PURE-v1 (whole-donor copy) and TN-PTCOMP-v1 (PT scaffold + donor)
from a donor registry + pose tensors.

Reproduces the paper §3.2 composition rules:
  - TN-PURE: downsample donor to 12.5 fps ([::2]), preserve full duration
  - TN-PTCOMP: downsample donor to 12.5 fps, linearly resample to PT scaffold duration,
    then upper-body follows scaffold; hands and face follow donor; with alpha blend
    on remaining body.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import hashlib
import numpy as np
import torch


# SLRTP178 layout (per paper §3.2)
SLRTP178_UPPER = list(range(0, 6))     # 0..5
SLRTP178Articulators = list(range(8, 178))  # 8..177 (hands + face)
SLRTP178_OTHER = [6, 7]


def compose_tn_pure(donor_pose: torch.Tensor) -> torch.Tensor:
    """Whole-donor copy with [::2] subsampling.

    Args:
        donor_pose: [T_orig, 178, 3] at 25 fps
    Returns:
        [T_sub, 178, 3] at 12.5 fps
    """
    return donor_pose[::2].clone()


def compose_tn_ptcomp(
    scaffold_pose: torch.Tensor,
    donor_pose: torch.Tensor,
    alpha: float = 0.88,
) -> torch.Tensor:
    """PT-scaffold composition with donor hands/face and alpha-blended upper body.

    Reproduces paper eq.:
        Upper body [t, U] = donor[0, U] + alpha * (donor[t, U] - donor[0, U])
        Hands + face = donor
        Hips + other = scaffold

    Args:
        scaffold_pose: [T_pt, 178, 3] PT output at 12.5 fps
        donor_pose: [T_donor, 178, 3] donor at 25 fps (will be [::2] and length-resampled)
        alpha: per-paper 0.88 for PHX-public
    Returns:
        [T_pt, 178, 3] composed pose at 12.5 fps
    """
    # Donor: subsample to 12.5 fps and linearly resample to PT length
    donor_sub = donor_pose[::2].float()
    T_pt = scaffold_pose.shape[0]
    T_d = donor_sub.shape[0]
    if T_d != T_pt:
        # Linear interpolation along time
        idx_src = torch.linspace(0, T_d - 1, T_pt, device=donor_sub.device)
        idx_lo = idx_src.floor().long().clamp(max=T_d - 1)
        idx_hi = idx_src.ceil().long().clamp(max=T_d - 1)
        weight = (idx_src - idx_lo).view(-1, 1, 1)
        donor_aligned = donor_sub[idx_lo] * (1 - weight) + donor_sub[idx_hi] * weight
    else:
        donor_aligned = donor_sub

    composed = scaffold_pose.clone().float()
    U = SLRTP178_UPPER
    A = SLRTP178Articulators
    O = SLRTP178_OTHER

    # Upper body: alpha-blended from donor
    composed[:, U] = donor_aligned[0:1, U] + alpha * (donor_aligned[:, U] - donor_aligned[0:1, U])
    # Hands + face: pure donor
    composed[:, A] = donor_aligned[:, A]
    # Hips + other joints: keep scaffold
    composed[:, O] = scaffold_pose[:, O]
    return composed
