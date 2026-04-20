"""Pose losses (weighted rotation/translation, RPMG, optional data weighting).

Derived from VIFT ``src/metrics/weighted_loss.py`` (https://github.com/ybkurt/vift).
CAVIO places this module under ``src/losses/`` and imports ``src.utils.*`` (VIFT used
bare ``utils.*``). Core classes (:class:`WeightedMSEPoseLoss`, :class:`WeightedMAEPoseLoss`,
:class:`RPMGPoseLoss`, :class:`DataWeightedRPMGPoseLoss`, etc.) follow the upstream layout.

:class:`CustomWeightedPoseLoss` is **modified** in CAVIO: configurable ``rot_w`` /
``trans_w`` and documented aggregation, replacing the fixed per-axis coefficients in VIFT.
:class:`RPMGPoseLoss` additionally exposes ``angle_loss`` / ``translation_loss`` for logging
in :class:`~src.models.weighted_vio_module.WeightedVIOLitModule`.
"""

import torch
import torch.nn as nn
from src.utils.kitti_utils import eulerAnglesToRotationMatrixTorch as etr
from src.utils import rpmg

class WeightedMSEPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super(WeightedMSEPoseLoss, self).__init__()
        self.angle_weight = angle_weight

    def forward(self, poses, gts):
        angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss
    
class WeightedMAEPoseLoss(nn.Module):
    def __init__(self, angle_weight=10):
        super(WeightedMAEPoseLoss, self).__init__()
        self.angle_weight = angle_weight

    def forward(self, poses, gts):
        angle_loss = torch.nn.functional.l1_loss(poses[:,:,:3], gts[:, :, :3])
        translation_loss = torch.nn.functional.l1_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss

class LieTorchPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super().__init__()
        self.angle_weight= angle_weight
    def forward(self, poses, gts): # poses : translation + SE3 matrix, gts : translation + Euler Angles

        pass

class RPMGPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.angle_weight = angle_weight
    def forward(self, poses, gts, weights, use_weighted_loss=True):
        angle_loss = torch.nn.functional.l1_loss(
            rpmg.simple_RPMG.apply(
                                   etr(poses[:,:,:3]).view(poses.shape[0]*poses.shape[1],9),
                                   1/4,
                                   0.01
                                   ).view(-1,9),
            etr(gts[:,:,:3]).view(poses.shape[0]*poses.shape[1],9)
        )
        
        translation_loss = torch.nn.functional.l1_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        self.angle_loss = angle_loss.detach()
        self.translation_loss = translation_loss.detach()
        return pose_loss

class DataWeightedRPMGPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.angle_weight = angle_weight
        self.base_loss_fn = base_loss_fn # make sure reduction is None
    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        angle_loss = self.base_loss_fn(
            rpmg.simple_RPMG.apply(
                                   etr(poses[:,:,:3]).view(poses.shape[0]*poses.shape[1],9),
                                   1/4,
                                   0.01
                                   ).view(-1,9),
            etr(gts[:,:,:3]).view(poses.shape[0]*poses.shape[1],9)
        )
        
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:, :, 3:])
        
        total_loss = self.angle_weight * torch.sum(angle_loss,-1).view(poses.shape[0],-1) + torch.sum(translation_loss,-1).view(poses.shape[0],-1)
        if use_weighted_loss and (weights is not None):
            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.view(-1, 1)  # reshape to (batch_size, 1, 1)
            
            # Apply weights
            total_loss = weights * total_loss.sum(dim=(1)).view(-1, 1)
        return total_loss.mean()

class DataWeightedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        batch_size, seq_len, _ = poses.shape

        # Calculate losses for rotation and translation separately
        angle_loss = self.base_loss_fn(poses[:,:,:3], gts[:,:,:3])
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:,:,3:])

        # Combine losses
        total_loss = self.angle_weight * angle_loss + translation_loss

        if use_weighted_loss and (weights is not None):
            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.view(-1, 1, 1)  # reshape to (batch_size, 1, 1)
            
            # Apply weights
            total_loss = weights * total_loss.sum(dim=(1, 2)).view(-1, 1, 1)
        
        return total_loss.mean()

class CustomWeightedPoseLoss(nn.Module):
    """
    Per-channel L1 (via ``base_loss_fn``) on ``[euler×3, t×3]``.

    Defaults: uniform rotation weights; translation weights uniform on X/Z with a higher
    weight on Y (index 4), i.e. vertical / out-of-plane vs a typical KITTI X–Z path.

    Combined as ``angle_weight * mean(rot) + mean(trans)`` so it lines up with
    :class:`RPMGPoseLoss` (rotation block scaled, then translation block).

    ``weights`` / ``use_weighted_loss`` are accepted for API compatibility with
    :class:`WeightedVIOLitModule` but LDS-style sample weighting is not applied here.
    """

    def __init__(
        self,
        base_loss_fn,
        angle_weight: float = 25.0,
        rot_w=(1.0, 1.0, 1.0),
        trans_w=(1.0, 2.5, 1.0),
    ):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight
        self.rot_w = tuple(float(x) for x in rot_w)
        self.trans_w = tuple(float(x) for x in trans_w)
        if len(self.rot_w) != 3 or len(self.trans_w) != 3:
            raise ValueError("rot_w and trans_w must each have length 3.")

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        loss1 = self.base_loss_fn(poses[:, :, 0], gts[:, :, 0])
        loss2 = self.base_loss_fn(poses[:, :, 1], gts[:, :, 1])
        loss3 = self.base_loss_fn(poses[:, :, 2], gts[:, :, 2])
        loss4 = self.base_loss_fn(poses[:, :, 3], gts[:, :, 3])
        loss5 = self.base_loss_fn(poses[:, :, 4], gts[:, :, 4])
        loss6 = self.base_loss_fn(poses[:, :, 5], gts[:, :, 5])

        r_sum = sum(self.rot_w)
        t_sum = sum(self.trans_w)
        rot_term = (
            self.rot_w[0] * loss1 + self.rot_w[1] * loss2 + self.rot_w[2] * loss3
        ) / r_sum
        trans_term = (
            self.trans_w[0] * loss4 + self.trans_w[1] * loss5 + self.trans_w[2] * loss6
        ) / t_sum

        total_loss = self.angle_weight * rot_term + trans_term
        self.angle_loss = rot_term.detach()
        self.translation_loss = trans_term.detach()
        return total_loss


class AngleWeightedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        batch_size, seq_len, _ = poses.shape

        # Calculate losses for rotation and translation separately
        angle_loss = self.base_loss_fn(poses[:,:,:3], gts[:,:,:3])
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:,:,3:])

        # Combine losses
        total_loss = self.angle_weight * angle_loss + translation_loss
        
        return total_loss.mean()

class TokenizedPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super().__init__()

    def forward(self, poses, gts):
        
        out, loss = poses
        return loss


class WeightedTokenizedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        out, ce_loss = poses
        return ce_loss
