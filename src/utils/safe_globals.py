"""PyTorch 2.x: allow-list classes for unpickling Lightning checkpoints under ``weights_only=True``."""

import functools

import torch

from src.data.components.cached_kitti_latent_dataset import CachedKittiLatentDataset
from src.data.vio_datamodule import VIODataModule
from src.losses.weighted_loss import (
    CustomWeightedPoseLoss,
    DataWeightedRPMGPoseLoss,
    RPMGPoseLoss,
)
from src.models.components.cavio import CAVIOPoseTransformer
from src.models.components.cavio_gated import GatedCAVIOPoseTransformer
from src.models.components.cavio_visual_residual import VisualResidualCAVIOPoseTransformer
from src.models.components.imu_only import IMUOnlyPoseTransformer
from src.models.components.vift import PoseTransformer
from src.models.weighted_vio_module import WeightedVIOLitModule
from src.testers.latent_kitti_eval_harness import LatentKittiEvalHarness


def register_safe_globals() -> None:
    """Call once at process startup before any checkpoint load (train / eval / resume)."""
    torch.serialization.add_safe_globals(
        [
            functools.partial,
            torch.optim.AdamW,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            torch.nn.L1Loss,
            CachedKittiLatentDataset,
            VIODataModule,
            RPMGPoseLoss,
            DataWeightedRPMGPoseLoss,
            CustomWeightedPoseLoss,
            CAVIOPoseTransformer,
            GatedCAVIOPoseTransformer,
            VisualResidualCAVIOPoseTransformer,
            IMUOnlyPoseTransformer,
            PoseTransformer,
            WeightedVIOLitModule,
            LatentKittiEvalHarness,
        ]
    )
