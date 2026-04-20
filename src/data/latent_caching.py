"""
Cache visual+IMU latents from the pretrained Encoder for KITTI sequences.

Same role as VIFT ``data/latent_caching.py`` and ``data/latent_val_caching.py``
(https://github.com/ybkurt/vift), consolidated here with argparse, ``rootutils``,
and CAVIO paths under ``data/``.

Writes both splits in one run:

  - train sequences -> data/kitti_latent_data/train/
  - val sequences   -> data/kitti_latent_data/val/

Example (from project root):

  python src/data/latent_caching.py
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rootutils
import torch
from tqdm import tqdm

# Resolve repo root via .project-root, set PROJECT_ROOT, extend PYTHONPATH.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

_PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
# Raw KITTI prep (data_prep.sh) and cached latents live in the top-level ``data/`` folder.
_REPO_DATA_DIR = _PROJECT_ROOT / "data"
_ENCODER_CHECKPOINT = _PROJECT_ROOT / "pretrained_models" / "vf_512_if_256_3e-05.model"

from src.data.components.raw_kitti_dataset import RawKITTIDataset
from src.models.components.vio_encoder import Encoder
from src.utils import custom_transform

# KITTI sequence IDs and output subfolder per split.
_SPLIT_PRESETS: dict[str, tuple[list[str], str]] = {
    "train": (
        ["00", "01", "02", "04", "06", "08", "09"],
        "train",
    ),
    "val": (
        ["05", "07", "10"],
        "val",
    ),
}


@dataclass
class EncoderParams:
    """Config namespace expected by `Encoder(opt)` (attribute access: opt.img_w, etc.)."""

    img_w: int
    img_h: int
    v_f_len: int
    i_f_len: int
    imu_dropout: float
    seq_len: int


class FeatureEncodingModel(torch.nn.Module):
    def __init__(self, params: EncoderParams):
        super(FeatureEncodingModel, self).__init__()
        self.Feature_net = Encoder(params)

    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        return feat_v, feat_i


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cache KITTI latents (train + val) with pretrained Encoder."
    )
    p.add_argument(
        "--kitti-root",
        type=str,
        default=str(_REPO_DATA_DIR / "kitti_data"),
        help="Root folder containing KITTI-style layout (sequences/, imus/, poses/).",
    )
    p.add_argument(
        "--sequence-length",
        type=int,
        default=11,
        help="Window length for KITTI dataset.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for encoding (cuda or cpu).",
    )
    return p


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()

    kitti_root = Path(args.kitti_root)

    transform = custom_transform.Compose(
        [
            custom_transform.ToTensor(),
            custom_transform.Resize((256, 512)),
        ]
    )

    params = EncoderParams(
        img_w=512,
        img_h=256,
        v_f_len=512,
        i_f_len=256,
        imu_dropout=0.1,
        seq_len=args.sequence_length,
    )

    model = FeatureEncodingModel(params)
    pretrained_w = torch.load(_ENCODER_CHECKPOINT, map_location="cpu")
    model_dict = model.state_dict()
    update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
    assert len(update_dict) == len(model_dict), "Some weights are not loaded"
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for param in model.Feature_net.parameters():
        param.requires_grad = False

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model.eval()
    model.to(device)

    for split_name, (sequences, subdir) in _SPLIT_PRESETS.items():
        save_dir = _REPO_DATA_DIR / "kitti_latent_data" / subdir
        print(f"latent_caching: {split_name} sequences={sequences} -> {save_dir}")

        dataset = RawKITTIDataset(
            str(kitti_root),
            train_seqs=sequences,
            transform=transform,
            sequence_length=args.sequence_length,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for i, ((imgs, imus, rot, w), gts) in tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f"latents {split_name}",
            ):
                imgs = imgs.to(device).float()
                imus = imus.to(device).float()
                feat_v, feat_i = model(imgs, imus)
                latent_vector = torch.cat((feat_v, feat_i), 2).squeeze(0)
                base = os.path.join(str(save_dir), f"{i}")
                np.save(f"{base}.npy", latent_vector.cpu().detach().numpy())
                np.save(f"{base}_gt.npy", gts.cpu().detach().numpy())
                np.save(f"{base}_rot.npy", rot.cpu().detach().numpy())
                np.save(f"{base}_w.npy", w.cpu().detach().numpy())


if __name__ == "__main__":
    main()
