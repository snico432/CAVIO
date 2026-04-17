"""Latent KITTI trajectory evaluation: encoder + per-sequence windowed inference loops.

``LatentKittiEvalRunner`` / ``LatentKittiEvalRunnerTokenized`` hold the eval logic.
Lightning wiring lives in ``LatentKittiEvalHarness`` (``latent_kitti_eval_harness.py``).
"""

from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from src.data.components.kitti_eval_sequence_dataset import KittiEvalSequenceDataset
from src.metrics.kitti_metrics import compute_kitti_odometry_metrics
from src.models.components.vio_encoder import Encoder
from src.utils.plotting.kitti_traj_plotting import plotAllPaths, plotPath_2D
from src.utils.kitti_utils import saveSequence


class WrapperModel(torch.nn.Module):
    def __init__(self, params):
        super(WrapperModel, self).__init__()
        self.Feature_net = Encoder(params)

    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        memory = torch.cat((feat_v, feat_i), 2)
        return memory


class LatentKittiEvalRunner:
    def __init__(self, args, wrapper_weights_path, use_history_in_eval=False, plot_label: str = "Ours"):
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(KittiEvalSequenceDataset(args, seq))
        self.args = args

        self.wrapper_model = WrapperModel(args)
        self.load_wrapper_weights(wrapper_weights_path)
        self.wrapper_model.eval()
        self.wrapper_model.to(self.args.device)
        self.use_history_in_eval = use_history_in_eval
        self.plot_label = plot_label

    def load_wrapper_weights(self, weights_path):
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Latent KITTI eval needs the Visual-Selective-VIO encoder weights. "
                f"Not found at {path}. "
                "From the repo root run: python pretrained_model/download_model.py"
            )
        pretrained_w = torch.load(path, map_location="cpu")

        model_dict = self.wrapper_model.state_dict()
        update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}

        assert len(update_dict.keys()) == len(
            self.wrapper_model.Feature_net.state_dict().keys()
        ), "Some weights are not loaded"

        self.wrapper_model.load_state_dict(update_dict)
        print(f"Loaded wrapper model weights from {path}")

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        self.hist = None
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).to(self.args.device)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).to(self.args.device)

            with torch.inference_mode():
                latents = self.wrapper_model(x_in, i_in)

                if (self.hist is not None) and self.use_history_in_eval:
                    results = torch.zeros(latents.shape[0], latents.shape[1], 6)
                    for idx in range(latents.shape[1]):
                        self.hist = torch.roll(self.hist, -1, 1)
                        self.hist[:, -1, :] = latents[:, idx, :]
                        x = (self.hist, None, None)
                        result = net(x, gt_seq)
                        results[:, idx, :] = result[:, -1, :]
                    pose = results
                else:
                    self.hist = latents
                    pose = net((latents, None, None), gt_seq)
            pose_list.append(pose[0, :, :].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def test(self, net, num_gpu: int = 1) -> dict:
        """Run all validation sequences: fill ``self.est`` / ``self.errors``; return relative poses per sequence."""
        pose_results: dict = {}
        error_metrics: dict = {}
        self.est = []

        for i, seq in enumerate(self.args.val_seq):
            print(f"Testing sequence {i + 1} of {len(self.args.val_seq)}")
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
            pose_gt = self.dataloader[i].poses_rel

            pose_results[seq] = {"estimated_poses": pose_est, "gt_poses": pose_gt}

            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = (
                compute_kitti_odometry_metrics(pose_est, pose_gt)
            )

            error_metrics[f"{seq}_t_rel"] = float(t_rel)
            error_metrics[f"{seq}_r_rel"] = float(r_rel)
            error_metrics[f"{seq}_t_rmse"] = float(t_rmse)
            error_metrics[f"{seq}_r_rmse"] = float(r_rmse)

            self.est.append(
                {"pose_est_global": pose_est_global, "pose_gt_global": pose_gt_global, "speed": speed}
            )

        return pose_results, error_metrics

    def generate_plots(self, save_dir, plot_speed: bool = False):
        plotAllPaths(
            self.args.val_seq,
            self.est,
            save_dir,
            method_label=self.plot_label,
            plot_speed=plot_speed,
        )

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir / "{}_pred.txt".format(seq)
            saveSequence(self.est[i]["pose_est_global"], path)
            print("Seq {} saved".format(seq))


class LatentKittiEvalRunnerTokenized:
    def __init__(self, args, wrapper_weights_path, use_history_in_eval=False, plot_label: str = "Ours"):
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(KittiEvalSequenceDataset(args, seq))
        self.args = args

        self.wrapper_model = WrapperModel(args)
        self.load_wrapper_weights(wrapper_weights_path)
        self.wrapper_model.eval()
        self.wrapper_model.to(self.args.device)
        self.use_history_in_eval = use_history_in_eval
        self.plot_label = plot_label

    def load_wrapper_weights(self, weights_path):
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Latent KITTI eval needs the Visual-Selective-VIO encoder weights. "
                f"Not found at {path}. "
                "From the repo root run: python pretrained_model/download_model.py"
            )
        pretrained_w = torch.load(path, map_location="cpu")

        model_dict = self.wrapper_model.state_dict()
        update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}

        assert len(update_dict.keys()) == len(
            self.wrapper_model.Feature_net.state_dict().keys()
        ), "Some weights are not loaded"

        self.wrapper_model.load_state_dict(update_dict)
        print(f"Loaded wrapper model weights from {path}")

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        self.hist = None
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).to(self.args.device)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).to(self.args.device)

            with torch.inference_mode():
                latents = self.wrapper_model(x_in, i_in)

                if (self.hist is not None) and self.use_history_in_eval:
                    results = torch.zeros(latents.shape[0], latents.shape[1], 6)
                    for idx in range(latents.shape[1]):
                        self.hist = torch.roll(self.hist, -1, 1)
                        self.hist[:, -1, :] = latents[:, idx, :]
                        x = (self.hist, None, None)
                        result, _ = net(x, gt_seq)
                        results[:, idx, :] = result[:, -1, :]
                    pose = results
                else:
                    self.hist = latents
                    pose, _ = net((latents, None, None), gt_seq)
            pose_list.append(pose[0, :, :].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def test(self, net, num_gpu: int = 1) -> dict:
        """Run all validation sequences: fill ``self.est`` / ``self.errors``; return relative poses per sequence."""
        pose_results: dict = {}
        error_metrics: dict = {}
        self.est = []

        for i, seq in enumerate(self.args.val_seq):
            print(f"Testing sequence {i + 1} of {len(self.args.val_seq)}")
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
            pose_gt = self.dataloader[i].poses_rel

            pose_results[seq] = {"estimated_poses": pose_est, "gt_poses": pose_gt}

            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = (
                compute_kitti_odometry_metrics(pose_est, pose_gt)
            )

            error_metrics[f"{seq}_t_rel"] = float(t_rel)
            error_metrics[f"{seq}_r_rel"] = float(r_rel)
            error_metrics[f"{seq}_t_rmse"] = float(t_rmse)
            error_metrics[f"{seq}_r_rmse"] = float(r_rmse)

            self.est.append(
                {"pose_est_global": pose_est_global, "pose_gt_global": pose_gt_global, "speed": speed}
            )

        return pose_results, error_metrics

    def generate_plots(self, save_dir, window_size=None):
        del window_size
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(
                seq,
                self.est[i]["pose_gt_global"],
                self.est[i]["pose_est_global"],
                save_dir,
                self.est[i]["speed"],
                method_label=self.plot_label,
            )

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir / "{}_pred.txt".format(seq)
            saveSequence(self.est[i]["pose_est_global"], path)
            print("Seq {} saved".format(seq))
