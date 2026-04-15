from dataclasses import dataclass
import os
from typing import Any, Dict

import numpy as np
import torch

from src.testers.base_tester import BaseTester
from src.testers.latent_kitti_eval_runner import LatentKittiEvalRunner
import json


class LatentKittiEvalHarness(BaseTester):
    def __init__(
        self,
        val_seqs,
        data_dir,
        seq_len,
        folder,
        img_w,
        img_h,
        wrapper_weights_path,
        device,
        v_f_len,
        i_f_len,
        use_history_in_eval=False,
        plot_label: str = "Ours",
        plot_speed: bool = False,
    ):
        super().__init__()
        self.val_seq = val_seqs
        self.plot_label = plot_label
        self.plot_speed = plot_speed
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.folder = folder
        self.img_w = img_w
        self.img_h = img_h
        self.wrapper_weights_path = wrapper_weights_path
        self.device = device
        self.v_f_len = v_f_len
        self.i_f_len = i_f_len

        @dataclass
        class Args:
            val_seq: list
            data_dir: str
            seq_len: int
            folder: str
            img_w: int
            img_h: int
            device: str
            v_f_len: int
            i_f_len: int
            imu_dropout: float

        self.args = Args(
            self.val_seq,
            self.data_dir,
            self.seq_len,
            self.folder,
            self.img_w,
            self.img_h,
            self.device,
            self.v_f_len,
            self.i_f_len,
            0.1,
        )

        self._runner = LatentKittiEvalRunner(
            self.args,
            self.wrapper_weights_path,
            use_history_in_eval=use_history_in_eval,
            plot_label=plot_label,
        )

    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        pose_results, error_metrics = self._runner.test(model)
        return pose_results, error_metrics

    def save_results(self, results: Dict[str, Any], error_metrics: Dict[str, Any], save_dir: str):
        for seq_name, seq_data in results.items():
            np.save(os.path.join(save_dir, f"{seq_name}_estimated_poses.npy"), seq_data["estimated_poses"])
            np.save(os.path.join(save_dir, f"{seq_name}_gt_poses.npy"), seq_data["gt_poses"])
        with open(os.path.join(save_dir, "error_metrics.json"), "w") as f:
            json.dump(error_metrics, f, indent=2, sort_keys=True)
            f.write("\n")
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        self._runner.generate_plots(plots_dir, plot_speed=self.plot_speed)

