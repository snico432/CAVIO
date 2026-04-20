from pathlib import Path

import torch
from lightning import LightningModule

from src.utils.plotting.plot_train_losses import plot_train_losses

class WeightedVIOLitModule(LightningModule):
    def __init__(
            self,
            net,
            optimizer,
            scheduler,
            criterion,
            compile,
            tester,
        ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.criterion = criterion
        self.tester = tester


    def forward(self, x, target):
        return self.net(x, target)

    def training_step(self, batch, batch_idx):
        x, target = batch
        out = self.forward(x, target)
        weight = x[-1]
        loss = self.criterion(out, target, weight)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # Log the current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        if hasattr(self.criterion, 'angle_loss'):
            self.log("train/L_rot", self.criterion.angle_loss, on_step=True, on_epoch=True)
            self.log("train/L_trans", self.criterion.translation_loss, on_step=True, on_epoch=True)
        if hasattr(self.criterion, "effective_angle_weight"):
            self.log("train/w_rot", self.criterion.effective_angle_weight, on_step=True, on_epoch=True)
            self.log("train/w_trans", self.criterion.effective_translation_weight, on_step=True, on_epoch=True)
        if hasattr(self.criterion, "weighted_task_loss"):
            self.log("train/loss_weighted_tasks", self.criterion.weighted_task_loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        out = self.forward(x, target)
        weight = x[-1]
        loss = self.criterion(out, target, weight, use_weighted_loss=False)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if hasattr(self.criterion, 'angle_loss'):
            self.log("val/L_rot", self.criterion.angle_loss, on_step=False, on_epoch=True)
            self.log("val/L_trans", self.criterion.translation_loss, on_step=False, on_epoch=True)
        if hasattr(self.criterion, "effective_angle_weight"):
            self.log("val/w_rot", self.criterion.effective_angle_weight, on_step=False, on_epoch=True)
            self.log("val/w_trans", self.criterion.effective_translation_weight, on_step=False, on_epoch=True)
        if hasattr(self.criterion, "weighted_task_loss"):
            self.log("val/loss_weighted_tasks", self.criterion.weighted_task_loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        # This method is not used for our custom testing
        pass

    def on_fit_end(self):
        run_root = Path(self.trainer.default_root_dir)
        csv_logger = None
        for logger in getattr(self.trainer, "loggers", []):
            if logger.__class__.__name__ == "CSVLogger":
                csv_logger = logger
                break
        if csv_logger is None:
            return
        metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
        if not metrics_path.is_file():
            return
        plots_dir = run_root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            loss_plot_path, component_plot_path = plot_train_losses(
                metrics_path=metrics_path, output_dir=plots_dir
            )
            self.print(f"Saved loss plot: {loss_plot_path}")
            if component_plot_path is not None:
                self.print(f"Saved component loss plot: {component_plot_path}")
        except Exception as e:
            print(f"Warning: failed to generate loss plots: {e}")

    def on_test_epoch_end(self):
        with torch.inference_mode():
            pose_results, error_metrics = self.tester.test(self.net)
        metric_sum = 0
        for name, value in error_metrics.items():
            self.log(f"test/{name}", value)
            metric_sum += value
        self.log("hp_metric", metric_sum)

        save_dir = Path(self.trainer.default_root_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.tester.save_results(pose_results, error_metrics, str(save_dir))

    def setup(self, stage):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # Training wraps `net` with torch.compile, so checkpoints store weights under
        # `net._orig_mod.*`. Eval/validate/predict must apply the same wrap before
        # Lightning loads the checkpoint, or load_state_dict key names won't match.
        # Do not compile twice: train→test reuses the same module, and
        # ``torch.compile`` on an already-compiled module breaks (TypeError).
        if self.hparams.compile and stage in ("fit", "test", "validate", "predict"):
            if not hasattr(self.net, "_orig_mod"):
                self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
