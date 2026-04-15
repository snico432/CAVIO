from typing import Any, Dict, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# Project root on PYTHONPATH, PROJECT_ROOT, optional .env — https://github.com/ashleve/rootutils

from src.utils.safe_globals import register_safe_globals

register_safe_globals()

# Set high precision instead of "highest" which is the default.
# Faster matmuls at the cost of lower fidelity.
torch.set_float32_matmul_precision("high")

from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)
from src.utils.lit_hydra import build_lit_stack, maybe_log_hyperparameters

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    datamodule, model, _callbacks, logger, trainer, object_dict = build_lit_stack(
        cfg, with_callbacks=False
    )

    maybe_log_hyperparameters(logger, object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
