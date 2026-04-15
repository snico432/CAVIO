"""Shared Hydra + Lightning instantiation for train and eval entrypoints."""

from typing import Any, Dict, List, Tuple

import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_lit_stack(
    cfg: DictConfig,
    *,
    with_callbacks: bool,
) -> Tuple[LightningDataModule, LightningModule, List[Callback], List[Logger], Trainer, Dict[str, Any]]:
    """Instantiate datamodule, model, loggers, and trainer (optionally with callbacks)."""
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks: List[Callback] = []
    if with_callbacks:
        log.info("Instantiating callbacks...")
        callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    if with_callbacks:
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=logger
        )
    else:
        trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict: Dict[str, Any] = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }
    if with_callbacks:
        object_dict["callbacks"] = callbacks

    return datamodule, model, callbacks, logger, trainer, object_dict


def maybe_log_hyperparameters(logger: List[Logger], object_dict: Dict[str, Any]) -> None:
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
