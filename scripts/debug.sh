#!/usr/bin/env bash
#
# Fast smoke test for the training stack: one epoch, a handful of train/val/test
# batches, no ``torch.compile``, ``data.num_workers=0`` for easy debugging.
# Use before long runs (e.g. ``schedule_report.sh``) to confirm Hydra, data, and GPU path work.
# Run from the CAVIO repository root: ``bash scripts/debug.sh``

python src/train.py \
  experiment=cavio_baseline \
  logger=many_loggers \
  trainer=gpu \
  trainer.max_epochs=1 \
  trainer.min_epochs=1 \
  +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=1 \
  +trainer.limit_test_batches=2 \
  +trainer.log_every_n_steps=1 \
  data.num_workers=0 \
  model.compile=false \
  test=true