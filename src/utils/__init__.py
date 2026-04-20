"""Training and evaluation utilities: Hydra/Lightning helpers, KITTI math, plotting.

Shared with the VIFT layout (https://github.com/ybkurt/vift) for ``utils.py``,
``instantiators``, ``rich_utils``, ``pylogger``, ``logging_utils``, ``custom_transform``,
and ``kitti_utils``. VIFT also keeps monolithic ``kitti_eval.py`` and
``kitti_latent_eval.py`` here; CAVIO splits trajectory metrics into ``src/metrics/``,
latent eval loops into ``src/testers/``, and adds ``lit_hydra.py``, ``safe_globals.py``,
and ``plotting/``.
"""

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.utils import extras, get_metric_value, task_wrapper
