"""Abstract tester interface (shared with VIFT ``src/testers/base_tester.py``).

https://github.com/ybkurt/vift — same ``test`` / ``save_results`` contract; concrete testers
implement latent or raw-KITTI evaluation.
"""

from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Any

class BaseTester(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Any], save_dir: str):
        pass