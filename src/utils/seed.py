"""
seed.py
=======
Reproducibility helper for the Exoplanet Vetting project.

Call ``set_all_seeds()`` at the very start of every script / notebook cell
that involves randomness so that experiments are fully reproducible.

    from src.utils.seed import set_all_seeds
    set_all_seeds()
"""

import os
import random
import numpy as np

from src.utils.config import RANDOM_SEED
from src.utils.logger import get_logger

log = get_logger(__name__)


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Fix random seeds for Python, NumPy, and (if available) TensorFlow
    and PyTorch to guarantee reproducible results.

    Parameters
    ----------
    seed : int
        The seed value to use everywhere.  Defaults to ``RANDOM_SEED``
        defined in ``config.py``.
    """
    # Python built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for hash-based randomness in Python 3.3+
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow (optional — only if installed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        log.debug("TensorFlow seed set.")
    except ImportError:
        pass  # TensorFlow not installed; skip silently

    # PyTorch (optional — only if installed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.debug("PyTorch seed set.")
    except ImportError:
        pass  # PyTorch not installed; skip silently

    log.info(f"All random seeds fixed to {seed}.")
