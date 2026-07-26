import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.debug(f"Random seed set to: {seed}")