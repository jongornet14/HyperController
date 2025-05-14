# tests/test_imports.py

def test_required_imports():
    try:
        import argparse
        import datetime
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import time
        import torch
        import transformers
        import sklearn
        import GPy
        from tqdm import tqdm

        from torch.utils.tensorboard import SummaryWriter
        from tensordict.nn import TensorDictModule
        from tensordict.nn.distributions import NormalParamExtractor
        from torch import nn
        import torchrl  # Assumes base torchrl is importable

    except ImportError as e:
        assert False, f"Missing dependency: {e}"
