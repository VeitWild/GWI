import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import random
import numpy as np

def make_deterministic(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)