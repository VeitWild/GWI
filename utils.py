import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import random
import numpy as np


def make_deterministic(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def build_L_matrix(L_diag, L_lower, jitter=0.01):
    M = L_diag.size(0)
    L = torch.ones(M, M)
    L = L - torch.diag(L) + torch.diag(torch.exp(L_diag)+jitter ) # I think this only works bc of some shady broadcasting
    indices = torch.tril_indices(M, M, -1)
    #print(L[indices[0,],indices[1,]]) # this gives a vector not a matrix for some wild reason
    L[indices[0,],indices[1,]] = L_lower

    return L


# Some tests
#M = 5
#M_tilde = int((M * (M - 1)) / 2)
#L_diag = torch.ones(M)
#L_lower = torch.arange(1,M_tilde+1).float()
#Sigma = build_L_matrix(L_diag, L_lower=L_lower)
#print(Sigma)
