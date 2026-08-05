import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

# helper functions

def exists(v):
    return v is not None

# classes

class LightLocoParkour(Module):
    def __init__(self):
        super().__init__()
