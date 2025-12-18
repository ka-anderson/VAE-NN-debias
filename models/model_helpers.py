from math import ceil
from torch import nn
import torch

def freeze_parameters(model: nn.Module, requires_grad=False):
    for param in model.parameters():
        param.requires_grad_(requires_grad)
    return model

def print_module_device(module: torch.nn.Module):
    for name, param in module.named_parameters():
        print(f"{name}: {param.device}")

def stacked_distance_matrix(a, b):
    '''
    in: two vectors (shape [BS])
    out: dist matrix
        * matrix rows: one point from a (in the chosen dim)
        * matrix columns: distances between that point in a and all points in b
    '''
    a_expanded = a.view(-1, 1)
    b_expanded = b.view(1, -1)

    distances = torch.abs(a_expanded - b_expanded)
    return distances

def gaussian_kernel_1d(sigma: float = 1, num_sigmas: float = 3.) -> torch.Tensor:
    radius = ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())

class IdentityModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.options = {}

    def forward(self, x):
        return x
    
class View(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape
        self.options = {"shape": shape}

    def forward(self, x):
        return x.view(x.shape[0], *self.shape) # shape[0] is the batch size

class DebugLogger(nn.Module):
    def __init__(self, prefix="") -> None:
        super().__init__()
        self.prefix = prefix

    def forward(self, x):
        print(f"----{self.prefix}----")
        print(x.shape)
        # print(f"---/{self.prefix}----")
        return x

class ResidualWrapper(torch.nn.Module):
    def __init__(self, inner_module) -> None:
        super().__init__()
        self.register_module("inner_module", inner_module)
        self.options = {
            "inner_module": inner_module,
        }

    def forward(self, x):
        return self.inner_module(x) + x
    
class _DimBased(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.options = {"dim": dim}

class SetToZero(_DimBased):
    def forward(self, x):
        x[:, self.dim] = 0
        return x

class GetDimension(_DimBased):
    '''
    Return a single dimension of the input (for the full batch)
    '''
    def forward(self, x):
        return x[:, self.dim]

class DimBasedEnsemble(torch.nn.Module):
    '''
    Only for evaluation purposes. Loading a model from this class (load_model_from_folder) will not currently work.
    '''
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        self.options = {
            "module_list": module_list,
        }

    def forward(self, x):
        out = torch.zeros_like(x)
        for dim in range(len(self.module_list)):
            out[:, dim] = self.module_list[dim](x[:, dim])
        
        return out