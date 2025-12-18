from typing import Any
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from ..logger import ConsoleLogger, TrainingLogger

class TrainInterface:
    def __init__(self, optimizer:torch.optim.Optimizer|Any=None, loss_fn:Any=None, model:torch.nn.Module|Any=None) -> None:
        '''
        Use this to add the default inputs to the interface: a single optimizer, loss_fn, and model
        '''
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model

        self.options = {
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "model": model,
            "base_version": 3.0,
        }

    # --------------------------------
    # Override:
    # --------------------------------
    def call_for_all_tensors(self, func):
        '''
        Sets all tensors to the value returned by func. Necessary since tensor.to() does not move the input tensor, but only returns a moved version
        '''
        return

    def get_frozen_models(self):
        '''return a list of all models that should be on the matching device, but are not changed.'''
        return []

    def get_learning_models(self):
        '''
        return all models that are part of the training, as dict: {model_name: model}
        (only models whose weights are changing)
        used to log model weights
        '''
        return {"model": self.model}
    
    def train(self, dataloader, logger:TrainingLogger, epoch: int) -> dict:
        '''
        Train a single epoch.
        * use the logger to print_substep(batch, total, metrics) in every subbatch
        * get the accumulated metrics with metrics = self.flush_substep_metrics() after an epoch
        * return a dict containing epoch metrics 
            * acculumated (using flush_substep) and/or additional
            * metrics ending with "_" will only be passed to tensorboard, not printed
            * metrics ending with "." will be printed and additionally saved to the logfile
            * "save_model": modelname to save a good model independently of model saving schedule). 
        '''
        raise NotImplementedError()

    # --------------------------------
    # Don't override:
    # --------------------------------
    def to(self, device, distributed=False, find_unused_parameters=False):
        ConsoleLogger.log(f"Moving all models to device {device} (distributed: {distributed})")

        self.device = device
        for model in list(self.get_learning_models().values()) + self.get_frozen_models():
            model.to(device)

        self.call_for_all_tensors(lambda in_tensor: in_tensor.to(device))

        if distributed:
            for _, model in self.get_learning_models().items():
                model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=find_unused_parameters)
    
