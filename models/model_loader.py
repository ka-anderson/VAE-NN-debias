from collections import OrderedDict
import os
import re
import json
from os.path import join

import torch

from models.simple_models import  CustomResNet, MLP
from models.soft_intro_vae import CustomSoftIntroVAE

MODELS = {
    "CustomResNet": CustomResNet,
    "MLP": MLP,
    "CustomSoftIntroVAE": CustomSoftIntroVAE,
}

def without_ddp_prefix(state_dict):
    '''
    ddp appends "module." to every weight
    https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    '''
    model_dict = OrderedDict()
    pattern = re.compile('module.') 
    for k,v in state_dict.items():
        if k.startswith("module."):
            model_dict[re.sub(pattern, '', k)] = v    
        else:
            model_dict = state_dict
    return model_dict

def load_model_from_folder(folder_path, model_name="model", weights_file_name="model_final"):
    '''
    * model_name: name of the model in the training interface (will be looking for model options in opt["train_interface"][model_name])
    * weights_file_name: which of the weightfiles from folder_path/model_weights (or model_weights_local) to load. ".pth" is appended in this method
    '''
    def instantiate_model_class(submodel_opt):
        sub_model_name = submodel_opt["classname"]
        assert sub_model_name in MODELS, f"{sub_model_name} was not found in MODELS and might not be a loadable model."

        for opt_to_delete in ["classname", "weights_from", "partial_weights_from", "version", "feature_return_nodes"]:
            if opt_to_delete in submodel_opt:
                del submodel_opt[opt_to_delete]

        sub_model = MODELS[sub_model_name](**submodel_opt)
        return sub_model

    with open(join(folder_path, "opt.json")) as opt_file:
        opt = json.loads(opt_file.read())
    
    assert model_name in opt["train_interface"], f"No model named {model_name} in training interface options."
    model_opt = opt["train_interface"][model_name]
    weights_path = join(weights_folder(folder_path), f"{weights_file_name}.pth")

    if type(model_opt) == list:
        model_list = []
        for submodel_opt in model_opt:
            model_list.append(instantiate_model_class(submodel_opt))
        model = torch.nn.ModuleList(model_list)
    else:
        model = instantiate_model_class(model_opt)

    model.load_state_dict(without_ddp_prefix(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)))

    if type(model_opt) == list: 
        for sub_model in model:
            sub_model.options.update({
                "partial_weights_from": weights_path
            })
    else:
        model.options.update({
            "weights_from": weights_path
        })

    return model

def weights_folder(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} does not exist")
    if os.path.isdir(join(path, "model_weights")):
        return join(path, "model_weights")
    if os.path.isdir(join(path, "model_weights_local")):
        return join(path, "model_weights_local")
    
    raise FileNotFoundError(f"There is no model_weights or model_weights_local folder in {path}.")