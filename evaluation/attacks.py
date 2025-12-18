from dataclasses import dataclass
import os
import shutil
from typing import Literal

import torch

from datasets.chexpert import DatasetChexpertSmall
from datasets.dataset_base import CustomDataset
from datasets.ffhq import DatasetFFHQAttributes
from datasets.mnist import DatasetMNISTBackground
from evaluation.metrics import generate_supervised_metrics_for_models
from misc.helpers import exp_name, repo_dir
from models.model_helpers import SetToZero, View
from models.model_loader import load_model_from_folder
from models.simple_models import MLP, CustomResNet
from models.one_dim_resnet import MSResNet
from models.vaes import VariationalLayer
from models.model_helpers import IdentityModule
from training.train_interfaces.simple_training import ClassificationTrainInterface
from training.trainer import run_training
from os.path import join

BATCH_SIZE = 256

@dataclass
class ATTACK_TYPES:
    '''
    * all: everything *but* the dimension given by "dim"

    * mask: set it to zero
    * encode: feed it to the encoder and insert it into the original latent position. 
    '''
    keep_all = 0
    keep_all_mask_dim = 1
    encode_all = 2
    encode_all_mask_dim = 3


class Attack():
    def __init__(self, attack_model_id:Literal["resnet18", "resnet34", "mlp", "linear", "1dResnet"], target_label, exp_id, output_run_id, attack_type, dim=None, training_epochs=30, latent_encoder=None, vae_encoder=None, vae_decoder=None, decoded_dim=None, img_size=None, output_filename_prefix="", use_existing_presave_folder=False):
        '''
        Given an input, encode it using an encoder-latent_encoder-decoder, then train a model to recognize a defined target label from the decoded result.

        * temporarily saves the output in exp_id/attack_run_id, to be deleted later. The evaluation result is saved in exp_id/output_run_id/attack_run_id.md.
        * alternative vae_encoder and vae_decoder can be provided, otherwise the default is used
        * vae_encoder: should return a latent (for vae encoders after the sampling)
        * decoded_dim: size of the output decoder(encoder(input_image)). If it is None, assumes that the output is img_size x img_size. Only required for mlp attackers, ResNets don't need to know the input size.
        * use_existing_presave_folder: for large datasets, if the training had already been done with the same encoder and is still saved to the presaved data folder, no need to encode again.
        '''
        self.target_label = target_label
        self.decoded_dim = decoded_dim
        self.img_size = img_size

        assert latent_encoder != None or attack_type in [ATTACK_TYPES.keep_all, ATTACK_TYPES.keep_all_mask_dim], f"Attack type {attack_type} needs a latent encoder."
        assert not use_existing_presave_folder or self.is_dataset_large(), "Preloaded folder can only exist if the data was saved before, which is only done for large datasets."


        dataset = self.get_dataset()
        vae_encoder = vae_encoder if vae_encoder != None else self.get_default_vae_encoder()
        vae_decoder = vae_decoder if vae_decoder != None else self.get_default_vae_decoder()
        attack_model = self.get_attack_model(attack_model_id)
        attack_model_loss = self.get_attack_model_loss()
        eval_metrics = self.get_eval_metrics()

        attack_run_id = "" if output_filename_prefix == "" else f"{output_filename_prefix}_"
        attack_run_id += f'{output_run_id}_atk{attack_type}_{attack_model_id}'
        if target_label != None:
            attack_run_id += f"_target{target_label}"
        if dim != None:
            attack_run_id += f"_dim{dim}"

        if attack_type != ATTACK_TYPES.keep_all or not isinstance(vae_encoder, IdentityModule) or not isinstance(vae_decoder, IdentityModule): # if the dataset needs to be encoded
            
            if not use_existing_presave_folder:
                encoder_module_list = [vae_encoder]

                if attack_type == ATTACK_TYPES.keep_all_mask_dim:
                    encoder_module_list.append(SetToZero(dim))
                
                elif attack_type == ATTACK_TYPES.encode_all:
                    encoder_module_list.append(latent_encoder)

                elif attack_type == ATTACK_TYPES.encode_all_mask_dim:
                    encoder_module_list.extend([latent_encoder, SetToZero(dim)])

                elif attack_type != ATTACK_TYPES.keep_all:
                    raise Exception(f"Unknown attack type: {attack_type}")
                
                encoder_module_list.append(vae_decoder)
                encoder = torch.nn.Sequential(*encoder_module_list)
            else: 
                encoder = None

            if self.is_dataset_large():
                dataset.encode_and_save_dataset(encoder, use_existing_folder=use_existing_presave_folder)
            else:
                dataset.preload_dataset_with_encoder(train=True, encoder=encoder)
                dataset.preload_dataset_with_encoder(train=False, encoder=encoder)

        # ---------------
        # Attack training
        # ---------------
        train_interface = ClassificationTrainInterface(
            optimizer=torch.optim.Adam(list(attack_model.parameters()), lr=self.get_lr()),
            loss_fn=attack_model_loss,
            model=attack_model,
        )

        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True) # e.g. when no_grad is used in a test_interface
        run_training(
            dataset=dataset,
            seed=0, 
            exp_id=exp_id,
            run_id=attack_run_id, 
            epochs=training_epochs, 
            save_model=True,
            save_model_freq=1,
            train_interface=train_interface,
            use_tboard=True,
            epoch_progressbar=self.is_dataset_large(),
        )
        torch.set_grad_enabled(grad_enabled)

        # ---------------
        # Attack eval
        # ---------------
        models = {}
        attack_model_path = repo_dir("experiments", exp_name(exp_id), "output", attack_run_id)
        for saved_model_name in os.listdir(join(attack_model_path, "model_weights_local")):
            saved_model_name = saved_model_name.replace(".pth", "")
            attack_model = load_model_from_folder(attack_model_path, model_name="model", weights_file_name=saved_model_name)
            models[saved_model_name] = attack_model

        output_path = repo_dir("experiments", exp_name(exp_id), "output", output_run_id)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.metrics = generate_supervised_metrics_for_models(
            model_dict=models,
            dataset=dataset,
            out_path=output_path,
            out_filename=attack_run_id,
            metrics_to_calc=eval_metrics,
        )
        shutil.rmtree(attack_model_path)


    def is_dataset_large(self) -> bool:
        '''
        * Print a progressbar for every epoch
        * Save the encoded data into a folder
        '''
        return False
    
    def get_lr(self):
        return 1e-3

    def get_dataset(self) -> CustomDataset:
        raise NotImplementedError
    
    def get_default_vae_encoder(self):
        raise NotImplementedError
    
    def get_default_vae_decoder(self):
        raise NotImplementedError
    
    def get_attack_model(self, attack_model_id):
        raise NotImplementedError
    
    def _get_defaut_attack_model(self, attack_model_id, out_dim, in_dim=None, image_channels=1, mlp_hidden_channel_override:list[int]|None=None):
        assert mlp_hidden_channel_override == None or attack_model_id == "mlp", "Default model size can only be changed of MLPs."
        flatten_out = out_dim == 1

        if attack_model_id.startswith("resnet"):
            return CustomResNet(size=attack_model_id.replace("resnet", ""), out_dim=out_dim, in_dim=image_channels, flatten_out=flatten_out)
        
        if attack_model_id == "1dResnet":
            return MSResNet(input_channel=image_channels, num_classes=out_dim)

        if in_dim == None:
            in_dim = self.decoded_dim if self.decoded_dim != None else self.img_size**2*image_channels

        if attack_model_id == "linear":
            return MLP(out_channels=out_dim, in_channels=in_dim, hidden_channels=[], flatten_out=flatten_out)
        elif attack_model_id == "mlp":
            hidden_channels = mlp_hidden_channel_override if mlp_hidden_channel_override != None else [64 for _ in range(4)]
            return MLP(out_channels=out_dim, in_channels=in_dim, hidden_channels=hidden_channels, flatten_out=flatten_out, activation="lrelu")

        raise Exception(f"Unknown attack model id: {attack_model_id}")

    def get_attack_model_loss(self):
        raise NotImplementedError
    
    def get_eval_metrics(self):
        raise NotImplementedError
    

class MnistBgAttack(Attack):
    def __init__(self, attack_model_id: Literal['resnet18'] | Literal['resnet34'] | Literal['mlp'] | Literal['linear'] | Literal['1dResnet'], target_label, exp_id, output_run_id, attack_type, dim=None, training_epochs=30, latent_encoder=None, vae_encoder=None, vae_decoder=None, decoded_dim=None, img_size=None, output_filename_prefix="", use_existing_presave_folder=False, target_label_noise_prob=0.):
        self.target_label_noise_prob = target_label_noise_prob
        super().__init__(attack_model_id, target_label, exp_id, output_run_id, attack_type, dim, training_epochs, latent_encoder, vae_encoder, vae_decoder, decoded_dim, img_size, output_filename_prefix, use_existing_presave_folder)

    def get_dataset(self):
        return DatasetMNISTBackground(img_size=self.img_size, batch_size=BATCH_SIZE, num_workers=4, normalization="none", correlation=0, neg_label=0, output_label=self.target_label, target_label_noise_prob=self.target_label_noise_prob)
    
    def get_default_vae_encoder(self):
        vae_encoder_base = load_model_from_folder(repo_dir("experiments", exp_name("0_0"), "results", "2b"), model_name="encoder", weights_file_name=f"encoder_best_loss") 
        return torch.nn.Sequential(vae_encoder_base, VariationalLayer(return_latent_only=True))
    
    def get_default_vae_decoder(self):
        vae_decoder_base = load_model_from_folder(repo_dir("experiments", exp_name("0_0"), "results", "2b"), model_name="decoder", weights_file_name=f"decoder_best_loss")
        return torch.nn.Sequential(vae_decoder_base, View((32, 32)))
    
    def get_attack_model(self, attack_model_id):
        out_dim = 1 if self.target_label == "bg" else 10
        return self._get_defaut_attack_model(attack_model_id, out_dim)
    
    def get_attack_model_loss(self):
        return torch.nn.BCEWithLogitsLoss() if self.target_label == "bg" else torch.nn.CrossEntropyLoss()
    
    def get_eval_metrics(self):
        return {"bin_acc": (0,)} if self.target_label == "bg" else {"class_acc": ()}


class FFHQAttack(Attack):
    def __init__(self, attack_model_id, target_label, exp_id, output_run_id, attack_type, dim=None, training_epochs=30, items_per_bucket=1, latent_encoder=None, vae_encoder=None, vae_decoder=None, decoded_dim=None, img_size=None, output_filename_prefix=""):
        self.soft_intro_vae = load_model_from_folder(repo_dir("experiments", exp_name("04_1"), "results", "1e"), model_name="model", weights_file_name=f"model_300")
        
        super().__init__(attack_model_id, target_label, exp_id, output_run_id, attack_type, dim, training_epochs, items_per_bucket, latent_encoder, vae_encoder, vae_decoder, decoded_dim, img_size, output_filename_prefix)

    def get_dataset(self):
        attributes = {
            "pose": [4, 5, 6],
            "smile": 23,
            "gender": 30,
        }
        return DatasetFFHQAttributes(img_size=self.img_size, batch_size=BATCH_SIZE//2, num_workers=4, attribute_index=attributes[self.target_label])
    
    def get_default_vae_encoder(self):
        return torch.nn.Sequential(self.soft_intro_vae.encoder, VariationalLayer(return_latent_only=True))

    def get_default_vae_decoder(self):
        return self.soft_intro_vae.decoder

    def get_attack_model(self, attack_model_id):
        out_dim = 3 if self.target_label == "pose" else 1
        return self._get_defaut_attack_model(attack_model_id, out_dim, image_channels=3)
    
    def get_attack_model_loss(self):
        return torch.nn.MSELoss() if self.target_label == "pose" else torch.nn.BCEWithLogitsLoss()
    
    def get_eval_metrics(self):
        return {"mse": (), "mae": ()} if self.target_label == "pose" else {"bin_acc": (0, True)} 
    
class ChexpertAttack(Attack):
    def __init__(self, attack_model_id: Literal["resnet18", "resnet34", "mlp", "linear", "1dResnet"], target_label, exp_id, output_run_id, attack_type, dim=None, training_epochs=30, latent_encoder=None, vae_encoder=None, vae_decoder=None, decoded_dim=None, img_size=320, output_filename_prefix="", use_existing_presave_folder=False):
        super().__init__(attack_model_id, target_label, exp_id, output_run_id, attack_type, dim, training_epochs, latent_encoder, vae_encoder, vae_decoder, decoded_dim, img_size, output_filename_prefix, use_existing_presave_folder)
        
    def get_dataset(self):
        return DatasetChexpertSmall(batch_size=BATCH_SIZE//8, num_workers=4, target_label=self.target_label, neg_label_minus=False, binarize_age=True)
    
    def get_default_vae_encoder(self):
        return IdentityModule()

    def get_default_vae_decoder(self):
        return IdentityModule()

    def get_attack_model(self, attack_model_id):
        return self._get_defaut_attack_model(attack_model_id, 
                                             out_dim=1 if self.target_label != None else 18, 
                                             image_channels=1, 
                                             mlp_hidden_channel_override=[512 for _ in range(8)] if attack_model_id == "mlp" else None
                                             )
    
    def get_attack_model_loss(self):
        # label 1: age
        # No label -> all 18 attributes also works with BCE
        return torch.nn.MSELoss() if self.target_label == 1 else torch.nn.BCEWithLogitsLoss()
    
    def get_lr(self):
        return 1e-4
    
    def get_eval_metrics(self):
        if self.target_label == 1:
            return {"mse": (), "mae": ()}
        if self.target_label != None:
            return {"bin_acc": (0, True)}
        
        return {"bin_acc_multi": (0, True)}
    
    def is_dataset_large(self):
        return self.decoded_dim == None # if the decoded version is the default (the img size), the dataset is large. Otherwise, assuming small encoded vectors.
    