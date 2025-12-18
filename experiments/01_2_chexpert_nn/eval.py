import pathlib

import torch
import os


from evaluation.attacks import ATTACK_TYPES, ChexpertAttack
from misc.helpers import exp_name
from misc.helpers import repo_dir
from models.model_helpers import DimBasedEnsemble
from models.model_loader import load_model_from_folder
from models.vaes import VariationalLayer
from training.trainer import seed_everything


EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]

def eval(run_id, attack_model_id):
    encoder_list = load_model_from_folder(repo_dir("experiments", EXP_ID, "output", run_id), model_name="model", weights_file_name=f"combined_best_loss") 
    vae_model = load_model_from_folder(repo_dir("experiments", exp_name("00_2"), "results", "0c_scale5_beta1_latent64"), model_name="model", weights_file_name=f"model_7")
    encoder = DimBasedEnsemble(encoder_list)

    data_encoded = False
    for target in [7, 9, 14, 17]:
        seed_everything(0)
        ChexpertAttack(
            attack_model_id=attack_model_id,
            target_label=target, 
            exp_id=EXP_ID,
            output_run_id=run_id,
            attack_type=ATTACK_TYPES.encode_all_mask_dim,
            training_epochs=10,
            vae_encoder=torch.nn.Sequential(vae_model.encoder, VariationalLayer(return_latent_only=True)),
            vae_decoder=vae_model.decoder,
            latent_encoder=encoder,
            dim=0,
            img_size=320,
            use_existing_presave_folder=data_encoded,
        )
        data_encoded=True
    


if __name__ == "__main__":
    for folder in os.listdir(repo_dir("experiments", EXP_ID, "output")):
        eval(folder, attack_model_id="mlp")

