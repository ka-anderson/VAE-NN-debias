import pathlib

import torch

from evaluation.attacks import ATTACK_TYPES, MnistBgAttack
from misc.helpers import exp_name, repo_dir
from models.model_helpers import DimBasedEnsemble
from models.model_loader import load_model_from_folder
from models.vaes import VariationalLayer
from training.trainer import seed_everything


EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]

def attack(target_label, run_id, noise_prob=0.):
    encoder_list = load_model_from_folder(repo_dir("experiments", EXP_ID, "results", run_id), model_name="model", weights_file_name=f"combined_best_loss") 
    encoder = DimBasedEnsemble(encoder_list)

    vae_model = load_model_from_folder(repo_dir("experiments", exp_name("00_0"), "results", "2e"), model_name="model", weights_file_name=f"model_300")

    seed_everything(0)
    MnistBgAttack(attack_model_id="mlp",
                  dim=0,
                  exp_id=EXP_ID,
                  output_run_id=run_id,
                  latent_encoder=encoder,
                  target_label=target_label,
                  vae_encoder=torch.nn.Sequential(vae_model.encoder, VariationalLayer(return_latent_only=True, variance_scale=1)),
                  vae_decoder=vae_model.decoder,
                  img_size=28,
                  attack_type=ATTACK_TYPES.encode_all_mask_dim,
                  target_label_noise_prob=noise_prob,
                  output_filename_prefix=f"noise{noise_prob}", 
                  )


if __name__ == "__main__":
    for run_id in ["B0b"]:
        for noise in range(11):
            attack("digit", run_id, noise/10)


