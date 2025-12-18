import pathlib

import torch

from evaluation.attacks import ATTACK_TYPES, MnistBgAttack
# from evaluation.images import plot_autoenc_results, plot_image_grid
from misc.helpers import repo_dir
from models.model_helpers import IdentityModule
from models.model_loader import load_model_from_folder

from models.vaes import VariationalLayer
from training.trainer import seed_everything

EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]
BATCH_SIZE = 64

def attack(run_id, noise_prob=0.):
    vae_model = load_model_from_folder(repo_dir("experiments", EXP_ID, "results", run_id), model_name="model", weights_file_name=f"model_300")

    for target_label in ["digit", "bg"]:
        seed_everything(0)
        MnistBgAttack(
            attack_model_id="mlp",
            target_label=target_label, 
            exp_id=EXP_ID,
            output_run_id=run_id,
            vae_encoder=torch.nn.Sequential(vae_model.encoder, VariationalLayer(return_latent_only=True)),
            vae_decoder=vae_model.decoder,
            img_size=28,
            target_label_noise_prob=noise_prob,
            output_filename_prefix=f"noise{noise_prob}" if noise_prob > 0 else "", 

            attack_type=ATTACK_TYPES.keep_all_mask_dim,
            dim=0, 
    )
        
def baseline(noise_prob):
    seed_everything(0)
    MnistBgAttack(
        attack_model_id="mlp",
        target_label="digit", 
        exp_id=EXP_ID,
        output_run_id=f"baseline",
        vae_encoder=IdentityModule(),
        vae_decoder=IdentityModule(),
        img_size=28,
        output_filename_prefix=f"noise{noise_prob}", 

        attack_type=ATTACK_TYPES.keep_all,
        target_label_noise_prob=noise_prob,
        dim=0,
    )

if __name__ == "__main__":
    for noise in range(11):
        attack("2e", noise/10)


