import pathlib
import torch


from evaluation.attacks import ATTACK_TYPES, ChexpertAttack
from misc.helpers import repo_dir
from models.model_loader import load_model_from_folder
from models.vaes import VariationalLayer
from training.trainer import seed_everything

EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]

def baseline(attack_model_id):
    for target_label in [7, 9, 14, 17]:
        seed_everything(0)
        ChexpertAttack(
            attack_model_id=attack_model_id,
            target_label=target_label, 
            exp_id=EXP_ID,
            output_run_id="baseline",
            attack_type=ATTACK_TYPES.keep_all,
            training_epochs=10,
        )

def eval(run_id, epoch, attack_model_id):
    vae_model = load_model_from_folder(repo_dir("experiments", EXP_ID, "results", run_id), model_name="model", weights_file_name=f"model_{epoch}") 
    data_encoded=False

    for target_label in [7, 9, 14, 17]:
        seed_everything(0)
        ChexpertAttack(
            attack_model_id=attack_model_id,
            target_label=target_label, 
            exp_id=EXP_ID,
            output_run_id=run_id,
            attack_type=ATTACK_TYPES.keep_all_mask_dim,
            training_epochs=5,
            vae_encoder=torch.nn.Sequential(vae_model.encoder, VariationalLayer(return_latent_only=True)),
            vae_decoder=vae_model.decoder,
            dim=0,
            use_existing_presave_folder=data_encoded,
            output_filename_prefix=f"epoch{epoch}",
        )
        data_encoded=True


if __name__ == "__main__":
    eval("0c_scale5_beta1_latent64", 7, "mlp")
