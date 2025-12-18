import pathlib


from evaluation.attacks import ATTACK_TYPES, FFHQAttack
from misc.helpers import repo_dir
from models.model_helpers import DimBasedEnsemble
from models.model_loader import load_model_from_folder
from training.trainer import seed_everything


EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]

def attack(run_id):

    encoder_list = load_model_from_folder(repo_dir("experiments", EXP_ID, "output", run_id), model_name="model", weights_file_name=f"combined_best_loss") 
    encoder = DimBasedEnsemble(encoder_list)

    for target in ["smile", "pose", "gender"]:
        seed_everything(0)
        FFHQAttack(
            attack_model_id="mlp",
            target_label=target, 
            exp_id=EXP_ID,
            output_run_id=run_id,
            attack_type=ATTACK_TYPES.encode_all_mask_dim,
            latent_encoder=encoder,
            dim=0,
            img_size=128,
        )


if __name__ == "__main__":
    seed_everything(0)
    for l in ["b", "c", "d", "e"]:
        attack(f"B0{l}")

