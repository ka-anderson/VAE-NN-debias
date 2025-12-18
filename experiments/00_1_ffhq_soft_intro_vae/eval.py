import json
from os.path import join
import pathlib

import torch

from datasets.ffhq import DatasetFFHQAttributes
from evaluation.attacks import ATTACK_TYPES, FFHQAttack
from evaluation.eval_helpers import torch_to_numpy_image
from evaluation.images import plot_autoenc_results, plot_image_grid
from evaluation.metrics import generate_supervised_metrics_for_models
from misc_helpers.helpers import repo_dir
from models.model_loader import load_model_from_folder

from models.vaes import VariationalLayer
from training.trainer import seed_everything

EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]
BATCH_SIZE = 64

def eval(run_id):
    dataset = DatasetFFHQAttributes(img_size=128, batch_size=BATCH_SIZE, num_workers=4)

    model_path = repo_dir("experiments", EXP_ID, "output", run_id)

    encoder_dict, decoder_dict, combined_ae_dict = {}, {}, {}
    # for i in [-1, 100, "best_fid"]:
    for i in [*sorted((-1, *range(0, 601, 100))), "best_fid"]:
        vae_model = load_model_from_folder(model_path, model_name="model", weights_file_name=f"model_{i}")

        encoder_dict[i] = torch.nn.Sequential(vae_model.encoder, VariationalLayer(return_latent_only=True))
        decoder_dict[i] = vae_model.decoder
        combined_ae_dict[i] = torch.nn.Sequential(encoder_dict[i], decoder_dict[i])

    # Reconstrucion metrics
    generate_supervised_metrics_for_models(
        model_dict=combined_ae_dict,
        dataset=dataset,
        metrics_to_calc={"rec_mse":()},
        out_path=model_path,
    )

    # Reconstruction images
    plot_autoenc_results(
        encoder_dict=encoder_dict,
        decoder_dict=decoder_dict,
        dataset=dataset,
        out_path=model_path,
        pretty_plots=True,
        adjust_to_img_range=True,
    )


def generate(run_id):
    model_path = repo_dir("experiments", EXP_ID, "output", run_id)
    with open(join(model_path, "opt.json")) as opt_file:
        opt = json.loads(opt_file.read())
        latent_dim = opt["train_interface"]["model"]["zdim"]

    interpol_steps, l_min, l_max = 7, -1.4, 1.4
    sample_latent = torch.randn((8, latent_dim)).repeat((8, 1)) # 8 latents, repeated 16 times for 16 interpolation steps
    offset = torch.arange(l_min, l_max+.1, (l_max - l_min)/interpol_steps).repeat_interleave(8) # offsets from -1 to 1 (first 8 latents get a -1 offset, last ~1)
    sample_latent[:, 0] += offset

    decoder = load_model_from_folder(model_path, model_name="model", weights_file_name=f"model_300").decoder.cuda()
    generated = decoder(sample_latent.cuda())
    generated = torch_to_numpy_image(generated)
    plot_image_grid(generated, ncols=8, nrows=interpol_steps, pretty_plots=True, output_path=join(model_path, "out_generated_inter.png"))

def attack(run_id, epoch):
    vae_model = load_model_from_folder(repo_dir("experiments", EXP_ID, "results", run_id), model_name="model", weights_file_name=f"model_{epoch}")

    for target in ["smile", "gender"]:
        seed_everything(0)
        FFHQAttack(
            attack_model_id="mlp",
            target_label=target, 
            exp_id=EXP_ID,
            output_run_id=run_id,
            img_size=128,
            vae_encoder=torch.nn.Sequential(vae_model.encoder, VariationalLayer(return_latent_only=True)),
            vae_decoder=vae_model.decoder,

            attack_type=ATTACK_TYPES.keep_all_mask_dim,
            dim=0,
            output_filename_prefix=f"epoch{epoch}"
        )
        
    # Baseline
    # FFHQAttack(
    #     attack_model_id="mlp",
    #     target_label="gender", 
    #     exp_id=EXP_ID,
    #     output_run_id=run_id,
    #     img_size=128,
    #     vae_encoder=IdentityModule(),
    #     vae_decoder=IdentityModule(),
    #     attack_type=ATTACK_TYPES.keep_all,
    #     output_filename_prefix="baseline",
    # )

if __name__ == "__main__":
    seed_everything(0)
    for epoch in [300]:
        attack("1e", epoch)


