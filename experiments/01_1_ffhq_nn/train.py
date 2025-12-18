import pathlib
from datasets.ffhq import DatasetFFHQAttributes
from misc_helpers.helpers import exp_name, repo_dir

import torch
from models.model_loader import load_model_from_folder
from models.vaes import VariationalLayer
from training.trainer import run_training, seed_everything
from vae_nn_v2_public.training.train_interfaces.nn_training import NNTrainInterface

EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]
BATCH_SIZE = 10000

def train():
    channel_size, depth = 64, 4

    dataset = DatasetFFHQAttributes(img_size=128, batch_size=BATCH_SIZE, num_workers=2, attribute_index=30, neg_label=-1)
    
    vae_encoder_base = load_model_from_folder(repo_dir("experiments", exp_name("04_1"), "results", "1e"), model_name="model", weights_file_name=f"model_300")
    vae_encoder_base.forward_output = "stacked"
    dataset.preload_train_dataset_with_encoder(vae_encoder_base, batch_size_override=512)

    # latent_encoders = torch.nn.ModuleList([MLP(1, 1, hidden_channels=[channel_size for _ in range(depth)], activation="lrelu") for _ in range(64)])
    latent_encoders = load_model_from_folder(repo_dir("experiments", EXP_ID, "output", "A2e"), model_name="model", weights_file_name=f"combined_best_loss") 
    optims = [torch.optim.Adam(latent_encoders[i].parameters(), lr=1e-4) for i in range(64)]

    train_interface = NNTrainInterface(
        optimizer=optims, 
        model=latent_encoders,
        encoder=VariationalLayer(return_latent_only=True, variance_scale=1),
        l=.9999,
        max_k=(BATCH_SIZE//2)*.8,
        smoothing=True,
        smoothing_kernel_size=7,
        div_loss="mse",
        latent_dim_range=(0, 64)
    )

    run_training(
        dataset=dataset,
        seed=0, 
        exp_id=EXP_ID,
        run_id="B0a", 
        # start_from_epoch=550,
        epochs=200000, 
        save_model=True,
        save_model_freq=500,
        eval_freq=100,

        train_interface=train_interface,
        use_tboard=True,
    )


if __name__ == "__main__":
    seed_everything(0)
    train()



