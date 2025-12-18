import pathlib
from datasets.mnist import DatasetMNISTBackground
from misc.helpers import exp_name, repo_dir

import torch
from models.model_loader import load_model_from_folder
from models.simple_models import MLP
from models.vaes import VariationalLayer
from training.trainer import run_training, seed_everything
from training.train_interfaces.nn_training import NNTrainInterface

EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]
BATCH_SIZE = 8000

def train_anon(latent_dim_range=(0, 32)):
    channel_size, depth = 32, 2

    dataset = DatasetMNISTBackground(img_size=28, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, normalization="none", correlation=0)
    vae_encoder_base = load_model_from_folder(repo_dir("experiments", exp_name("04_0"), "results", "2e"), model_name="model", weights_file_name=f"model_300")
    vae_encoder_base.forward_output = "stacked"
    dataset.preload_dataset_with_encoder(True, vae_encoder_base)

    latent_encoders = torch.nn.ModuleList([MLP(1, 1, hidden_channels=[channel_size for _ in range(depth)], activation="relu") for _ in range(32)])
    # latent_encoders = load_model_from_folder(repo_dir("experiments", EXP_ID, "output", "A2a"), model_name="model", weights_file_name=f"combined_best_loss") 

    optims = [torch.optim.Adam(latent_encoders[i].parameters(), lr=1e-3) for i in range(32)]

    train_interface = NNTrainInterface(
        optimizer=optims, 
        model=latent_encoders,
        encoder=VariationalLayer(return_latent_only=True, variance_scale=1),
        l=.9,
        max_k=(BATCH_SIZE//2)*.9,
        smoothing=False,
        latent_dim_range=latent_dim_range,
        div_loss="frac",
    )

    test_interface = None

    run_training(
        dataset=dataset,
        seed=0, 
        exp_id=EXP_ID,
        run_id="C0a", 
        epochs=200000, 
        save_model=True,
        save_model_freq=500,
        eval_freq=100,

        train_interface=train_interface,
        test_interface=test_interface,

        use_tboard=True,
    )

if __name__ == "__main__":
    seed_everything(0)
    train_anon()



