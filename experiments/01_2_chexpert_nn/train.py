import pathlib
from datasets.chexpert import DatasetChexpertSmall
from misc.helpers import exp_name, repo_dir

import torch
from models.model_loader import load_model_from_folder
from models.simple_models import MLP
from models.vaes import VariationalLayer
from training.logger import MultiRunLogger
from training.trainer import run_training, seed_everything
from train_interfaces.nn_training import NNTrainInterface

EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]
BATCH_SIZE = 10000

def train_anon(multi_run_logger, smoothing_kernel_size, depth, width, l):
    seed_everything(0)

    dataset = DatasetChexpertSmall(batch_size=BATCH_SIZE, num_workers=4, target_label=17, neg_label_minus=True)
    
    vae_encoder_base = load_model_from_folder(repo_dir("experiments", exp_name("00_2"), "results", "0c_scale5_beta1_latent64"), model_name="model", weights_file_name=f"model_7")
    vae_encoder_base.forward_output = "stacked"
    dataset.preload_dataset_with_encoder(train=True, encoder=vae_encoder_base, batchsize_override=32)

    latent_encoders = torch.nn.ModuleList([MLP(1, 1, hidden_channels=[width for _ in range(depth)], activation="lrelu") for _ in range(64)])
    # latent_encoders = load_model_from_folder(repo_dir("experiments", EXP_ID, "output", "A2e"), model_name="model", weights_file_name=f"combined_best_loss") 

    optims = [torch.optim.Adam(latent_encoders[i].parameters(), lr=1e-4) for i in range(64)]

    train_interface = NNTrainInterface(
        optimizer=optims, 
        model=latent_encoders,
        encoder=VariationalLayer(return_latent_only=True, variance_scale=1),
        l=l,
        max_k=(BATCH_SIZE//2)*.8,
        smoothing=True,
        smoothing_kernel_size=smoothing_kernel_size,
        div_loss="mse",
    )

    run_training(
        dataset=dataset,
        seed=0, 
        exp_id=EXP_ID,
        run_id=f"A0_l{l}_kernel{smoothing_kernel_size}_depth{depth}_width{width}", 
        epochs=800, 
        save_model=True,
        save_model_freq=-1,
        train_interface=train_interface,

        use_tboard=True,
        multi_run_logger=multi_run_logger,
        epoch_progressbar=False,
    )


if __name__ == "__main__":
    multi_run_logger = MultiRunLogger(total_runs=4, notification_interval=300)

    train_anon(multi_run_logger, smoothing_kernel_size=3, depth=4, width=64, l=.999) 
    train_anon(multi_run_logger, smoothing_kernel_size=3, depth=4, width=64, l=.99) 
    train_anon(multi_run_logger, smoothing_kernel_size=3, depth=8, width=256, l=.999) 
    train_anon(multi_run_logger, smoothing_kernel_size=200, depth=4, width=64, l=.999) 




