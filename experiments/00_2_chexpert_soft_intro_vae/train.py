import pathlib
import sys

from datasets.chexpert import DatasetChexpertSmall
from models.soft_intro_vae import CustomSoftIntroVAE
from models.vaes import VariationalLayer
from training.logger import MultiRunLogger
from training.test_interfaces.image_generation import VAETestInterface
from training.train_interfaces.soft_intro_vae import SoftIntroVAETrainInterface
from training.trainer import run_training, seed_everything
import torch

EXP_ID = str(pathlib.Path(__file__).parent.resolve()).split("/")[-1]
BATCH_SIZE = 16
LABEL_DIMS = 1

def train(mean_scale, beta_kl, latent_size):
    seed_everything(0)
    mean_scale, beta_kl, latent_size = float(mean_scale), float(beta_kl), int(latent_size)

    dataset = DatasetChexpertSmall(batch_size=BATCH_SIZE, num_workers=4, target_label=17)

    vae_model = CustomSoftIntroVAE(cdim=1, zdim=latent_size, channels=(64, 128, 256, 512, 512, 32), image_size=320)

    optimizer_e = torch.optim.Adam(vae_model.encoder.parameters(), lr=2e-4)
    optimizer_d = torch.optim.Adam(vae_model.decoder.parameters(), lr=2e-4)

    e_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    d_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

    train_interface = SoftIntroVAETrainInterface(
        optimizer_e=optimizer_e,
        optimizer_d=optimizer_d,
        rec_loss=torch.nn.MSELoss(),
        vae_model=vae_model,
        mean_scale=mean_scale,

        beta_kl=beta_kl,
        beta_rec=1.,

        gamma_r=1e-8,
        beta_neg=256,
        latent_size=latent_size,
        num_label_dims=LABEL_DIMS,
    )
    
    sample_latent = torch.randn((8, latent_size)).repeat((16, 1)) # 8 latents, repeated 16 times for 16 interpolation steps
    offset = torch.arange(-mean_scale - .25, mean_scale + .25, (2 * (mean_scale + .25))/16).repeat_interleave(8) # offsets from -1 to 1 (first 8 latents get a -1 offset, last ~1)
    sample_latent[:, 0:LABEL_DIMS] += offset.unsqueeze(1).repeat(1, LABEL_DIMS) 

    test_interface = VAETestInterface(
        encoder=torch.nn.Sequential(vae_model.encoder, VariationalLayer(return_mean_only=True)),
        decoder=vae_model.decoder,
        dataset=dataset,
        sample_latent=sample_latent,
    )


    run_training(
        dataset=dataset,
        seed=0, 
        exp_id=EXP_ID,
        run_id=f'0_scale{mean_scale}_beta{beta_kl}_latent{latent_size}', 
        epochs=150, 
        eval_freq=1,
        save_model_freq=1,
        save_model=True,
        lr_scheduler=[e_scheduler, d_scheduler],

        use_tboard=True,
        train_interface=train_interface,
        test_interface=test_interface,

        epoch_progressbar=True,
    )

if __name__ == "__main__":
    multi_run_logger = MultiRunLogger(total_runs=1, notification_interval=30)

    _, mean_scale, beta_kl, latent_size = sys.argv
    train(multi_run_logger, int(mean_scale), float(beta_kl), int(latent_size))


