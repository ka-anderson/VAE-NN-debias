
#######################################################################
#
# Mainly from the Soft Intro VAE repo (https://github.com/taldatech/soft-intro-vae-pytorch)
#
#######################################################################


import torch
from models.vaes import VariationalLayer
from training.logger import TrainingLogger
from training.train_interfaces.train_interface_base import TrainInterface


class SoftIntroVAETrainInterface(TrainInterface):
    def __init__(self, optimizer_e, optimizer_d, vae_model, rec_loss, mean_scale, beta_kl, beta_rec, beta_neg, gamma_r, latent_size, num_label_dims=1, image_size=128) -> None:
        super().__init__(None, loss_fn=rec_loss, model=vae_model)

        self.optimizer_e = optimizer_e
        self.optimizer_d = optimizer_d
        self.beta_neg = beta_neg
        self.beta_kl = beta_kl
        self.beta_rec = beta_rec
        self.gamma_r = gamma_r
        self.latent_size = latent_size
        self.mean_scale = mean_scale
        self.num_label_dims = num_label_dims
        self.var_layer = VariationalLayer()
        self.lowest_loss = torch.tensor(torch.inf)
        self.scale = 1 / (image_size ** 2)  # normalize by images size (channels * height * width)


        self.options.update({
            "mean_scale": mean_scale,
            "beta_kl": beta_kl,
            "beta_neg": beta_neg,
            "beta_rec": beta_rec,
            "gamma_r": gamma_r,
            "num_label_dims": num_label_dims,
            "optimizer_e": optimizer_e,
            "optimizer_d": optimizer_d,
            "latent_size": latent_size,
            "image_size": image_size,
        })

    
    def kl_distance(self, log_var, mean, labels=None, reduce="mean"):
        target_mean = torch.zeros_like(mean, device=self.device)
        if self.mean_scale != 0:
            target_mean[:, 0:self.num_label_dims] += self.mean_scale * labels.unsqueeze(1).repeat(1, self.num_label_dims)

        kl = -0.5 * torch.sum(1 + log_var - (target_mean - mean).pow(2) - log_var.exp(), dim = 1)
        if reduce == "mean":
            return torch.mean(kl)
        return kl

    def train(self, dataloader, logger: TrainingLogger, epoch: int) -> dict:
        self.model.train()
        num_samples = len(dataloader)

        for batch, (real_image, labels) in enumerate(dataloader):
            labels = labels.to(self.device)
            batch_size = real_image.shape[0]

            ##### Copied from the Soft Intro VAE repo ####
            noise_batch = torch.randn(size=(batch_size, self.latent_size)).to(self.device)
            real_batch = real_image.to(self.device)

            # =========== Update E ================
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            for param in self.model.decoder.parameters():
                param.requires_grad = False

            fake = self.model.sample(noise_batch)

            real_mu, real_logvar = self.model.encode(real_batch)
            z = reparameterize(real_mu, real_logvar)
            rec = self.model.decoder(z)

            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")

            lossE_real_kl = self.kl_distance(real_logvar, real_mu, labels=labels, reduce="mean")

            rec_mu, rec_logvar, z_rec, rec_rec = self.model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = self.model(fake.detach())

            kl_rec = self.kl_distance(rec_logvar, rec_mu, labels=labels, reduce="none")
            kl_fake = self.kl_distance(fake_logvar, fake_mu, labels=labels, reduce="none")

            loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type="mse", reduction='none')
            while len(loss_rec_rec_e.shape) > 1:
                loss_rec_rec_e = loss_rec_rec_e.sum(-1)
            loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type="mse", reduction='none')
            while len(loss_rec_fake_e.shape) > 1:
                loss_rec_fake_e = loss_rec_fake_e.sum(-1)

            expelbo_rec = (-2 * self.scale * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)).exp().mean()
            expelbo_fake = (-2 * self.scale * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)).exp().mean()

            lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
            lossE_real = self.scale * (self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl)

            lossE = lossE_real + lossE_fake
            self.optimizer_e.zero_grad()
            lossE.backward()
            self.optimizer_e.step()

            # ========= Update D ==================
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.decoder.parameters():
                param.requires_grad = True

            fake = self.model.sample(noise_batch)
            rec = self.model.decoder(z.detach())
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")

            rec_mu, rec_logvar = self.model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = self.model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec = self.model.decode(z_rec.detach())
            rec_fake = self.model.decode(z_fake.detach())

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type="mse", reduction="mean")
            loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type="mse", reduction="mean")

            lossD_rec_kl = self.kl_distance(rec_logvar, rec_mu, labels=labels, reduce="mean")
            lossD_fake_kl = self.kl_distance(fake_logvar, fake_mu, labels=labels, reduce="mean")

            lossD = self.scale * (loss_rec * self.beta_rec + (
                    lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl + self.gamma_r * 0.5 * self.beta_rec * (
                                        loss_rec_rec + loss_fake_rec))

            self.optimizer_d.zero_grad()
            lossD.backward()
            self.optimizer_d.step()

            logger.substep({
                "rec": loss_rec, 
                "kl": lossE_real_kl, 

                "kl_fake": lossD_fake_kl,
                "kl_rec": lossD_rec_kl,

                "exp_elbo_f": expelbo_fake,
                "exp_elbo_r": expelbo_rec,
            }, batch, num_samples,)
        
        metrics = logger.flush_substep_metrics()

        return metrics
    

def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = torch.nn.functional.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = torch.nn.functional.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error
