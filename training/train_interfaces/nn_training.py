import copy
from typing import Literal

import torch
from models.model_helpers import gaussian_kernel_1d, stacked_distance_matrix
from training.logger import TrainingLogger
from training.train_interfaces.train_interface_base import TrainInterface


class NNTrainInterface(TrainInterface):
    def __init__(self, optimizer, model, l, encoder, max_k, div_loss:Literal["mse", "frac"], min_k=1, smoothing=True, latent_dim_range=(0, 32), smoothing_kernel_size=3) -> None:
        super().__init__(optimizer, None, model)
        assert div_loss in ["mse", "frac"]

        self.l = l
        self.smoothing = smoothing
        self.div_loss = div_loss
        self.encoder = encoder
        self.latent_dim_range = latent_dim_range
        self.k = torch.arange(min_k, max_k, 1, dtype=torch.int)

        self.rec_loss = torch.nn.MSELoss()
        self.lowest_metrics = {"div": torch.tensor(torch.inf), "mse_div": torch.tensor(torch.inf), "rec": torch.tensor(torch.inf)}
        self.best_combined_model = copy.deepcopy(self.model)

        self.options.update({
            "l": l,
            "encoder": encoder,
            "max_k": max_k,
            "min_k": min_k,
            "smoothing": smoothing,
            "latent_dim_range": latent_dim_range,
            "div_loss": div_loss,
            "smoothing_kernel_size": smoothing_kernel_size,
        })

        if smoothing:
            self.smoothing_kernel = gaussian_kernel_1d(sigma=1, num_sigmas=smoothing_kernel_size).view(1, 1, -1)
            self.options["smoothing_kernel"] = self.smoothing_kernel

    def get_learning_models(self):
        return {"base": self.model, "combined": self.best_combined_model}

    def get_frozen_models(self):
        return [self.encoder]

    def call_for_all_tensors(self, func):
        self.k = func(self.k)
        if self.smoothing:
            self.smoothing_kernel = func(self.smoothing_kernel)
    
    def anon_loss_mult_dim(self, a, b, k_mult_a=1, k_mult_ab=1):
        a_distances = stacked_distance_matrix(a, a) + 1e-10
        ab_distances = stacked_distance_matrix(a, b) + 1e-10 # one row for every sample of a

        k_dist_a = torch.sort(a_distances, dim=1)[0][:, (self.k * k_mult_a).to(torch.int)]
        k_dist_ab = torch.sort(ab_distances, dim=1)[0][:, (self.k * k_mult_ab).to(torch.int)]

        if self.smoothing:
            # cut off the outer edges (largest and smallest k)
            # add (and remove) the channel dim, because torch demands it
            k_dist_ab = torch.nn.functional.conv1d(k_dist_ab.unsqueeze(1), weight=self.smoothing_kernel).flatten(1)
            k_dist_a = torch.nn.functional.conv1d(k_dist_a.unsqueeze(1), weight=self.smoothing_kernel).flatten(1)

        if self.div_loss == "mse":
            return torch.mean((k_dist_ab - k_dist_a)**2) # run A
        else:
            return torch.mean(torch.pow(1 - k_dist_a / k_dist_ab, 2)) # run B

    def train(self, dataloader, logger: TrainingLogger, epoch: int) -> dict:
        for (X, label) in dataloader:
            batchsize = X.shape[0]

            X, label = X.to(self.device), label.to(self.device)
            org_latent = self.encoder(X).detach()
            pos_size = label[label == 1].shape[0]
            neg_size = batchsize - pos_size

            for dim in range(*self.latent_dim_range):
                sub_org_latent = org_latent[:, dim]
                sub_anon_latent = self.model[dim](sub_org_latent)

                neg = self.anon_loss_mult_dim(sub_anon_latent[label == -1], sub_anon_latent, k_mult_ab=batchsize/neg_size)
                pos = self.anon_loss_mult_dim(sub_anon_latent[label == 1], sub_anon_latent, k_mult_ab=batchsize/pos_size)
                div = pos + neg
                            
                rec_loss = torch.nn.functional.mse_loss(sub_anon_latent, sub_org_latent)
                loss = self.l * div + (1 - self.l) * rec_loss

                loss.backward()
                self.optimizer[dim].step()
                self.optimizer[dim].zero_grad()

                logger.substep({
                    f"{dim}_div": div,
                    f"{dim}_rec": rec_loss,
                    f"{dim}_loss": loss,
                    "div": div, # metrics with the same name are accumulated over all 32 latents.
                    "rec": rec_loss,
                    "loss": loss,
                })

        combined_metrics = logger.flush_substep_metrics()

        changed = False
        loss_name = "loss"
        for i in range(*self.latent_dim_range):
            key = f"{i}_lowest_{loss_name}"
            if key not in self.lowest_metrics or self.lowest_metrics[key] > combined_metrics[f"{i}_{loss_name}"]:
                changed = True
                self.best_combined_model[i] = copy.deepcopy(self.model[i])
                combined_metrics[key] = combined_metrics[f"{i}_{loss_name}"]
                self.lowest_metrics[key] = combined_metrics[f"{i}_{loss_name}"]
        if changed == True:
            combined_metrics["save_model"] = f"best_{loss_name}"

        return combined_metrics