import torch


class VariationalLayer(torch.nn.Module):
    '''
    To be put in between an encoder (input -> intermediate latent) and a decoder (sampled latent -> output/reconstructed input). 
    Turns them into a VAE, by adding a mean/var split to the encoder and implementing the reparameterization. 

    Expects (stacked) mean and var as input, returns sampled latent.
    '''
    def __init__(self, return_latent_only=False, return_mean_only=False, variance_scale=1) -> None:
        super(VariationalLayer, self).__init__()
        self.return_latent_only = return_latent_only
        self.return_mean_only = return_mean_only
        self.variance_scale = variance_scale

        self.options = {
            "return_latent_only": return_latent_only,
            "return_mean_only": return_mean_only,
            "variance_scale": variance_scale,
            "version": 2,
        }

    def forward(self, intermediate_latent):
        if torch.is_tensor(intermediate_latent): # mean and logvar stacked horizontally. Mean comest first.
            split = intermediate_latent.shape[1] // 2
            mean = intermediate_latent[:, :split]
            logvar = intermediate_latent[:, split:]
        else: # tuple with (mean, logvar)
            mean = intermediate_latent[0]
            logvar = intermediate_latent[1]
        
        if self.return_mean_only:
            return mean

        epsilon = torch.randn_like(logvar, device=logvar.device) 
        latent = mean + torch.exp(self.variance_scale * logvar) * epsilon
        if self.return_latent_only:
            return latent
        return latent, mean, logvar
