# Soft Intro VAE for FFHQ

Training a VAE based on the training code from the [Soft Intro VAE repository](https://github.com/taldatech/soft-intro-vae-pytorch/tree/main), with the only difference that the kl divergence is computed differently, aiming to incorporate the sensitve label into dimension 0. 

The VAE with the best results (see the results folder) was trained for 300 epochs.