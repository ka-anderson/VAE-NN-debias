# Soft Intro VAE for Chexpert

Training a VAE based on the training code from the [Soft Intro VAE repository](https://github.com/taldatech/soft-intro-vae-pytorch/tree/main), with the only difference that the kl divergence is computed differently, aiming to incorporate the sensitve label into dimension 0. 

Note that this implementation uses only one color channel for the chexpert images (instead of duplicating the same channel 3 times like most applications). All images are scaled to 320x320. 

The VAE with the best results (see the results folder) was trained for 7 epochs.