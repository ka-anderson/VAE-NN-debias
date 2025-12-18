# Soft Intro VAE for Mnist BG

Training a VAE based on the training code from the [Soft Intro VAE repository](https://github.com/taldatech/soft-intro-vae-pytorch/tree/main), with the only difference that the kl divergence is computed differently, aiming to incorporate the sensitve label into dimension 0. 

The file `embedding.ipynb` creates the t-SNE embeddings shown in the paper.