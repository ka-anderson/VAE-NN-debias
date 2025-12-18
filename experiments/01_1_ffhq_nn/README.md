# NN Density with the Soft Intro Latents: FFHQ

Use the nn density loss on latents computed by the VAE from 00_1.

Our first step is to use the mse loss (`div_loss=mse`) on a fresh MLP. After the training finishes, the trained MLP is then loaded again, to be trained with the fraction nn loss (`div_loss=frac`). The two configuration files for the steps that proved most successfull for us are included. Note that the training did not run for a fixed number of epochs, but was rather stopped manually after we detected no further improvement. 