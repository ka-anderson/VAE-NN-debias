# Implementation for: Nearest-Neighbor Density Estimation for Dependency Suppression

The ability to remove unwanted dependencies from data is crucial in various domains, including fairness, robust learning, and privacy protection. In this work, we propose an encoder-based approach that learns a representation independent of a sensitive variable but otherwise preserving essential data characteristics. Unlike existing methods that rely on decorrelation or adversarial learning, our approach explicitly estimates and modifies the data distribution to neutralize statistical dependencies. To achieve this, we combine a specialized variational autoencoder with a novel loss function driven by non-parametric nearest-neighbor density estimation, enabling direct optimization of independence. We evaluate our approach on multiple datasets, demonstrating that it can outperform existing unsupervised techniques and even rival supervised methods in balancing information removal and utility.

**Please be aware that this repository is an exerpt of our research code, which is made not as a lightweight library, but as a flexible research environment.**

## Setup
To run the experiments in the exact same manner as we did, use the provided docker container:

```
docker build --build-arg currUID=$(id -u) -t vae_nn:latest -f misc/Dockerfile .
docker run -dit --ipc=host --memory=20G --gpus all -v /PATH/TO/REPOSITORY/PARENT/:/data/ --name=vae_nn vae_nn
```

The mounted volume (`/PATH/TO/REPOSITORY/PARENT/`) is expected to point at the folder that contains the repository. Downloaded models and datasets are put into this parent folder, next to the repository.

We ran all experiments on Nvidia RTX 4090 GPUs.

### FFHQ

The FFHQ dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq), and scaled down using the `vae_nn_v2_public/datasets/ffhq.py/save_scaled_version` function. The default implementation expects 128x128 jpg images in `dataset_downloads/ffhq/128_jpg/0`, dataset_downloads being a folder on the same level as the repository folder. This path can be changed in the file `datasets/dataset_base.py`.

For the attributes, we used `vae_nn_v2_public/datasets/ffhq.py/json_to_csv_attributes` on json attributes from [here](https://github.com/DCGM/ffhq-features-dataset).

### CheXpert
The CheXpert dataset has been downloaded from [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert). The default implementation expects the (unzipped) data in `dataset_downloads/chexpert_small`.

The chexpert dataset consists of 224,316 chest radiographs, labeled with 14 observations, additionally including patient age, gender, and radiograph view position. The test dataset of 200 images was manually labeled by certified radiologists, while training labels were automatically extracted from the corresponding radiology reports. Each label can either be positive, negative, uncertain (e.g., "...may represent mild interstitial pulmonary edema") or missing (not mentioned in the report). We follow the common practice of interpreting uncertain and missing observations as negative.

To prevent biases in our utility evaluation, we select observations which (i) are mentioned with certainty in at least 50% of the training labels, (ii) are classified correctly by the original label predictor with at least 95\% accuracy and (iii) are positive in at least 25% of the dataset, to avoid trivial solutions where predicting only negatives yields high accuracy. Our chosen subset of target observations therefore includes "Lung Opacity", "Edema" and "Pleural Effusion".


## Experiments
In the `experiments` folder, there are folders for the different steps and datasets used. To run an experiment, run the train_**.py file in the corresponding folder.

Our training consists of two steps: 
1. Train a VAE (00_*)
2. Train a latent encoder for the VAE latents (01_*).

Each experiment folder also includes corrsponding evaluation scripts. For MNIST, we uploaded the saved weights for the most successfull hyperparameter combinations, most easily loaded with `load_model_from_folder`, as used in e.g. `experiments/00_0_mnist_soft_intro_vae/eval.py` (VAEs for )

## Requirements
(see also `misc/requirements.txt`)

Training:
* [Pytorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)

Evaluation:
* [Pandas](https://pandas.pydata.org/)
* [Plotly](https://plotly.com/)