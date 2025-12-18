from tqdm import tqdm
import torch
import plotly.graph_objects as go
import numpy as np


def generate_interactive_1D_histogram(img_encoder, latent_encoder_dict, dataloader, device="cuda"):
    '''
    * img_encoder: input is the original image, output needs to be scalar (like one dim of a vae latent)
    * latent_encoder_dict: input is the (scalar) output of the img_encoder. Output needs to be scalar as well. Can be empty.
    '''
    org_latents_flat, labels_flat = [], []
    encoded_latents_flat = {key: [] for key in latent_encoder_dict.keys()}

    img_encoder.to(device)
    latent_encoder_dict = {key: value.to(device) for key, value in latent_encoder_dict}

    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.to(device)
            latents = img_encoder(img)
            for key, latent_encoder in latent_encoder_dict:
                encoded_latents_flat[key].extend(latent_encoder(latents).tolist())

            labels_flat.extend(label.tolist())
            org_latents_flat.extend(latents.tolist())

    org_latents_flat = np.array(org_latents_flat)
    labels_flat = np.array(labels_flat)
    encoded_latents_flat = {key:np.array(value) for key, value in encoded_latents_flat.items()}

    bin_min, bin_max = np.min(org_latents_flat), np.max(org_latents_flat)
    bin_ranges = np.arange(bin_min, bin_max, (bin_max - bin_min)/100)

    org_bins, _ = np.histogram(org_latents_flat, bin_ranges, density=True)
    org_bins_neg, _ = np.histogram(org_latents_flat[labels_flat==-1], bin_ranges, density=True)
    org_bins_pos, _ = np.histogram(org_latents_flat[labels_flat==1], bin_ranges, density=True)


    for key, encoded_latents in encoded_latents_flat.items():
        out_bins_neg, _ = np.histogram(encoded_latents[labels_flat==-1], bin_ranges, density=True)
        out_bins_pos, _ = np.histogram(encoded_latents[labels_flat==1], bin_ranges, density=True)

        frames += [go.Frame(
            data=[
                go.Bar(y=out_bins_neg, name="neg"),
                go.Bar(y=out_bins_pos, name="pos"),
                go.Bar(y=org_bins_neg, name="org_neg"),
                go.Bar(y=org_bins_pos, name="org_pos"),
                go.Bar(y=out_bins_pos - out_bins_neg, name="pos-neg"),
                go.Bar(y=out_bins_neg - org_bins, name="neg-org"),
                go.Bar(y=out_bins_pos - org_bins, name="pos-org"),
                ],
            name=key,
            )]


    layout_frame = go.Frame(
        data=[
            # go.Bar(y=org_bins_neg, x=bin_ranges[:-1], marker_color="firebrick", width=.04, opacity=1, offset=0, name="neg"),
            # go.Bar(y=org_bins_pos, x=bin_ranges[:-1], marker_color="steelblue", width=.04, opacity=1, offset=.04, name="pos"),
            # go.Bar(y=org_bins_neg, x=bin_ranges[:-1], marker_color="yellowgreen", width=.04, opacity=.3, offset=0, name="org_neg", visible="legendonly"),
            # go.Bar(y=org_bins_pos, x=bin_ranges[:-1], marker_color="green", width=.04, opacity=.3, offset=.04, name="org_pos", visible="legendonly"),
            # go.Bar(x=bin_ranges[:-1], marker_color="orange", offset=0, name="pos-neg", visible="legendonly"),
            # go.Bar(x=bin_ranges[:-1], marker_color="firebrick",  offset=0, name="neg-org", width=.04, visible="legendonly"),
            # go.Bar(x=bin_ranges[:-1], marker_color="steelblue",  offset=0.04, name="pos-org", width=.04, visible="legendonly"),
            go.Bar(y=org_bins_neg, x=bin_ranges[:-1], marker_color="firebrick", opacity=1, name="neg"),
            go.Bar(y=org_bins_pos, x=bin_ranges[:-1], marker_color="steelblue", opacity=1,  name="pos"),
            go.Bar(y=org_bins_neg, x=bin_ranges[:-1], marker_color="yellowgreen", opacity=.3, name="org_neg", visible="legendonly"),
            go.Bar(y=org_bins_pos, x=bin_ranges[:-1], marker_color="green", opacity=.3, name="org_pos", visible="legendonly"),
            go.Bar(x=bin_ranges[:-1], marker_color="orange", name="pos-neg", visible="legendonly"),
            go.Bar(x=bin_ranges[:-1], marker_color="firebrick", name="neg-org", visible="legendonly"),
            go.Bar(x=bin_ranges[:-1], marker_color="steelblue", name="pos-org", visible="legendonly"),
            ])
    frames = []
    
    # https://stackoverflow.com/questions/74526203/how-to-create-a-plotly-animation-from-a-list-of-figure-objects
    # https://stackoverflow.com/questions/69867334/multiple-traces-per-animation-frame-in-plotly
    fig = go.Figure(data=[layout_frame.data[i] for i in range(7)], layout=layout_frame.layout, frames=frames)

    sliders = [{"steps": [{
                            "args": [[f.name]],
                            "label": f.name,
                            "method": "animate",
                        } for f in fig.frames
                    ],
                }]

    fig.update_layout(sliders=sliders)
    return fig

