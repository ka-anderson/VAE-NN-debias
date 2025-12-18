import pandas as pd
from evaluation.eval_helpers import get_number, read_pd_markdown
from os.path import join
import os
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np


def multi_target_attack_md_to_dataframe(output_folder, sensitive_index=17, target_indices=[7, 9, 14], filter_attack_model="resnet18", model="max"):
    df_list = []
    for run_id in os.listdir(output_folder):
        run_folder = join(output_folder, run_id)

        for file in os.listdir(run_folder):
            attack_properties = file.replace(".md", "").split("_")
            if not file.endswith(".md") or filter_attack_model not in attack_properties or "target" in file:
                continue
            df = read_pd_markdown(join(run_folder, file))

            sensitive_acc = df[(df["data"] == "test") & (df["metric"] == f"bin_acc_multi_{sensitive_index + 1}") & (df["model"] == model)]["value"].mean() # mean just to get the item, should be one value anyways
            target_acc = 0
            for target_index in target_indices:
                target_acc += df[(df["data"] == "test") & (df["metric"] == f"bin_acc_multi_{target_index + 1}") & (df["model"] == model)]["value"].mean()
            target_acc /= len(target_indices)

            if np.isnan(sensitive_acc).any() or np.isnan(target_acc).any():
                print(f"Skipping {file} (missing target or sensitive acc)")
                continue

            df_list.append({
                "run": run_id,
                "file": file,
                "sensitive": sensitive_acc,
                "target": target_acc,
                "diff": target_acc - sensitive_acc
            })
    return pd.DataFrame(df_list)

def single_target_attack_md_to_dataframe(output_folder, sensitive_index=17, target_indices=[7, 9, 14], filter_attack_model="resnet18", model="max"):
    exp_df_list = []

    for run_folder in os.listdir(output_folder):
        run_df_list = []
        found_target_indices = {index: False for index in target_indices + [sensitive_index]}
        for file in os.listdir(join(output_folder, run_folder)):

            attack_properties = file.replace(".md", "").split("_")
            if not file.endswith(".md") or filter_attack_model not in attack_properties or "target" not in file:
                continue

            target, epoch = None, -1
            for atk_property in attack_properties:
                if "target" in atk_property:
                    target = get_number(atk_property)
                elif "epoch" in atk_property:
                    epoch = get_number(atk_property)

            if target not in target_indices + [sensitive_index]:
                continue

            markdown_df = read_pd_markdown(join(output_folder, run_folder, file))

            run_df_list.append({
                "acc": markdown_df[(markdown_df["data"] == "test") & (markdown_df["model"] == model)]["value"].mean(),
                "target": target,
                "epoch": epoch,
            })
            found_target_indices[target] = True

        if False in found_target_indices.values():
            print(f"Missing target mds in run {run_folder}: {found_target_indices}")
            continue

        run_df = pd.DataFrame(run_df_list)
        run_df = run_df.pivot_table(columns="target", index="epoch", values="acc").reset_index() # new "index" is called "target"...
        run_df = run_df.rename(columns={sensitive_index: "sensitive"})
        run_df["target"] = run_df.loc[:, target_indices].mean(axis=1)
        run_df["diff"] = run_df["target"] - run_df["sensitive"]
        run_df["run"] = run_folder

        exp_df_list.append(run_df)

    return pd.concat(exp_df_list)


def dataframe_to_plots(df, y_values, x_value, color=None, symbol=None, run_id=None):
    '''
    * run_id: None -> all runs
    '''
    if run_id != None:
        df = df[df["run_id"] == run_id]
    
    fig = make_subplots(rows=len(y_values), cols=1, subplot_titles=y_values)
    for i, target_y in enumerate(y_values):
        for trace in px.scatter(df, x=x_value, y=target_y, color=color, symbol=symbol, opacity=.3).data:
            fig.add_trace(trace, row=i+1, col=1)

    fig.update_layout(height=1000)
    return fig