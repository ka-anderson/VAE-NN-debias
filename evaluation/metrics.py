import torch
import pandas as pd
from os.path import join
from datasets.dataset_base import CustomDataset
from tqdm import tqdm

from typing import Dict


def generate_supervised_metrics_for_models(model_dict: Dict[str, torch.nn.Module], dataset: CustomDataset, metrics_to_calc: Dict[str, tuple], out_path, out_filename="stats", save_file=True):
    """
    Calculate the chosen metric modules for each model in the model_dict, for test and trainset. Assumes metric modules are calulated for (model(input), target), whereas input and target are taken from the dataset 
    """
    def calc(dataloader):
        out = []
        for (img, label) in tqdm(dataloader, total=len(dataloader)):
            img, label = img.cuda(), label.cuda()
            for model_key, model in model_dict.items():
                model = model.cuda()
                for metric_key, module in metric_modules.items():
                    metric_result = module(model(img), label, img)
                    if type(metric_result) == list and len(metric_result) > 1:
                        out.extend([{"model": model_key, "metric": f"{metric_key}_{i}", "value": get_value(metric_result_attr)} for i, metric_result_attr in enumerate(metric_result)])
                    else:
                        out.append({"model": model_key, "metric": metric_key, "value": get_value(module(model(img), label, img))})

        df = pd.DataFrame(out)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(df) # replace string model ids with ints if they are numbers
        df = df.groupby(["model", "metric"], as_index=False).agg("mean")

        max = df.groupby("metric", as_index=False).agg("max")
        max["model"] = "max"
        df = pd.concat([df, max])
        
        return df
            
    metric_modules = {}
    for metric_key, parameters in metrics_to_calc.items():
        metric_modules[metric_key] = (METRICS[metric_key](*parameters)).to("cuda")

    with torch.no_grad():
        train_metrics = calc(dataset.get_dataloader(train=True))
        test_metrics = calc(dataset.get_dataloader(train=False))
    train_metrics["data"] = "train"
    test_metrics["data"] = "test"

    combined = pd.concat([train_metrics, test_metrics])

    if save_file:
        combined.to_markdown(join(out_path, f"{out_filename}.md"))
        print(join(out_path, f"{out_filename}.md"))

    return combined

def get_value(x):
    if type(x) == torch.tensor:
        return x.item()
    return x

class BinaryAccuracy(torch.nn.Module):
    '''
    rounds the output to either 1 or neg_label (-1 or 0) and counts how many predictions match the target
    '''
    def __init__(self, neg_label=-1, binarize_target=False) -> None:
        super(BinaryAccuracy, self).__init__()
        self.binarize_target = binarize_target
        self.neg_label = neg_label
        self.split = (1 + neg_label) / 2

    def binarize(self, x, target):
        x[x < self.split] = self.neg_label
        x[x >= self.split] = 1

        if self.binarize_target:
            target[target < self.split] = self.neg_label
            target[target >= self.split] = 1
        
        return x, target

    def forward(self, x, target, x_org=None):
        x, target = self.binarize(x, target)
        return torch.sum(x == target) / target.shape[0]

class BinaryMultiLabelAccuracy(BinaryAccuracy):
    def forward(self, x, target, x_org=None):
        x, target = self.binarize(x, target)
        correct = x == target

        average_over_all = (torch.sum(correct) / (target.shape[0] * target.shape[1])).cpu().numpy()
        average_per_attr = (torch.sum(correct, dim = 0) / target.shape[0]).cpu().numpy()
        return [average_over_all.item(), *average_per_attr]

class ClassificationAccuracy(torch.nn.Module):
    def forward(self, x, target, x_org=None):
        x = torch.argmax(x, dim=1)
        return torch.sum(x == target) / target.shape[0]
    
class ReconstructionMSE(torch.nn.Module):
    def forward(self, x, target, x_org):
        return torch.nn.functional.mse_loss(x, x_org)

class ReconstructionMAE(torch.nn.Module):
    def forward(self, x, target, x_org):
        return torch.nn.functional.l1_loss(x, x_org)

class MSEWrapper(torch.nn.Module):
    def forward(self, x, target, x_org):
        return torch.nn.functional.mse_loss(x, target)
    
class MAEWrapper(torch.nn.Module):
    def forward(self, x, target, x_org):
        return torch.nn.functional.l1_loss(x, target)
    


METRICS = {
    "mse": MSEWrapper,
    "mae": MAEWrapper,
    "bin_acc": BinaryAccuracy,
    "bin_acc_multi": BinaryMultiLabelAccuracy,
    "class_acc": ClassificationAccuracy,
    "rec_mse": ReconstructionMSE,
    "rec_mae": ReconstructionMAE,
}

if __name__ == "__main__":
    test_x = torch.tensor([[0, 1, 1, 1], [0, .9, 1, 1], [1, 1, 1, 1]])
    test_target = torch.tensor([[1, 1, 1, 0], [0, .9, 1, 0], [1, 0, 1, 0]])

    print(test_x)
    print(test_target)

    metric = BinaryMultiLabelAccuracy(neg_label=0, binarize_target=True)
    print(metric(test_x, test_target))