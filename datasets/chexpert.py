'''
Chexpert data, originally downloaded from https://www.kaggle.com/datasets/ashery/chexpert
Labels: 
(Path) 
0: Sex, 
1: Age, 
2: Frontal/Lateral, 
3: AP/PA, 
4: No Finding, 
5: Enlarged Cardiomediastinum,
6: Cardiomegaly,
7: Lung Opacity,
8: Lung Lesion,
9: Edema,
10: Consolidation,
11: Pneumonia,
12: Atelectasis,
13: Pneumothorax,
14: Pleural Effusion,
15: Pleural Other,
16: Fracture,
17: Support Devices

Missing training set labels:
* 0, 1, 2: 0%
* 3: 14%
* 7, 14, 17: ~50%
* others: more than 80%

Lung Opacity
Edema
Pleural Effusion 

'''
import csv
import torch
import torchvision

from training.logger import ConsoleLogger
torchvision.disable_beta_transforms_warning()
from torch.utils.data import Dataset
from os.path import join
from PIL import Image
from datasets.dataset_base import CustomDataset, DOWNLOAD_FOLDER

CHEXPERT_DATA_FOLDER = "chexpert_small"
NUM_LABELS = 18
DEFAULT_IMAGE_SIZE = 320 # same as the original repo

DEFAULT_ONE = [] # for those labels, the replacement for missing labels is 1. For all others, it is zero.
STRING_MAPPING = {
    "Male": 0.,
    "Female": 1.,

    "Frontal": 0.,
    "Lateral": 1.,

    "AP": 0.,
    "PA": 1.,
    "LL": 1.,
    "RL": 1.,
}

class DatasetChexpertSmall(CustomDataset):
    def __init__(self, batch_size: int, target_label: int|None, neg_label_minus=False, num_workers: int = 0, pin_memory=False, train_set_size=-1, img_size:int|None=None, binarize_age=False):
        assert target_label in [None, *range(NUM_LABELS)], f"There are only {NUM_LABELS} target labels (starting with index 0), invalid target: {target_label}"
        super().__init__(ds_id='ChexpertSmall', batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, train_set_size=train_set_size)

        self.target_label = target_label
        self.neg_label_minus = neg_label_minus
        self.binarize_age = binarize_age
        self.img_size = img_size if img_size != None else DEFAULT_IMAGE_SIZE
        self.options.update({
            "target_label": target_label,
            "neg_label_minus": neg_label_minus,
            "img_size": img_size,
            "binarize_age": binarize_age,
        })

    def _get_child_dataset(self, train) -> Dataset:
        return  _CustomCheXpert(target_label_index=self.target_label, train=train, neg_label_minus=self.neg_label_minus, img_size=self.img_size, binarize_age=self.binarize_age)

class _CustomCheXpert(Dataset):
    def __init__(self, target_label_index: int|None, train: bool, neg_label_minus: bool, img_size: int, binarize_age: bool):
        """
        (Roughly) based on https://github.com/Stomper10/CheXpert/blob/master/materials.py
        """
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
        ])
        folder = "train" if train else "valid"

        image_names = []
        labels = []

        with open(join(DOWNLOAD_FOLDER, CHEXPERT_DATA_FOLDER, f"{folder}.csv"), 'r') as f:
            csvReader = csv.reader(f)
            next(csvReader, None) 

            for line in csvReader:
                image_name = line[0]
                image_name = "/".join(image_name.split("/")[1:]) # remove the "CheXpert-v1.0-small" because we are using a custom parent folder name

                if target_label_index != None:
                    if target_label_index == 1 and binarize_age:
                        label = int(self._csv_label_to_training_label(line[target_label_index+1], target_label_index) > .6)
                    else:
                        label = self._csv_label_to_training_label(line[target_label_index+1], target_label_index)
                else:
                    label = [self._csv_label_to_training_label(single_atr, i) for i, single_atr in enumerate(line[1:])]
                    label = torch.tensor([self._csv_label_to_training_label(single_atr, i) for i, single_atr in enumerate(line[1:])])

                if neg_label_minus:
                    label = (label * 2) - 1 # type: ignore

                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        ConsoleLogger.log(f"Loaded CheXpertSmall {'train' if train else 'test'} set with {len(image_names)} samples, target label: {target_label_index if target_label_index else 'all'}.")
        # ConsoleLogger.log(f"Loaded CheXpertSmall {'train' if train else 'test'} set with {len(image_names)} samples, {round(missclassifications/len(image_names) * 100, 1)}% missing lables, target label {target_label_index}.")
    
    def _csv_label_to_training_label(self, label, label_index):
        if len(str(label)) == 0 or label == "-1.0" or label == "Unknown":
            return 1. if label_index in DEFAULT_ONE else 0.
        
        elif label.replace(".", "").isnumeric():
            label = float(label)
            if label_index == 1: # age
                label = label/100
        else:
            label = STRING_MAPPING[label]

        return label
        # return label, True

    def __getitem__(self, index):
        image_name = join(DOWNLOAD_FOLDER, CHEXPERT_DATA_FOLDER, self.image_names[index])
        image = Image.open(image_name)
        # image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.image_names)