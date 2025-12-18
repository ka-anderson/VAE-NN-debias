import json
import os
import pandas as pd
import torchvision
from tqdm import tqdm
import torch
from torch.utils.data import random_split
torchvision.disable_beta_transforms_warning()
from skimage import io
from torch.utils.data import Dataset

from os.path import join, isfile
from os import listdir
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from datasets.dataset_base import CustomDataset, DOWNLOAD_FOLDER

IMAGE_FOLDER_APPENDIX = "_jpg"

class DatasetFFHQ(CustomDataset):
    '''
    Data downloaded from https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq?select=00000.png

    * if to_zero_one is true, images are in [0, 1] (default). Otherwise, they are shifted to [-1, 1]
    '''
    def __init__(self, batch_size: int, img_size: int = 128, num_workers: int = 0, pin_memory=False, to_zero_one=True, train_set_size=-1):
        super().__init__(ds_id='DatasetFFHQ', batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, train_set_size=train_set_size)

        self.img_size = img_size
        if img_size <= 128:
            self.source_img_size = 128
        elif img_size <= 256:
            self.source_img_size = 256
        else:
            self.source_img_size = 0

        self.options.update({
            "img_size": img_size,
            "source_img_size": self.source_img_size,
            "to_zero_one": to_zero_one,
        })
        
        self.dataset = self._base_dataset(to_zero_one)
        generator = torch.Generator().manual_seed(0)
        self.child_test_ds, self.child_train_ds = random_split(self.dataset, [0.05, 0.95], generator=generator)

    def _get_child_dataset(self, train) -> Dataset:
        if train:
            return self.child_train_ds
        return self.child_test_ds

    def _base_dataset(self, to_zero_one) -> Dataset:
        return datasets.ImageFolder(
            root=join(DOWNLOAD_FOLDER, "ffhq", f"{self.source_img_size}{IMAGE_FOLDER_APPENDIX}"), 
            transform=self._get_transform(to_zero_one),
        )

    def _get_transform(self, to_zero_one):
        return_transform = []
        return_transform.append(transforms.ToTensor()) # ToTensor NEEDs to happen before the resizing, otherwise the resize is undone for some reason

        if self.img_size != self.source_img_size:
            return_transform.append(transforms.Resize((self.img_size, self.img_size), antialias=False))

        if not to_zero_one:
            return_transform.append(transforms.Normalize(
                    [0.5 for _ in range(3)],
                    [0.5 for _ in range(3)],
                ))

        return transforms.Compose(return_transform)

# MARK: FFHQ Attr    
class DatasetFFHQAttributes(DatasetFFHQ):
    def __init__(self, batch_size: int, img_size: int = 128, num_workers: int = 0, pin_memory=False, to_zero_one=True, train_set_size=-1, attribute_index=None,
                 return_img=True, return_index=False, neg_label=0):
        '''
        * attribute_index: if != None, return only this attribute as label, not the whole vector
        * neg_label: either -1 or 0. The attributes are normalized to be between [neglabel, 1]
        '''
        assert neg_label in [-1, 0]
        self.attribute_index = attribute_index
        self.neg_label = neg_label
        self.return_img = return_img
        self.return_index = return_index

        super().__init__(batch_size, img_size, num_workers, pin_memory, to_zero_one, train_set_size)

        self.options.update({
            "return_img": return_img,
            "return_index": return_index,
            "neg_label": neg_label,
        })
        if attribute_index != None:
            self.options["attribute_index"] = attribute_index

    def _base_dataset(self, to_zero_one):
        return _DatasetFFHQAttributes(self.source_img_size, self._get_transform(to_zero_one), self.attribute_index, self.return_img, self.return_index, self.neg_label)

class _DatasetFFHQAttributes(Dataset):
    def __init__(self, source_img_size, transform, attribute_index, return_img, return_index, neg_label):

        self.source_img_size = str(source_img_size)
        self.transform = transform

        labels_raw = pd.read_csv(join(DOWNLOAD_FOLDER, "ffhq", "attributes.csv"), index_col=0)
        image_folder = join(DOWNLOAD_FOLDER, "ffhq", f"{self.source_img_size}{IMAGE_FOLDER_APPENDIX}", "0")
        file_ending = os.listdir(image_folder)[0].split(".")[-1] # usually either png or jpg

        self.images = []
        self.indeces = []
        self.labels = []

        print("DatasetFFHQ - Preparing Attribute Dataset")
        for i in tqdm(range(70000)):
            if self.check_existence(i, labels_raw, image_folder, file_ending, attribute_index):
                self.labels.append(torch.tensor(labels_raw.loc[[i]].values[0], dtype=torch.float))
                if return_img:
                    self.images.append(join(image_folder, f"{str(i).zfill(5)}.{file_ending}"))
                if return_index:
                    self.indeces.append(i)
                
        self.labels = torch.vstack(self.labels)

        min, max = torch.min(self.labels, dim=0)[0], torch.max(self.labels, dim=0)[0]
        self.labels = (self.labels - min) / torch.abs(max - min)
        if neg_label == -1:
            self.labels = self.labels * 2 - 1

        if attribute_index != None:
            self.labels = self.labels[:, attribute_index]

    def check_existence(self, index, labels_raw, image_folder, file_ending, attribute_index):
        if index not in labels_raw.index:
            return False
        if not os.path.isfile(join(image_folder, f"{str(index).zfill(5)}.{file_ending}")):
            return False
        return True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(io.imread(self.images[idx])), self.labels[idx]


def save_scaled_version(size):
    # mkdir(join(DOWNLOAD_FOLDER, "ffhq", f"{size}_jpg", "0"))
    for path in tqdm(listdir(join(DOWNLOAD_FOLDER, "ffhq", "org", "0"))):
        image = Image.open(join(DOWNLOAD_FOLDER, "ffhq", "org", "0", path))
        image = image.resize((size, size))

        filename = path.split(".")[0]
        image.save(join(DOWNLOAD_FOLDER, "ffhq", f"{size}_jpg", "0", f"{filename}.jpg"))

# MARK: json to csv attr
def json_to_csv_attributes():
    GLASSES = {"NoGlasses": 0, "ReadingGlasses": 1, "Sunglasses": 2, "SwimmingGoggles": 3}

    result = []
    bug_count = 0
    for file_name in tqdm(listdir(join(DOWNLOAD_FOLDER, "ffhq", "attributes_json"))):
        if isfile(join(DOWNLOAD_FOLDER, "ffhq", "attributes_json", file_name)):
            with open(join(DOWNLOAD_FOLDER, "ffhq", "attributes_json", file_name)) as file:
                img_data = json.load(file)
                if len(img_data) <= 0:
                    bug_count += 1
                    continue

                img_data = img_data[0]

            img_vector = [file_name.split(".")[0]]

            groups = [
                img_data["faceRectangle"], # 0-3
                img_data["faceAttributes"]["headPose"], # 4-6
                img_data["faceAttributes"]["facialHair"], # 7-9
                img_data["faceAttributes"]["emotion"], # 10-17
                img_data["faceAttributes"]["makeup"], # 18-19
                img_data["faceAttributes"]["occlusion"], # 20-22
                ]
            for group in groups:
                img_vector.extend([float(i) for _, i in group.items()])

            single_floats = [
                img_data["faceAttributes"]["smile"], # 23         
                img_data["faceAttributes"]["age"], # 24         
                img_data["faceAttributes"]["blur"]["value"], # 25         
                img_data["faceAttributes"]["exposure"]["value"], # 26         
                img_data["faceAttributes"]["noise"]["value"], # 27        
                img_data["faceAttributes"]["hair"]["bald"], # 28          
                float(img_data["faceAttributes"]["hair"]["invisible"]), # 29
                ]
            img_vector.extend(single_floats)

            img_vector.append(float(img_data["faceAttributes"]["gender"] == "female")) # 30
            img_vector.append(GLASSES[img_data["faceAttributes"]["glasses"]]) # 31

            # hair_color = [float(i["confidence"]) for i in img_data["faceAttributes"]["hair"]["hairColor"]] # 30-35
            # img_vector.extend(hair_color)

            result.append(img_vector)     

    df = pd.DataFrame(result)
    df = df.sort_values(by=0)
    print(bug_count)
    df.to_csv(join(DOWNLOAD_FOLDER, "ffhq", "attributes.csv"), index=False)

if __name__ == "__main__":
    save_scaled_version(128)


