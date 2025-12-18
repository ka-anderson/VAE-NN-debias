import logging
import os
from os.path import join
import shutil
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
import torchvision
from tqdm import tqdm
from misc.helpers import data_dir, repo_dir
import torch
import numpy as np
from PIL import Image


from training.logger import ConsoleLogger


DOWNLOAD_FOLDER = data_dir("dataset_downloads")
ENCODED_DATA_DIR = repo_dir("datasets", "encoded_dataset")

class CustomDataset:
    '''
    Wrapper for datasets to add 
    * training/test dataloaders
    * distributed sampler
    * options dict for the logger
    * train_set_size: option to retrieve a smaller training set
    * a method to preload the dataset (with an optional encoder)
    * train_transform (optional): custom transform/augmentation for the training set

    There are three stages to a dataset
    * child dataset: blank pytorch dataset class (Subclasses may, but don't have to, save them as self.child_test_ds),
    * wrapped child dataset: child dataset with adjusted size or sorting (Subclasses cannot override self.wrapped_child_test_ds),
    * dataset: instance of CustomDataset.

    A child of this object must include an _get_child_train_dataset and _get_child_test_dataset method, returning pytorch dataset classes. Do not override any other methods.
    '''
    def __init__(self, ds_id: str, batch_size: int, num_workers=2, pin_memory=True, train_set_size=-1, train_transform=None, training_data_with_index=False):
        assert train_set_size == -1 or train_set_size >= batch_size, f"Batchsize ({batch_size}) can not be larger than the dataset size ({train_set_size})."

        self.ds_id = ds_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_set_size = train_set_size
        self.train_transform = train_transform
        self.training_data_with_index = training_data_with_index

        self.target_device = None

        self.datasetset_preloaded: dict[str, bool] = {}
        self.dataloaders: dict[str, DataLoader] = {}
        self.wrapped_child_datasets: dict[str, Dataset] = {}

        self.training_sampler = None
        self.test_sampler = None

        self.options = {
            "ds_id": self.ds_id,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "train_transform": self.train_transform,
            "train_set_size": self.train_set_size,
            "training_data_with_index": self.training_data_with_index,
        }

    def get_dataloader(self, train, num_replicas=1, rank=0, batchsize_override=None):
        data_key = "train" if train else "test"

        if data_key in self.dataloaders and self.dataloaders[data_key] != None and batchsize_override == None:
            ConsoleLogger.log(f"Using existing {data_key} dataloader.")
            return self.dataloaders[data_key]

        self.init_wrapped_child_dataset(train)

        batchsize = self.batch_size if batchsize_override == None else batchsize_override
        ConsoleLogger.log(f"Creating {data_key} dataloader for batchsize {batchsize} (batchsize_override: {batchsize_override != None})")
        wrapped_child_ds = self.wrapped_child_datasets[data_key]

        if num_replicas == 1:
            dataloader = DataLoader(
                wrapped_child_ds,
                batch_size=batchsize,
                shuffle=train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
                persistent_workers=self.num_workers > 1 # starting at 2 workers, the init at the start of the epoch usually makes the model freeze for a few seconds
            )
        else:
            dataloader = DataLoader(
                wrapped_child_ds,
                sampler=DistributedSampler(wrapped_child_ds, num_replicas=num_replicas, rank=rank, shuffle=False, drop_last=False),
                batch_size=batchsize,
                shuffle=train,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True, # more gpus - more effort to create the workers
            )

        if data_key in self.datasetset_preloaded and self.datasetset_preloaded[data_key] == True:
            self._move_ds_to_target_device(dataloader)
        
        if batchsize_override == None: # only save the dataloader if it matches the Dataset batchsize
            self.dataloaders[data_key] = dataloader

        return dataloader

    def init_wrapped_child_dataset(self, train):
        data_key = "train" if train else "test"

        if data_key not in self.wrapped_child_datasets or self.wrapped_child_datasets[data_key] == None:
            ConsoleLogger.log(f"Creating new {data_key} dataset.")
            self.wrapped_child_datasets[data_key] = self._get_child_dataset(train)

            if train:
                child_ds = self.wrapped_child_datasets[data_key]

                if self.train_set_size > -1:
                    ConsoleLogger.log("Train set size in the new repo untested", level=logging.WARN)
                    if self.train_set_size >= 100:
                        ConsoleLogger.log(f"Using a subset of {self.train_set_size} samples")
                        train_indices = np.random.choice(len(child_ds), size=(self.train_set_size), replace=False)
                    else:
                        ConsoleLogger.log(f"Using a *balanced* subset of {self.train_set_size} samples")
                        train_indices = []
                        all_samples = np.arange(0, len(child_ds))
                        unique_targets = np.unique(child_ds.targets)
                        assert len(unique_targets) <= self.train_set_size, f"train_set_size ({self.train_set_size}) can not be smaller than the number of classes ({len(unique_targets)})."
                        for target in unique_targets:
                            train_indices.extend(np.random.choice(all_samples[child_ds.targets == target], size=(self.train_set_size//len(unique_targets)), replace=False))

                    train_indices = torch.tensor(train_indices)
                    self.wrapped_child_train_ds = Subset(dataset=child_ds, indices=train_indices)  
                else:
                    self.wrapped_child_train_ds = child_ds

                if self.training_data_with_index:
                    ConsoleLogger.log("Index wrapping in the new repo untested", level=logging.WARN)
                    ConsoleLogger.log("Wrapping the training dataset to return index, (item).")
                    self.wrapped_child_train_ds = _DatasetWithIndex(self.wrapped_child_train_ds)

        else: 
            # ConsoleLogger.log(f"{data_key} dataset already initialized.")
            pass
    

    def _get_child_dataset(self, train) -> Dataset:
        '''
        Override this. Needs to use self.train_transform for the training dataset if train_transform is supplied upon initialization.
        '''
        raise NotImplementedError()

    def _move_ds_to_target_device(self, dataloader):
        for sample in dataloader:
            for sample_component in sample:
                if type(sample_component) == torch.Tensor:
                    sample_component = sample_component.to(self.target_device, non_blocking=True)

    def _init_preloading(self, train, target_device):
        '''
        Moving the data to the target device before creating the dataloader usually raises errors
        https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390
        '''
        data_key = "train" if train else "test"
        self.datasetset_preloaded[data_key] = True
        if data_key in self.dataloaders: # key is not in dataloaders if the dataloader was a temporary batchsize override version
            del self.dataloaders[data_key]
        self.target_device = target_device

    def preload_dataset(self, train, target_device="cuda"):
        data_key = "train" if train else "test"
        ConsoleLogger.log(f"Preloading the {data_key} dataset (no encoding).", level=logging.INFO)

        self.init_wrapped_child_dataset(train)
        self.wrapped_child_datasets[data_key] = _PreloadedDataset(self.wrapped_child_datasets[data_key])
        self._init_preloading(train, target_device)
        return self

    def preload_dataset_with_encoder(self, train, encoder, batchsize_override=None, target_device="cuda"):
        data_key = "train" if train else "test"
        ConsoleLogger.log(f"Preloading and encoding the {data_key} dataset.", level=logging.INFO)
        
        self.init_wrapped_child_dataset(train)
        self.wrapped_child_datasets[data_key] = _PreloadedDatasetWithEncoder(self.wrapped_child_datasets[data_key], self.get_dataloader(train, batchsize_override=batchsize_override), encoder, train and self.training_data_with_index)
        self._init_preloading(train, target_device)
        return self
    
    def encode_and_save_dataset(self, encoder, use_existing_folder=False):
        '''
        Use the encoder on the entire dataset, and save the result in a temporary folder (datasets/encoded_data), to be deleted later.

        Useful when there is some costly encoding required, but the dataset too large to be kept in memory.
        Note that it (currently) always encodes both train and test data, to avoid any confusion with two child_datasets for train/test data
        
        use_existing_folder: do not encode and save the dataset, if the dataset has already been created
        '''
        if not use_existing_folder:
            if os.path.exists(ENCODED_DATA_DIR):
                ConsoleLogger.log("Deleting the previous content of the encoded data dir", logging.WARN)
                shutil.rmtree(ENCODED_DATA_DIR) # faster to delete the entire dir than every single file in it
            os.mkdir(ENCODED_DATA_DIR)

        for data_key, is_train in zip(["train", "test"], [True, False]):
            if not use_existing_folder:
                os.mkdir(join(ENCODED_DATA_DIR, data_key))
            self.init_wrapped_child_dataset(is_train)
            self.wrapped_child_datasets[data_key] = _SavedDatasetWithEncoder(self.wrapped_child_datasets[data_key], self.get_dataloader(is_train), encoder, subfolder=data_key, use_existing_folder=use_existing_folder)


class _DatasetWithIndex(Dataset):
    def __init__(self, base_ds) -> None:
        super().__init__()
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)
    
    def __getitem__(self, index):
        return index, self.base_ds.__getitem__(index)

# MARK: Preloaded Datasets
class _PreloadedDataset(Dataset):
    def __init__(self, base_ds) -> None:
        super().__init__()
        self.base_ds = base_ds

        self.items = []
        for i in tqdm(range(len(base_ds))):
            self.items.append(base_ds.__getitem__(i))

        self.options = {
            "base_ds": base_ds,
        }

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    

class _PreloadedDatasetWithEncoder(Dataset):
    def __init__(self, base_ds, base_ds_dataloader, encoder, with_index):
        super().__init__()
        self.with_index = with_index

        item_list, label_list = [], []
        # i = 0
        encoder.to("cuda")
        with torch.no_grad():
            if with_index:
                for _, items, labels in tqdm(base_ds_dataloader):
                    items = encoder(items.to("cuda"))
                    item_list.append(items)
                    label_list.append(labels)
            else:
                for items, labels in tqdm(base_ds_dataloader):
                    items = encoder(items.to("cuda"))
                    item_list.append(items)
                    label_list.append(labels)
                    # i+=1
                    # if i >= 10:
                    #     break

        # self.items = torch.vstack(item_list).cpu() # raises Errors when kept on the gpu before dataloader init
        self.items = torch.cat(item_list).cpu() # raises Errors when kept on the gpu before dataloader init

        if len(label_list[0].shape) > 1:
            self.labels = torch.vstack(label_list)
        else:
            self.labels = torch.hstack(label_list)

        self.options = {
            "base_ds": base_ds,
            "encoder": encoder,
            "with_index": with_index,
        }    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        if not self.with_index:
            return self.items[index], self.labels[index]
        return index, (self.items[index], self.labels[index])
    
class _SavedDatasetWithEncoder(Dataset):
    def __init__(self, base_ds, base_ds_dataloader, encoder, subfolder, use_existing_folder=False) -> None:
        super().__init__()

        self.target_folder = join(ENCODED_DATA_DIR, subfolder)

        item_path_list, label_list = [], []
        if not use_existing_folder:
            encoder.to("cuda")
            ConsoleLogger.log(f"Encoding the dataset and saving it to {self.target_folder}")
        else:
            ConsoleLogger.log(f"Using encoded dataset from {self.target_folder}", logging.INFO)

        with torch.no_grad():
            for batch_index, (items, labels) in tqdm(enumerate(base_ds_dataloader), total=len(base_ds_dataloader)):
                batch_size = labels.shape[0]

                if not use_existing_folder:
                    items = encoder(items.to("cuda"))
                    
                for item_in_batch_index, (item, label) in enumerate(zip(items, labels)):
                    if not use_existing_folder:
                        pil_image = torchvision.transforms.functional.to_pil_image(item)
                        pil_image.save(join(ENCODED_DATA_DIR, subfolder, f"{batch_index * batch_size + item_in_batch_index}.jpeg"), "JPEG")
                        # torchvision.io.write_jpeg(item.to(torch.uint8), join(ENCODED_DATA_DIR, subfolder, f"{batch_index * batch_size + item_in_batch_index}.jpeg"), quality=100) # does not accept 1 channel images
                        # torchvision.utils.save_image(item, join(ENCODED_DATA_DIR, subfolder, f"{batch_index * batch_size + item_in_batch_index}.jpeg")) # slow
                    label_list.append(label)

        if len(label_list[0].shape) > 1:
            self.labels = torch.vstack(label_list)
        else:
            self.labels = torch.hstack(label_list)

        self.options = {
            "base_ds": base_ds,
            "encoder": encoder,
            "subfolder": subfolder,
            "use_existing_folder": use_existing_folder,
        }    

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return Image.open(join(self.target_folder, f"{index}.png")), self.labels[index]