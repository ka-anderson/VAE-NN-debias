
from datasets.dataset_base import CustomDataset

class TestInterface():
    def __init__(self, model, dataset:CustomDataset) -> None:
        self.model = model
        self.dataset = dataset
        self.options = {}

    def to(self, device):
        self.device = device
        if self.model != None:
            self.model.to(device)

    def eval(self, logger):
        '''
        * return a dict containing test metrics, 
        * if "save_model" is in the metric dict, the model will be saved (independently of model saving schedule)
        * metrics ending with "_" will only be passed to tensorboard, not printed
        * metrics ending with "." will be printed and additionally saved to the logfile (if print_logfile is true)
        * can use the logger to write additional information (images, embeddings, histograms) to tboard via logger.tboard_dict
        '''
        raise NotImplementedError()
    


