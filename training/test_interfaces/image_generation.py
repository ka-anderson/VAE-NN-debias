import torch
from datasets.dataset_base import CustomDataset
from training.logger import TrainingLogger
from training.test_interfaces.test_interface_base import TestInterface

class AutoencoderTestInterface(TestInterface):
    '''
    eval returns decoder(encoder(x)), where x is the first row of the dataset batch

    usual case: encoder gets an image and returrns a latent code, decoder gets a latent code and returns an image
    '''
    def __init__(self, encoder, decoder, dataset: CustomDataset) -> None:
        super().__init__(None, dataset)
        self.encoder = encoder
        self.decoder = decoder

        self.train_batch_A, self.train_batch_B = next(iter(dataset.get_dataloader(train=True)))
        self.test_batch_A, self.test_batch_B = next(iter(dataset.get_dataloader(train=False)))

        self.train_batch_A, self.train_batch_B = self.train_batch_A[:8], self.train_batch_B[:8]
        self.test_batch_A, self.test_batch_B = self.test_batch_A[:8], self.test_batch_B[:8]

        self.org_images_sent = False

        self.options.update({
            "dataset": dataset,
            "encoder": encoder,
            "decoder": decoder,
        })

    def to(self, device):
        super().to(device)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.train_batch_A = self.train_batch_A.to(device)
        self.test_batch_A = self.test_batch_A.to(device)
        self.train_batch_B = self.train_batch_B.to(device)
        self.test_batch_B = self.test_batch_B.to(device)

    def eval(self, logger: TrainingLogger):
        images = {}
        if not self.org_images_sent:
            images.update({
                "train_input": self.train_batch_A,
                "test_input": self.test_batch_A
            })
            self.org_images_sent = True

        images["train"] = self.decoder(self.encoder(self.train_batch_A))
        images["test"] = self.decoder(self.encoder(self.test_batch_A))
        logger.tboard_dict(images, "img")

        return {}
    
class VAETestInterface(AutoencoderTestInterface):
    '''
    Additionally to the reconstruction (from the AutoencoderTestInterface), 
    * calculates the output of the decoder for a fixed input latent (to evaluate generated images)
    '''
    def __init__(self, encoder, decoder, dataset: CustomDataset, sample_latent) -> None:
        super().__init__(encoder, decoder, dataset)
        self.sample_latent = sample_latent


    def to(self, device):
        super().to(device)
        self.sample_latent = self.sample_latent.to(device)

    def to_fid_tensor(self, x):
        x = x.to(self.device)
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1)) # convert to "rgb"
        return x.type(torch.uint8)
    
    def eval(self, logger):
        metrics = super().eval(logger)
        logger.tboard_dict({"generated": self.decoder(self.sample_latent)}, "img")
        return metrics