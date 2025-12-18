import torch
from ..train_interfaces.train_interface_base import TrainInterface
from ..logger import TrainingLogger

class ClassificationTrainInterface(TrainInterface):
    '''
    Simple default training:
    dataloader sample = X, y
    loss = lossfn(model(X), y)
    '''
    def __init__(self, optimizer: torch.optim.Optimizer, loss_fn, model: torch.nn.Module, input_encoder=None) -> None:
        super().__init__(optimizer, loss_fn, model)
        self.using_bce_loss = type(self.loss_fn) == type(torch.nn.BCEWithLogitsLoss())
        self.input_encoder = input_encoder
        self.options["input_encoder"] = input_encoder

    def get_frozen_models(self):
        return [self.input_encoder] if self.input_encoder != None else []

    def train(self, dataloader, logger:TrainingLogger, epoch):

        size = len(dataloader.dataset)
        self.model.train()
        for batch_index, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            if self.using_bce_loss:
                y = y.type(torch.float)

            if self.input_encoder != None:
                with torch.no_grad():
                    X = self.input_encoder(X)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logger.substep({"loss": loss}, batch_index * len(X), size)

        return logger.flush_substep_metrics()
    
class AutoEncoderTrainInterface(TrainInterface):
    def __init__(self, optimizer, loss_fn, encoder, decoder, freeze_decoder=False) -> None:
        super().__init__(optimizer, loss_fn)

        self.encoder = encoder
        self.decoder = decoder
        self.freeze_decoder = freeze_decoder

        self.options.update({
            "encoder": encoder,
            "decoder": decoder,
            "freeze_decoder": freeze_decoder,
        })

    def call_for_all_models(self, func):
        self.encoder = func(self.encoder)
        self.decoder = func(self.decoder)

    def call_for_all_learning_models(self, func):
        if self.freeze_decoder:
            self.encoder = func(self.encoder)
        else:
            self.call_for_all_models(func)

    def get_learning_models(self):
        out = {"encoder": self.encoder}
        if not self.freeze_decoder:
            out["decoder"] = self.decoder
        return out
    
    def get_model_out(self, input):
        return self.decoder(self.encoder(input))

    def train(self, dataloader, logger: TrainingLogger, epoch: int) -> dict:
        size = len(dataloader.dataset)
        self.encoder.train()
        if not self.freeze_decoder:
            self.decoder.train()
            
        for batch_index, (X, _) in enumerate(dataloader):
            X = X.to(self.device)

            encoded = self.encoder(X)
            decoded = self.decoder(encoded)

            loss = self.loss_fn(decoded, X)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logger.substep({"loss": loss}, batch_index * len(X), size)
        
        return logger.flush_substep_metrics()