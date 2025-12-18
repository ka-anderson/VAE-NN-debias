import logging
import torch
from datasets.dataset_base import CustomDataset
import os
from os.path import join
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist

from misc.helpers import repo_dir
from training.test_interfaces.test_interface_base import TestInterface
from .logger import ConsoleLogger, MultiRunLogger, TrainingLogger
from .train_interfaces.train_interface_base import TrainInterface

def run_training(
    seed: int,
    exp_id: str,
    run_id: str,
    epochs: int,
    train_interface: TrainInterface,
    dataset: CustomDataset,
    start_from_epoch: int = 0, 
    gpus: list = [0],
    debug_distributed: bool = False,
    save_model: bool = True,
    save_model_freq: int = 0, 
    verbose_mode: bool = False,
    use_tboard: bool = False,
    lr_scheduler: torch.optim.lr_scheduler = None,
    test_interface: TestInterface = None,
    eval_freq: int = 0,
    multi_run_logger: MultiRunLogger = None,
    epoch_progressbar = False,
    ):
    '''
    * exp_id: parent folder the experiment
    * run_id: name of this run. Results are saved into exp_id/output/run_id
    * start_from_epoch: if the training process stopped at epoch n (loading model_n.pth), start_from_epoch would be n+1
    * debug_distributed: use the distributed method even for single gpu training
    * save_model_freq: save if epoch % save_model_freq == 0
    '''
    assert len(gpus) == 1, "Distributed training for the current logger version is not supported (fix training metric collection)"

    torch.set_printoptions(profile="full", linewidth=350)

    parameters = locals()
    folder = repo_dir("experiments", exp_id, "output", run_id)

    logger = TrainingLogger(folder, verbose=verbose_mode, multi_run_logger=multi_run_logger, use_epoch_progressbar=epoch_progressbar)
    logger.write_options(parameters)

    if multi_run_logger != None:
        multi_run_logger.next_run(epochs, f"{exp_id}/{run_id}")

    if lr_scheduler != None and type(lr_scheduler) != list:
        lr_scheduler = [lr_scheduler]
        
    # torch.backends.cudnn.enabled = False

    function_args = (len(gpus),                    
            folder,
            gpus, 
            epochs, 
            dataset, 
            train_interface, 
            save_model,
            save_model_freq,
            logger, # the tboard writer of the logger must not be initialized before passing it to the process. For very important pytorch reasons.
            use_tboard,
            start_from_epoch,
            seed,
            lr_scheduler,
            test_interface,
            len(gpus) > 1 or debug_distributed,
            eval_freq,
            epoch_progressbar
            )

    if len(gpus) > 1 or debug_distributed:
        mp.spawn(_train_process, 
                args=function_args,
                nprocs=len(gpus),
                join=True)
    else:
        _train_process(0, *function_args)

def _train_process(rank: int, world_size: int, 
                   folder,
                   gpus: list, 
                   epochs, 
                   dataset: CustomDataset, 
                   train_interface: TrainInterface, 
                   save_model,
                   save_model_freq,
                   logger: TrainingLogger,
                   use_tboard,
                   start_from_epoch,
                   seed,
                   lr_scheduler,
                   test_interface: TestInterface,
                   is_distributed,
                   eval_freq,
                   epoch_progressbar,
                ):
        '''
        The process that is run on a single GPU
        '''

        # torch.autograd.set_detect_anomaly(True)

        # set seeds in every process (https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)
        seed_everything(seed)

        is_main_process = (rank == 0) or (is_distributed == False)

        if is_distributed:
            rank = gpus[rank]
            ConsoleLogger.log(f"Initializing process {rank} (main process: {is_main_process})", "Trainer")
        else:
            ConsoleLogger.log("Initializing single-GPU training", "Trainer")
            
        if is_distributed:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

        train_interface.to(rank, distributed=is_distributed)
        if is_main_process:
            if test_interface != None:
                test_interface.to(rank)

            if use_tboard:
                logger.init_tboard_writer() # initialized writer cannot be passed to a distributed process

        if dataset != None: # some experiments are not using a dataset
            dataloader = dataset.get_dataloader(train=True, num_replicas=world_size, rank=rank)
        else:
            dataloader = None
            
        for epoch in range(start_from_epoch, epochs):

            # train interface step, only pass the logger to the main gpu
            if is_main_process:
                train_out = train_interface.train(dataloader, logger, epoch)
            else:
                train_out = train_interface.train(dataloader, None, epoch)

            if is_distributed:
                torch.distributed.barrier()

            if is_main_process:
                metrics = {f"trainI_{k}": v for k, v in train_out.items()}

                if test_interface != None and eval_freq != 0 and epoch % eval_freq == 0:
                    with torch.no_grad():
                        test_out = test_interface.eval(logger)
                    metrics.update({f"testI_{k}": v for k, v in test_out.items()})

                if lr_scheduler != None:
                    for i, sched in enumerate(lr_scheduler):
                        metrics.update({f"lr_{i}": sched.get_last_lr()[0]})

                # (optional) save the partially trained model at a fixed frequency
                if save_model and (epoch != 0 or save_model_freq == 1) and save_model_freq != 0 and epoch % save_model_freq == 0:
                    ConsoleLogger.log(f"Saving model for epoch {epoch}.", "Trainer")
                    save_model_weights(folder, epoch, train_interface.get_learning_models())

                # (optional) save the partially trained model when an interface identifies a good model
                if "trainI_save_model" in metrics.keys():
                    name = metrics["trainI_save_model"]
                    ConsoleLogger.log(f"Saving model for epoch {epoch}. (new {name} model from the train interface)", "Trainer")
                    save_model_weights(folder, name, train_interface.get_learning_models())
                    del metrics["trainI_save_model"] # tensorboard seems unable to handle string values
                    
                elif "testI_save_model" in metrics.keys():
                    name = metrics["testI_save_model"]
                    ConsoleLogger.log(f"saving model for epoch {epoch}. (new {name} model from the test interface)", "Trainer")
                    save_model_weights(folder, name, train_interface.get_learning_models())
                    del metrics["testI_save_model"]

                logger.step(epoch, metrics)
                 
                if lr_scheduler != None:
                    for sched in lr_scheduler:
                        sched.step()

        if is_main_process and save_model:
            save_model_weights(folder, "final", train_interface.get_learning_models())

        if is_distributed:
            dist.destroy_process_group()

        ConsoleLogger.log(f"Training ({rank}) done!", "Trainer", logging.INFO)

def save_model_weights(base_folder:str, model_id, models:dict):
    '''
    models: {model_name: model} (with model.state_dict returning its weights)
    path of the saves weights will be: {base_folder}/model_weights_local/{model_name}_{model_id}.pth
    '''
    if not os.path.exists(join(base_folder, "model_weights_local")):
        os.makedirs(join(base_folder, "model_weights_local"))
    for name, model in models.items():
        torch.save(model.state_dict(), join(base_folder, "model_weights_local", f'{name}_{model_id}.pth'))

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)



