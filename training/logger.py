import datetime
import inspect
import logging
import os
from os.path import join
import json
import socket
import sys
import threading
import time
from typing import Literal
import requests
import torch
import numbers
import torchvision
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from misc.helpers import dict_list_add


MAX_PRINT_INTERVAL = 2
SCHEDULER_OR_OPTIMIZER_PARAMS = {
    lr_scheduler.ExponentialLR: ["gamma"],
    lr_scheduler.StepLR: ["step_size", "gamma"],
    lr_scheduler.MultiStepLR:["milestones", "gamma"],
    lr_scheduler.CosineAnnealingLR: ["T_max", "eta_min"],

    torch.optim.Adam: ["lr", "betas", "eps", "weight_decay"],
    torch.optim.SGD: ["lr", "momentum", "weight_decay"],
}

'''
https://blog.miguelgrinberg.com/post/the-ultimate-guide-to-python-decorators-part-iii-decorators-with-arguments
same as func = check_tboard(print_warnings)(func) -> func = inner_decorator(func) with print_warning accessible from the outer wrapper
'''
def check_tboard(print_warning=True):
    def inner_decorator(func):
        def wrapper(*args, **kwargs):
            if not args[0].use_tboard: # args[0] is self
                if print_warning:
                    ConsoleLogger.log(f"Tboard not initialized, skipping function {func.__name__}", level=logging.WARNING)
            else:
                return func(*args, **kwargs)
        return wrapper
    return inner_decorator
class ConsoleLogger():
    '''
    Used by both TrainingLogger and other classes for the sole purpose of printing stuff to the console. 
    
    Separate from the TrainingLogger so that it can be used statically, without passing the same logger instance to everything.
    '''
    grey = "\x1b[90;20m"
    green = "\x1b[92;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    reset = "\x1b[0m"

    LOG_FORMAT = {
        # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
        logging.DEBUG: (grey + "[LOG] %(asctime)s - %(source)s - %(message)s" + reset, "%d-%m_%H:%M"),
        logging.INFO: (green + "[INF] %(asctime)s - %(source)s - %(message)s" + reset, "%d-%m_%H:%M"),
        logging.WARNING: (yellow + "[WRN] %(asctime)s - %(source)s - %(message)s" + reset, "%d-%m_%H:%M"),
        logging.ERROR: (red + "[ERR] %(asctime)s - %(source)s - %(message)s" + reset, "%d-%m_%H:%M"),
        "default": ("[%(levelname)3s] %(asctime)s - %(source)s - %(message)s", "%d-%m_%H:%M"),
    }

    @staticmethod
    def log(message, source=None, level=logging.DEBUG):
        if source == None:
            # Try to infer the name of the calling class
            # https://stackoverflow.com/questions/17065086/how-to-get-the-caller-class-name-inside-a-function-of-another-class-in-python
            frame = inspect.stack()[1][0]
            args, _, _, value_dict = inspect.getargvalues(frame)
            if len(args) and args[0] == 'self' and "self" in value_dict:
                source = value_dict["self"].__class__.__name__

        logger = logging.getLogger(f"main.level{level}")
        if len(logger.handlers) == 0:
            handler = logging.StreamHandler(sys.stdout)
            format = ConsoleLogger.LOG_FORMAT[level] if level in ConsoleLogger.LOG_FORMAT.keys() else ConsoleLogger.LOG_FORMAT["default"]
            handler.setFormatter(logging.Formatter(*format))

            logger.addHandler(handler)

            logger.setLevel(level)
            handler.setLevel(level)

        logger.log(level, message, extra={"source": source})
        


class TrainingLogger:
    def __init__(self, folder, verbose=True, multi_run_logger=None, use_epoch_progressbar=False) -> None:
        '''
        * folder: the directory into which all logfiles (options, training logs, tboard) will be saved
        * verbose: ignore all printing restrictions (time, masked values). Print not only the end of every epoch, but also the substeps for every batch
        '''
        self.folder = folder
        self.verbose = verbose

        if not os.path.exists(folder):
            os.makedirs(folder)
        
        self.start_time = time.time()
        self.last_epoch_time = time.time()
        self.epoch = 0
        self.use_tboard = False
        self.logfile_created = False

        self.substep_metrics = {}
        self.substep_metrics_counter = {} # count the added values, to return the mean when flushing the metrics

        self.multi_run_logger = multi_run_logger
        if use_epoch_progressbar:
            self.progressbar = tqdm()
        
    def init_tboard_writer(self):
        '''
        Called after initialization, because torch cannot pass a tboard writer to a distributed training instance
        '''
        if not os.path.exists(self.folder):
            os.makedirs(join(self.folder, "tboard"))

        self.tboard_writer = SummaryWriter(join(self.folder, "tboard"))
        self.use_tboard = True
        self.global_step = 0

    def init_logfile(self):
        '''
        Only if there is data to be written to the log
        '''
        self.log_file_path = join(self.folder, "log.json")
        ConsoleLogger.log(f'writing new log-file: {self.log_file_path}', level=logging.INFO)
        with open(self.log_file_path, 'w') as file:
            json.dump(dict(), file)
        self.logfile_created = True

    def write_options(self, parameters:dict):
        '''
        Once per run: translate all the options dicts from all training parameters into a json file, to be put into self.folder
        '''
        def _to_opt_or_str(object):
            if hasattr(object, "options"):
                out = {"classname": object.__class__.__name__}
                for k, v in object.options.items():
                    out[k] = _to_opt_or_str(v)
                return out
            if type(object) in [tuple, list, nn.Sequential, nn.ModuleList]:
                return [_to_opt_or_str(item) for item in object]
            if type(object) in [dict, nn.ModuleDict]:
                return {key: _to_opt_or_str(value) for key, value in object.items()}
            if isinstance(object, transforms.Compose):
                return [_to_opt_or_str(item) for item in object.transforms]

            if isinstance(object, (torch.optim.lr_scheduler.LRScheduler, torch.optim.Optimizer)):
                return self.get_scheduler_or_optimizer_params(object)
            if isinstance(object, numbers.Number) or object == None:

                return object
            
            return str(object)

        out = {}
        for k, v in parameters.items():
            out[k] = _to_opt_or_str(v)

        with open(join(self.folder, "opt.json"), "w+") as file:
            file.write(json.dumps(out, indent=4))

    def get_scheduler_or_optimizer_params(self, scheduler_or_optimizer):
        assert type(scheduler_or_optimizer) in SCHEDULER_OR_OPTIMIZER_PARAMS, f"Relevant hyperparameters for {scheduler_or_optimizer.__class__.__name__} have not been defined."

        out = {"type": scheduler_or_optimizer.__class__.__name__}
        for key in SCHEDULER_OR_OPTIMIZER_PARAMS[type(scheduler_or_optimizer)]:
            if hasattr(scheduler_or_optimizer, key):
                out[key] = getattr(scheduler_or_optimizer, key)
            else:
                out[key] = scheduler_or_optimizer.param_groups[0][key]

        return out


    def step(self, epoch, metrics:dict=None):
        '''
        metrics: use flush_substep_metrics during training, optionally appending to or manipulating the options dict before returning it in the train() function. Remember to always flush the substep metrics. 

        At the end of every epoch: 
        * add timing and epoch information to the metrics dict, 
        * print the metrics to tensorboard 
        * print the metrics to the console (with a fixed frequency limit)
        * print designated metrics (ending with ".") to a file (very large logfiles will eventually slow down the printing)
        '''
        self.epoch = epoch
        curr_time = time.time()
        time_total = curr_time - self.start_time

        if self.multi_run_logger != None:
            self.multi_run_logger.step(time_total, epoch)

        metrics.update({
            "time_total": time_total,
            "time_per_epoch" : curr_time - self.last_epoch_time,
            "epoch": epoch,
        })
        self.last_epoch_time = curr_time
        metrics = {key: metric.item() if hasattr(metric, "item") else metric for key, metric in metrics.items()}

        if self.verbose: 
            ConsoleLogger.log(f"Epoch {epoch}: {metrics}", source=threading.current_thread().name)
        else:
            last_print = self.last_print if hasattr(self, "last_print") else -MAX_PRINT_INTERVAL
            if curr_time - last_print > MAX_PRINT_INTERVAL: 
                masked_metrics = {key: TrainingLogger.round_metric(value) for key, value in metrics.items() if not key.endswith("_")}
                ConsoleLogger.log(f"Epoch {epoch}: {masked_metrics}", source=threading.current_thread().name)
                self.last_print = curr_time


        logfile_metrics = {key: value for key, value in metrics.items() if key.endswith("")}
        if len(logfile_metrics) > 0:
            if not self.logfile_created:
                self.init_logfile()
                output_file_dict = {}
            else:
                with open(self.log_file_path, 'r') as file: # reading the file is slow for very large files (TODO)
                    output_file_dict = json.load(file)

            for k, v in metrics.items():
                if k.endswith("."):
                    if k not in output_file_dict.keys():
                        output_file_dict[k] = []
                    output_file_dict[k].append(v)
        
                with open(self.log_file_path, 'w') as file:
                    json.dump(output_file_dict, file)

        if self.use_tboard:
            metrics["epoch"] = int(epoch) # epoch could differ from the tboard printing interval
            self.global_step += 1
            self.tboard_writer.add_scalars("metrics", metrics, global_step=self.global_step)

        if hasattr(self, "progressbar"):
            self.progressbar.start_t = time.time()

    @staticmethod
    def round_metric(value):
        if not isinstance(value, numbers.Number):
            return value
        if value > .1:
            return round(value, 4)
        return f"{value:.4e}"

    def substep(self, substep_metrics, batch=0, total=0):
        '''
        For every minibatch: 
        * Accumulate metrics computed for the current substep/minibatch, 
        * Print the metrics in verbose mode.
        * update the (optional) progressbar 
        '''
        for key, value in substep_metrics.items():
            self.substep_metrics = dict_list_add(self.substep_metrics, key, value.detach().item() if type(value) == torch.Tensor else value)
            self.substep_metrics_counter = dict_list_add(self.substep_metrics_counter, key, 1)

        if self.verbose:
            batch = str(batch).zfill(len(str(total)))
            substep_metrics["time"] = time.time() - self.start_time
            metrics = ", ".join([f"{k}: {v}" for k, v in substep_metrics.items()])
            ConsoleLogger.log(f"{threading.current_thread().name} - {batch}/{total}, {metrics}")

        if hasattr(self, "progressbar"):
            self.progressbar.total = total
            self.progressbar.n = batch
            self.progressbar.refresh()

    def flush_substep_metrics(self):
        '''
        Return the accumulated substep metrics and reset them.
        '''
        metrics = {k: v/self.substep_metrics_counter[k] for k, v in self.substep_metrics.items()}
        self.substep_metrics = {}
        self.substep_metrics_counter = {}
        return metrics

    @check_tboard(print_warning=False)
    def tboard_done(self):
        ConsoleLogger.log("Logger - closing tboard writer")
        self.tboard_writer.close()
    
    @check_tboard()
    def tboard_dict(self, dict, data_type:Literal["img", "hist", "emb"]):
        '''
        To print special tensorboard data, other than the default scalar metrics. Embeddings (emb) dict values are expected as another dict with data, (optional) labels and (optional) images.
        '''
        for k, v in dict.items():
            if data_type == "img":
                grid = torchvision.utils.make_grid(v)
                self.tboard_writer.add_image(k, grid, global_step=self.epoch)
            elif data_type == "hist":
                self.tboard_writer.add_histogram(values=v, global_step=self.epoch, tag=k)
            elif data_type == "emb":
                self.tboard_writer.add_embedding(v.data, v.labels, v.images, global_step=self.epoch, tag=f"{k}_{self.epoch}")
            else:
                raise Exception(f"Unknown tboard output type: {data_type}")

        self.tboard_writer.flush()

    @check_tboard()
    def tboard_model_weights(self, model: nn.Module, id=""):
        """
        No longer part of trainer.py, needs to be called manually.
        based on https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md
        """
        def weight_histograms_linear(weights, layer_name):
            flattened_weights = weights.flatten()
            self.tboard_writer.add_histogram(f"weights_{id}_{layer_name}", flattened_weights, global_step=self.epoch)

        def weight_histograms_conv2d(weights, layer_name):
            weights_shape = weights.shape
            num_kernels = weights_shape[0]
            for k in range(num_kernels):
                flattened_weights = weights[k].flatten()
                self.tboard_writer.add_histogram(f"weights_{id}_{layer_name}/kernel_{k}", flattened_weights, global_step=self.epoch)

        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight_histograms_conv2d(module.weight, name)
            elif isinstance(module, nn.Linear):
                weight_histograms_linear(module.weight, name)

class MultiRunLogger():
    '''
    Timing information about more than one run. Not tied to a single run/folder/id or trainer session, no permanent log entries
    '''
    WEBHOOK_STATUS_URL = "https://discord.com/api/webhooks/932579396924100658/geqLn9JAgFznaOpQmz88yqFur2Q--YKYUqqKvFE8tpvhFsAi9BqCk2fa2DLarN5Q6kUx"
    WEBHOOK_FINISHED_URL = "https://discord.com/api/webhooks/929471197459140628/1PM6XQSyRyQ0n8TLJjx7S3HHHen8SH_HPdQ_Qgppv8KFpIZ6I3gmaa-pWDVgaigPBn30"

    def __init__(self, total_runs, notification_interval=10, use_discord=False):
        self.total_runs = total_runs
        self.notification_interval = notification_interval
        self.use_discord = use_discord
        self.current_run_index = 0
        self.current_run_total_epochs = None
        self.exp_and_run_id = ""
        self.init_time = time.time()
        self.last_print_time = self.init_time
    
    def next_run(self, epochs, exp_and_run_id):
        self.exp_and_run_id = exp_and_run_id
        self.current_run_total_epochs = epochs
        self.current_run_index += 1

    def step(self, time_total, epoch):
        if self.current_run_index == self.total_runs and epoch == self.current_run_total_epochs:
            MultiRunLogger.send_to_discord(channel="finished", msg="Training done!")
            return

        curr_time = time.time()
        if epoch == 0 or curr_time - self.last_print_time < self.notification_interval:
            return
        
        self.last_print_time = curr_time
        
        avg_time_per_epoch = time_total/epoch # avg instead of the time for the last epoch for smoother predictions
        time_to_go_current_run = (self.current_run_total_epochs - epoch) * avg_time_per_epoch

        time_to_go_total = time_to_go_current_run + (self.total_runs - self.current_run_index) * self.current_run_total_epochs * avg_time_per_epoch
        time_passed = curr_time - self.init_time

        msg = f"Run {self.current_run_index}/{self.total_runs} ({self.exp_and_run_id}) {round((epoch / self.current_run_total_epochs)*100, 2)}%. Time passed: {datetime.timedelta(seconds=time_passed)}. Predicted time to go: {datetime.timedelta(seconds=time_to_go_total)}"
        ConsoleLogger.log(msg, level=logging.INFO)
        if self.use_discord:
            MultiRunLogger.send_to_discord(channel="status", msg=msg)


    @staticmethod
    def send_to_discord(channel:Literal["status", "finished"], msg):
        url = MultiRunLogger.WEBHOOK_STATUS_URL if channel == "status" else MultiRunLogger.WEBHOOK_FINISHED_URL
        try:
            requests.post(url, {"content": msg, "username": socket.gethostname()})
        except requests.exceptions.HTTPError as _:
            ConsoleLogger.log("Web request to discord failed.", logging.ERROR)
