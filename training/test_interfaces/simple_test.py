from datasets.dataset_base import CustomDataset
from evaluation.metrics import BinaryAccuracy, BinaryMultiLabelAccuracy, ClassificationAccuracy
from training.test_interfaces.test_interface_base import TestInterface
import torch


class SimpleTestInterface(TestInterface):
    '''
    acc = distance(model_out, target)
    '''
    def __init__(self, model, dataset: CustomDataset) -> None:
        super().__init__(model, dataset)
        self.test_dataloader = dataset.get_dataloader(train=True)
        self.train_dataloader = dataset.get_dataloader(train=False)

        self.best_test_acc = self.initial_best()

    def eval(self, _):
        def calc(dataloader):
            acc = torch.zeros(self.get_label_size(), device=self.device, dtype=torch.float)
            for img, label in dataloader:
                img, label = img.cuda(), label.cuda()
                pred = self.model(img)
                acc += self.distance(pred, label, img)
            return (acc / len(dataloader)).detach().cpu()

        test_acc = calc(self.test_dataloader)
        train_acc = calc(self.train_dataloader)
        metrics = self.metric_array_to_dict(test_acc, "acc_test") | self.metric_array_to_dict(train_acc, "acc_train")

        if self.is_better(old_value=self.best_test_acc, new_value=test_acc):
            self.best_test_acc = test_acc
            metrics = metrics | self.metric_array_to_dict(test_acc, "best_test_acc.")
            # metrics["best_test_acc."] = test_acc
        return metrics

    def distance(self, pred, target, input):
        raise NotImplementedError()
    
    def initial_best(self):
        return torch.zeros(self.get_label_size())
    def is_better(self, old_value, new_value):
        return new_value > old_value
    def get_label_size(self):
        return 1
    
    def metric_array_to_dict(self, metric, key):
        if metric.shape[0] == 1:
            return {key: metric}
        return {f"{key}_{i}": metric[i] for i in range(metric.shape[0])}


class RegressionTestInterface(SimpleTestInterface):
    def distance(self, pred, target, input):
        return torch.abs(pred - target).mean()
    def initial_best(self):
        return torch.inf
    def is_better(self, old_value, new_value):
        return new_value < old_value

    
class ReconstructionTestInterface(SimpleTestInterface):
    def distance(self, pred, target, input):
        return torch.abs(pred - input).mean()
    def initial_best(self):
        return torch.inf
    def is_better(self, old_value, new_value):
        return new_value < old_value


class ClassificationTestInterface(SimpleTestInterface):
    def __init__(self, model, dataset) -> None:
        super().__init__(model, dataset)
        self.acc = ClassificationAccuracy()

    def distance(self, pred, target, input):
        return self.acc(pred, target)

class BinaryClassificationTestInterface(SimpleTestInterface):
    def __init__(self, model, dataset, neg_label, binarize_target=False) -> None:
        super().__init__(model, dataset)
        self.bin_acc = BinaryAccuracy(neg_label=neg_label, binarize_target=binarize_target)
        self.options.update({
            "neg_label": neg_label,
            "binarize_target": binarize_target,
        })

    def distance(self, pred, target, input):
        return self.bin_acc(pred, target)
    
class BinaryMultiLabelClassificationTestInterface(SimpleTestInterface):
    def __init__(self, model, dataset, neg_label, label_size, binarize_target=False) -> None:
        self.label_size = label_size

        super().__init__(model, dataset)
        self.bin_acc = BinaryMultiLabelAccuracy(neg_label=neg_label, binarize_target=binarize_target)

        self.options.update({
            "neg_label": neg_label,
            "binarize_target": binarize_target,
            "label_size": label_size,
        })

    def distance(self, pred, target, input):
        return torch.tensor(self.bin_acc(pred, target), device=self.device)
    
    def is_better(self, old_value, new_value):
        return (new_value > old_value).any()

    def get_label_size(self):
        return self.label_size