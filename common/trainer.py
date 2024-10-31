import sys
import os
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
import torch
import pandas as pd
import random
import numpy as np


def set_seed(seed=42):
    """ Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)

    # deterministic algorithm behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, model_dir, model_name):
    path = os.path.join(model_dir, model_name + ".pth")
    Path(os.path.dirname(path)).mkdir(exist_ok=True)

    # if not os.path.exists(path):
    torch.save(model.state_dict(), path)
    print(f">> {model_name}.pth has been saved in {model_dir}")


def save_history(history, model_dir, model_name):
    path = os.path.join(model_dir, model_name + ".csv")
    Path(os.path.dirname(path)).mkdir(exist_ok=True)

    # if not os.path.exists(path):
    df = pd.DataFrame(history)
    df.to_csv(path)
    print(f">> {model_name}.csv has been saved in {model_dir}")


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_model = None
        self.counter = 0
        self.triggered = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            model.load_state_dict(self.best_model)
            self.triggered = True
            return True

        return False


class Trainer:
    def __init__(self, model, optimizer, loss_fn, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = {"loss": loss_fn}
        if metrics is not None:
            self.metrics.update(metrics)

        self.device = next(model.parameters()).device
        self.kwargs = dict(file=sys.stdout, leave=False, ascii=True, unit=" batch", ncols=120)

    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        return {name: func(pred, y).item() for name, func in self.metrics.items()}

    @torch.no_grad()
    def test_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model(x)
        return {name: func(pred, y).item() for name, func in self.metrics.items()}

    def fit(self, train_loader, n_epochs, valid_loader=None, stopper=None, scheduler=None):
        history = {name: [] for name in self.metrics}
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in self.metrics})

        def update(history, res, cnt):
            for name, value in res.items():
                history[name].append(value / cnt)

        for e in range(n_epochs):
            epoch = str(e + 1).rjust(len(str(n_epochs)), ' ')
            epoch = f"Epoch[{epoch}/{n_epochs}]"

            ## Training
            self.model.train()
            res = {name: 0 for name in self.metrics}
            with tqdm(train_loader, postfix="training", **self.kwargs) as pbar:
                for i, (x, y) in enumerate(pbar):
                    res_step = self.train_step(x, y)

                    desc = ""
                    for name in self.metrics:
                        res[name] += res_step[name]
                        desc += f" {name}: {res[name]/(i + 1):.3f}"
                    pbar.set_description(epoch + desc)

            ## Learning rate scheduling
            if scheduler is not None:
                scheduler.step()

            if valid_loader is None:
                print(epoch + desc)
                update(history, res, i + 1)
                continue

            ## Validation
            self.model.eval()
            val_res = {f"val_{name}": 0 for name in self.metrics}
            with tqdm(valid_loader, postfix="validation", **self.kwargs) as pbar:
                for j, (x, y) in enumerate(pbar):
                    res_step = self.test_step(x, y)

                    val_desc = ""
                    for name in self.metrics:
                        val_res[f"val_{name}"] += res_step[name]
                        val_desc += f" val_{name}: {val_res[f'val_{name}']/(j + 1):.3f}"
                    pbar.set_description(epoch + val_desc)

            print(epoch + desc + " |" + val_desc)
            update(history, res, i + 1)
            update(history, val_res, j + 1)

            ## Early Stopping
            if stopper is not None:
                val_loss = val_res["val_loss"] / (j + 1)
                stopper.step(val_loss, self.model)
                if stopper.triggered:
                    print(f">> Early stopped! (best_loss: {stopper.best_loss:.3f})")
                    break
        return history

    def evaluate(self, test_loader):
        self.model.eval()
        test_res = {f"test_{name}": 0 for name in self.metrics}
        with tqdm(test_loader, postfix="evaluation", **self.kwargs) as pbar:
            for j, (x, y) in enumerate(pbar):
                res_step = self.test_step(x, y)

                test_desc = ">>"
                for name in self.metrics:
                    test_res[f"test_{name}"] += res_step[name]
                    test_desc += f" test_{name}: {test_res[f'test_{name}']/(j + 1):.3f}"
                pbar.set_description(test_desc)
        print(test_desc)
        return {name: value/(j + 1) for name, value in test_res.items()}

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        return self.model(x)
        
if __name__ == "__main__":

    pass