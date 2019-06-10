# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/5 下午2:29
"""
import time

import numpy as np
import torch
from tqdm import tqdm


class ModelTrainer(object):
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device="cpu"):
        """model trainer
        :param model: `torch.nn.Module` subclass.
        :param loss_fn: loss function.
        :param optimizer: training optimizer.
        :param scheduler:
        :param device: str. cpu or cuda
        """
        if device == "cuda":
            model = model.cuda()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_one_epoch(self, dataloader, y_true="output", verbose_step=False):
        avg_loss = 0.0
        self.model.train()
        for x_batch in tqdm(dataloader, disable=not verbose_step):
            y_batch = x_batch[-1]
            x_batch = x_batch[:-1]
            y_pred = self.model(*x_batch)
            self.optimizer.zero_grad()
            if y_true == "input":
                loss = self.loss_fn(*x_batch, y_true=y_batch)
            else:
                loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.step()
            avg_loss += loss.item() / len(dataloader)
        return avg_loss

    def predict(self, dataloader, has_label=False):
        """predict from dataloader
        :param dataloader: `torch.utils.data.Dataloader`. input x to predict
        :param has_label: bool. whether has label in dataloader
        :return: `np.array`.
        """
        self.model.eval()
        prediction = []
        for x_batch in dataloader:
            if has_label:
                x_batch = x_batch[:-1]
            y_pred = self.model(*x_batch)
            if not isinstance(y_pred, torch.Tensor):
                prediction.extend(y_pred)
                continue
            if self.device == "cuda":
                y_pred = y_pred.cpu()
            prediction.extend(y_pred.detach().numpy())
        return np.array(prediction)

    def train(self, train_dataloader, val_dataloader=None, epochs=1, verbose=0, metrics=None, y_true="output",
              verbose_step=False):
        """train model
        :param train_dataloader: train set dataloader.
        :param val_dataloader: validation set dataloader.
        :param epochs: int.
        :param verbose: bool. whether print log message. default False
        :param metrics: list or func. default None. if given, then the message with print in the log message
        :param y_true: str. compute loss when train. {"input", "output}
            if "input", then use X to compute loss. else, use model output.
        :param verbose_step: bool.
        :return:
        """
        metrics = metrics or []
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        for epoch in range(epochs):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_dataloader, y_true=y_true, verbose_step=verbose_step)
            log_msg = f"Epoch: {epoch + 1}, train loss: {train_loss:.4f}"
            if val_dataloader is not None:
                val_pred = self.predict(val_dataloader, has_label=True)
                # t_val_pred = torch.tensor(val_pred, dtype=torch.long, device=self.device)
                val_loss = self.loss_fn(*val_dataloader.dataset.tensors[:-1],
                                        y_true=val_dataloader.dataset.tensors[-1]).item()
                log_msg += f", val loss: {val_loss:.4f}"
                if self.device == "cuda":
                    y_val = val_dataloader.dataset.tensors[-1].cpu()
                else:
                    y_val = val_dataloader.dataset.tensors[-1]
                y_val = y_val.detach().numpy()
                for metric in metrics:
                    metric_val = metric(y_pred=val_pred, y_true=y_val)
                    log_msg += f", {metric.__name__}: {metric_val:.4f}"
            use_seconds = time.time() - t0
            log_msg += f", time: {use_seconds:.1f}"
            if verbose:
                print(log_msg)
        return self.model
