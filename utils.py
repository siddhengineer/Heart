# utils.py
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, checkpoint_path=None):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"⚠️  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.checkpoint_path:
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"✅ Saved best model to {self.checkpoint_path}")

class CosineAnnealingWithRestartsScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=2, eta_min=1e-6, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_i = T_0
        self.cycle = 0
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWithRestartsScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr