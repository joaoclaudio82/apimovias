import torch
import numpy as np
from torchmetrics import Metric

def agg_median(x):
    return np.median(x)

def agg_mean(x):
    return np.mean(x)

def agg_max(x):
    return np.max(x)

def agg_p25(x):
    return np.percentile(x, 25)

def agg_p75(x):
    return np.percentile(x, 75)

class AggregatedWMAPE(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    AGG_MAP = {
        'median': agg_median,
        'mean': agg_mean,
        'max': agg_max,
        'p25': agg_p25,
        'p75': agg_p75,
    }
    def __init__(self, n, agg='median'):
        super().__init__()
        self.add_state("wmapes", default=[], dist_reduce_fx=None)
        self.n = n
        self.agg = agg.lower()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # if preds.shape[0] == target.shape[0] and preds.shape[0] // self.n == 
        # zero_dim = preds.shape[0]

        if preds.shape[0] != 28 * 128:
            print(preds.shape)
            print(target.shape)

        if preds.dim() == 3:
            preds = preds.squeeze(0)
            target = target.squeeze(0)

        abs_error_sums = torch.sum(torch.abs(target - preds), dim=-1)
        abs_target_sums = torch.sum(torch.abs(target), dim=-1)
        wmapes = abs_error_sums / (abs_target_sums + 1e-8)
        self.wmapes.extend(wmapes.cpu().tolist())
    def compute(self):
        if len(self.wmapes) == 0:
            return torch.tensor(0.0)
        wmapes_np = np.array(self.wmapes)
        if self.agg not in self.AGG_MAP:
            raise ValueError(f"Agregação '{self.agg}' não suportada. Escolha entre {list(self.AGG_MAP.keys())}")
        result = self.AGG_MAP[self.agg](wmapes_np)
        return torch.tensor(result, dtype=torch.float32)
    def __repr__(self):
        return self.__class__.__name__
    def __str__(self):
        return self.__repr__()

class MedianWMAPE(AggregatedWMAPE):
    def __init__(self, n):
        super().__init__(n, agg='median')

class MeanWMAPE(AggregatedWMAPE):
    def __init__(self, n):
        super().__init__(n, agg='mean')

class MaxWMAPE(AggregatedWMAPE):
    def __init__(self, n):
        super().__init__(n, agg='max')

class P25WMAPE(AggregatedWMAPE):
    def __init__(self, n):
        super().__init__(n, agg='p25')

class P75WMAPE(AggregatedWMAPE):
    def __init__(self, n):
        super().__init__(n, agg='p75')