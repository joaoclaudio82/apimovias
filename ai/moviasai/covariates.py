import numpy as np
import pandas as pd

from darts import TimeSeries 
from darts.dataprocessing.transformers import Scaler

from functools import reduce


def week_of_month(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.array([int(np.ceil((dt.day + dt.replace(day=1).weekday()) / 7.0)) for dt in idx])

def is_weekday(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.array([1 if dt.weekday() < 5 else 0 for dt in idx])

def default_encoders(supports_future_covariates=True):
    if supports_future_covariates:
        types = ['past', 'future']
        covariates_type = 'future' 
    else: 
        types = ['past']
        covariates_type = 'past'

    
    return {
        'cyclic': {covariates_type: ['dayofweek']},
        'datetime_attribute': {covariates_type: ['dayofweek']},
        'position': {t: ['relative'] for t in types},
        'custom': {
            covariates_type: [
                week_of_month, 
                is_weekday
            ]
        },
        'transformer': Scaler()
    }


class Covariate(object):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def extract(self, ts: TimeSeries) -> TimeSeries:
        raise NotImplementedError

        
class MovingAvg(Covariate):
    def __init__(
        self,
        prefix: str,
        history_size: int,
        freq: int
    ):
        super().__init__(prefix)  
        self.history_size = history_size
        self.freq = freq

    def _get_moving_avg(self, window, idx, values):
        ma = values.rolling(window, min_periods=1).mean()
        return TimeSeries.from_times_and_values(
            idx, ma.values, columns=[f"{self.prefix}_{window}"]
        )

    def extract(self, ts: TimeSeries) -> TimeSeries:
        idx = ts.time_index
        values = pd.Series(ts.values().flatten())
            
        n_weeks = self.history_size // self.freq
        ma_features = [self._get_moving_avg(w * self.freq, idx, values) for w in range(1, n_weeks + 1)]
        covariates = reduce(lambda x, y: x.stack(y), ma_features)
        return covariates
    

    