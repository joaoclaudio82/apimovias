import os
import optuna
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.callbacks import TFMProgressBar

class OptunaExperiment:
    def __init__(
        self,
        model_cls,
        train_data,
        val_data,
        metric_fn,
        param_space_fn,
        model_kwargs=None,
        log_dir="optuna_logs",
        seed=13,
        n_trials=50,
        optimize_metric="median_wmape",
        early_stopper_monitor="val_loss",
        threshold=30,
    ):
        self.model_cls = model_cls
        self.train_data = train_data
        self.val_data = val_data
        self.metric_fn = metric_fn
        self.param_space_fn = param_space_fn
        self.model_kwargs = model_kwargs or {}
        self.log_dir = log_dir
        self.seed = seed
        self.n_trials = n_trials
        self.optimize_metric = optimize_metric
        self.early_stopper_monitor = early_stopper_monitor
        self.threshold = threshold

        self._log_experiment = os.path.join(self.log_dir, 
                                            self.model_cls.__name__,
                                            self.optimize_metric
                                           )
        
    def _pl_trainer_kwargs(self):
        
        early_stopper = EarlyStopping(
            monitor=self.early_stopper_monitor,         
            patience=5,                 
            min_delta=0.0,             
            mode="min"                  
        )
        
        progress_bar = TFMProgressBar(enable_train_bar_only=True)
    
        pl_trainer_kwargs = {
            "callbacks": [early_stopper, progress_bar],
            "val_check_interval": None,
            "enable_checkpointing": True,  
        }
    
        return pl_trainer_kwargs

  
    def objective(self, trial):
        model_params, fit_params = self.param_space_fn(trial)

        model_params["pl_trainer_kwargs"] = self._pl_trainer_kwargs()
        
        model = self.model_cls(
            **model_params,
            **self.model_kwargs
        )
        model.fit(
            self.train_data.series,
            val_series=self.val_data.series,
            past_covariates=self.train_data.past_covariates,
            future_covariates=self.train_data.future_covariates,
            val_past_covariates=self.val_data.past_covariates,
            val_future_covariates=self.val_data.future_covariates,
            verbose=False,
            **fit_params
        )
        pred_val_series = model.predict(
            n=self.val_data.output_series[0].shape[0],
            series=self.val_data.input_series,
            past_covariates=self.val_data.input_past_covariates,
            future_covariates=self.val_data.output_future_covariates
        )

        wmapes = self.metric_fn(self.val_data.output_series, pred_val_series)
        median_wmape = np.median(wmapes)
        mean_wmape = np.mean(wmapes)
        max_wmape = np.max(wmapes)
        p25_wmape = np.percentile(wmapes, 25)
        p75_wmape = np.percentile(wmapes, 75)
        prop_below_threshold = np.mean(np.array(wmapes) < self.threshold)

        trial.set_user_attr("median_wmape", float(median_wmape))
        trial.set_user_attr("mean_wmape", float(mean_wmape))
        trial.set_user_attr("max_wmape", float(max_wmape))
        trial.set_user_attr("p25_wmape", float(p25_wmape))
        trial.set_user_attr("p75_wmape", float(p75_wmape))
        trial.set_user_attr("prop_below_threshold", float(prop_below_threshold))

        writer = SummaryWriter(log_dir=self._log_experiment)
        writer.add_scalar("median_wmape", median_wmape, trial.number)
        writer.add_scalar("mean_wmape", mean_wmape, trial.number)
        writer.add_scalar("max_wmape", max_wmape, trial.number)
        writer.add_scalar("p25_wmape", p25_wmape, trial.number)
        writer.add_scalar("p75_wmape", p75_wmape, trial.number)
        writer.add_scalar("prop_below_threshold", prop_below_threshold, trial.number)
        writer.flush()
        writer.close()

        metric_map = {
            "median_wmape": median_wmape,
            "mean_wmape": mean_wmape,
            "max_wmape": max_wmape,
            "p25_wmape": p25_wmape,
            "p75_wmape": p75_wmape,
            "prop_below_threshold": -prop_below_threshold,  
        }
        return metric_map[self.optimize_metric]
  
    def run(self):
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(self.objective, n_trials=self.n_trials)
        self.study = study
        return study