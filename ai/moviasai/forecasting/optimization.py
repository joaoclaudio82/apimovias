"""
Pipeline de otimização de hiperparâmetros com Optuna.

Foco: minimizar risco de atraso na decisão de manutenção (k=2 semanas).

Cada trial executa:
1. Sugere hiperparâmetros (alpha_daily, lr, hidden_dim, conv_filters, dropout)
2. Treina o modelo com early stopping
3. Avalia no conjunto de validação
4. Retorna P90_k2 + 0.5 * MAE_k2 como métrica objetivo
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import optuna
import pytorch_lightning as lightning

from moviasai.data import (
    ProfileDataset,
    ProfileDatasetGenerator,
    ProfileDatasetNormalizer,
)
from moviasai.data.utils import train_test_split
from moviasai.forecasting.training_pipeline import TrainingPipeline
from moviasai.profiling.profile import VersionedVehicleProfile

logger = logging.getLogger(__name__)


def _yaml_safe(v):
    """Converte valores numpy/torch em tipos nativos Python para YAML."""
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, float) and (v != v):  # NaN
        return None
    return v


class HyperparameterOptimizer:
    """Otimização de hiperparâmetros com Optuna para qualquer TrainingPipeline."""

    def __init__(
        self,
        target: str,
        dataset: ProfileDataset,
        normalizer: ProfileDatasetNormalizer,
        base_model_hparams: Dict,
        base_training_hparams: Dict,
        production_training_hparams: Dict,
        val_size: float,
        test_size: float,
        shuffle: bool,
        random_state: int,
        model_dir: Path,
        training_dir: Path,
        pipeline_cls: type[TrainingPipeline],
        optuna_cfg: Optional[Dict] = None,
    ):
        self.target = target
        self.dataset = dataset
        self.normalizer = normalizer
        self.base_model_hparams = base_model_hparams
        self.base_training_hparams = base_training_hparams
        self.production_training_hparams = production_training_hparams
        self.val_size = val_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.model_dir = model_dir
        self.training_dir = training_dir

        cfg = optuna_cfg or {}
        self.n_trials = cfg.get("n_trials", 30)
        self.max_epochs_per_trial = cfg.get("max_epochs_per_trial", 60)
        self.seed = cfg.get("seed", 13)
        self.mae_weight = cfg.get("mae_weight", 0.5)
        self.bias_weight = cfg.get("bias_weight", 0.3)
        self.max_p90_k3 = cfg.get("max_p90_k3", 1.0)
        self.k_target = cfg.get("k_target", 2)

        # Forçar split temporal (sem shuffle) durante otimização
        self.shuffle = False

        # Espaço de busca
        search = cfg.get("search_space", {})
        self.alpha_daily_range = search.get("alpha_daily", [0.05, 0.6])
        self.lr_range = search.get("lr", [3e-4, 3e-3])
        self.hidden_dim_choices = search.get("hidden_dim", [32, 64, 96])
        self.conv_filters_choices = search.get("conv_filters", [16, 32, 48])
        self.dropout_range = search.get("dropout", [0.0, 0.25])

        # Classe do pipeline
        self.pipeline_cls = pipeline_cls

        # Preparar splits uma única vez (temporal, sem shuffle)
        self._prepare_splits()

        # Resultados
        self.study: Optional[optuna.Study] = None
        self.best_pipeline: Optional[TrainingPipeline] = None

    @classmethod
    def from_config(
        cls,
        df_daily,
        dataset_cfg,
        training_cfg,
        model_cfg,
        output_config,
        optuna_cfg,
        target: str,
        pipeline_cls: type[TrainingPipeline],
    ) -> "HyperparameterOptimizer":
        """Cria optimizer a partir das configurações existentes."""
        import pandas as pd

        if not isinstance(df_daily, pd.DataFrame):
            df_daily = df_daily.to_pandas()

        profile = VersionedVehicleProfile.load(
            profile_dir=str(output_config.profile_dir(target)),
            classifier_path=str(output_config.classifier_path(target)),
        )

        cache_dir = output_config.dataset_cache_dir(target)
        dataset = ProfileDatasetGenerator(
            vehicle_profile=profile,
            df_daily=df_daily,
            min_weeks_general=dataset_cfg.min_weeks_general,
            num_weeks_recent=dataset_cfg.num_weeks_recent,
            horizon_weeks=dataset_cfg.horizon_weeks,
            daily_horizon=dataset_cfg.daily_horizon,
            min_recent_active_days=dataset_cfg.min_recent_active_days,
            cache_dir=cache_dir,
            cluster_features=dataset_cfg.cluster_features,
        ).generate()

        normalizer = ProfileDatasetNormalizer(
            cv_max=training_cfg.normalization.cv_max,
            ratio_max=training_cfg.normalization.ratio_max,
        )

        fc = model_cfg
        model_hparams = {
            "hidden_dim": fc.model.hidden_dim,
            "conv_filters": fc.model.conv_filters,
            "conv_kernel": fc.model.conv_kernel,
            "conv_layers": fc.model.conv_layers,
            "dropout": fc.model.dropout,
            "loss_heads_type": fc.model.loss_heads_type,
            "loss_daily_type": fc.model.loss_daily_type,
            "lr": fc.optimizer.lr,
            "weight_decay": fc.optimizer.weight_decay,
            "alpha_daily": fc.model.alpha_daily,
        }

        # Hparams rápidos para trials Optuna
        oc = optuna_cfg
        trial_training_hparams = {
            "seed": oc.seed,
            "batch_size": oc.data.batch_size,
            "num_workers": oc.data.num_workers,
            "max_epochs": oc.trainer.max_epochs,
            "accelerator": oc.trainer.accelerator,
            "precision": oc.trainer.precision,
            "gradient_clip_val": oc.trainer.gradient_clip_val,
            "early_stopping": {
                "monitor": oc.trainer.early_stopping.monitor,
                "patience": oc.trainer.early_stopping.patience,
                "min_delta": oc.trainer.early_stopping.min_delta,
            },
            "checkpoint": {
                "monitor": oc.trainer.checkpoint.monitor,
                "save_top_k": oc.trainer.checkpoint.save_top_k,
                "mode": oc.trainer.checkpoint.mode,
            },
            "onnx": {
                "enabled": False,
                "opset_version": 18,
                "verify": False,
            },
        }

        # Hparams de produção para retrain_best
        tc = training_cfg
        production_training_hparams = {
            "seed": tc.seed,
            "batch_size": tc.data.batch_size,
            "num_workers": tc.data.num_workers,
            "max_epochs": tc.trainer.max_epochs,
            "accelerator": tc.trainer.accelerator,
            "precision": tc.trainer.precision,
            "gradient_clip_val": tc.trainer.gradient_clip_val,
            "early_stopping": {
                "monitor": tc.trainer.early_stopping.monitor,
                "patience": tc.trainer.early_stopping.patience,
                "min_delta": tc.trainer.early_stopping.min_delta,
            },
            "checkpoint": {
                "monitor": tc.trainer.checkpoint.monitor,
                "save_top_k": tc.trainer.checkpoint.save_top_k,
                "mode": tc.trainer.checkpoint.mode,
            },
            "onnx": {
                "enabled": tc.onnx.enabled,
                "opset_version": tc.onnx.opset_version,
                "verify": tc.onnx.verify,
            },
        }

        split = training_cfg.split

        optuna_dict = {
            "n_trials": optuna_cfg.n_trials,
            "max_epochs_per_trial": optuna_cfg.max_epochs_per_trial,
            "seed": optuna_cfg.seed,
            "mae_weight": optuna_cfg.objective.mae_weight,
            "bias_weight": optuna_cfg.objective.bias_weight,
            "max_p90_k3": optuna_cfg.objective.max_p90_k3,
            "k_target": optuna_cfg.objective.k_target,
            "search_space": {
                "alpha_daily": [optuna_cfg.search_space.alpha_daily.low, optuna_cfg.search_space.alpha_daily.high],
                "lr": [optuna_cfg.search_space.lr.low, optuna_cfg.search_space.lr.high],
                "hidden_dim": optuna_cfg.search_space.hidden_dim,
                "conv_filters": optuna_cfg.search_space.conv_filters,
                "dropout": [optuna_cfg.search_space.dropout.low, optuna_cfg.search_space.dropout.high],
            },
        }

        return cls(
            target=target,
            dataset=dataset,
            normalizer=normalizer,
            base_model_hparams=model_hparams,
            base_training_hparams=trial_training_hparams,
            production_training_hparams=production_training_hparams,
            val_size=split.val_size,
            test_size=split.test_size,
            shuffle=split.shuffle,
            random_state=split.random_state,
            model_dir=Path(output_config.models.forecasting),
            training_dir=output_config.training_dir(target),
            pipeline_cls=pipeline_cls,
            optuna_cfg=optuna_dict,
        )

    def _prepare_splits(self):
        """Normaliza e divide o dataset uma única vez (reutilizado em todos os trials)."""
        dataset_norm = self.normalizer.fit_normalize_dataset(self.dataset)

        has_test = self.test_size > 0
        has_val = self.val_size > 0
        remaining = dataset_norm

        self._test_ds = None
        self._val_ds = None

        if has_test:
            remaining, self._test_ds = train_test_split(
                remaining,
                test_size=self.test_size,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

        if has_val:
            val_ratio = self.val_size / (1.0 - self.test_size) if has_test else self.val_size
            remaining, self._val_ds = train_test_split(
                remaining,
                test_size=val_ratio,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

        self._train_ds = remaining

        print(f"Splits preparados — Treino: {len(self._train_ds):,}  "
              f"Val: {len(self._val_ds):,}  "
              f"Teste: {len(self._test_ds) if self._test_ds else 0:,}")

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        """Função objetivo para um trial do Optuna."""
        # Sugerir hiperparâmetros
        alpha_daily = trial.suggest_float(
            "alpha_daily", self.alpha_daily_range[0], self.alpha_daily_range[1],
        )
        lr = trial.suggest_float(
            "lr", self.lr_range[0], self.lr_range[1], log=True,
        )
        hidden_dim = trial.suggest_categorical(
            "hidden_dim", self.hidden_dim_choices,
        )
        conv_filters = trial.suggest_categorical(
            "conv_filters", self.conv_filters_choices,
        )
        dropout = trial.suggest_float(
            "dropout", self.dropout_range[0], self.dropout_range[1],
        )

        # Construir hparams do trial
        model_hparams = deepcopy(self.base_model_hparams)
        model_hparams["alpha_daily"] = alpha_daily
        model_hparams["lr"] = lr
        model_hparams["hidden_dim"] = hidden_dim
        model_hparams["conv_filters"] = conv_filters
        model_hparams["dropout"] = dropout

        training_hparams = deepcopy(self.base_training_hparams)
        training_hparams["max_epochs"] = self.max_epochs_per_trial

        # Criar pipeline com splits já preparados
        pipeline = self.pipeline_cls(
            target=self.target,
            dataset=self.dataset,
            normalizer=self.normalizer,
            val_size=self.val_size,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
            model_hparams=model_hparams,
            training_hparams=training_hparams,
            model_dir=self.model_dir,
            training_dir=self.training_dir,
        )

        # Reutilizar splits pré-calculados
        pipeline.train_ds = self._train_ds
        pipeline.val_ds = self._val_ds
        pipeline.test_ds = None  # Não avaliar teste durante otimização

        # Treinar
        lightning.seed_everything(self.seed)
        try:
            pipeline.train()
        except Exception as e:
            logger.warning("Trial %d falhou no treino: %s", trial.number, e)
            return float("inf")

        # Avaliar na validação
        try:
            metrics = pipeline.evaluate()
        except Exception as e:
            logger.warning("Trial %d falhou na avaliação: %s", trial.number, e)
            return float("inf")

        if "val" not in metrics:
            return float("inf")

        val_maint = metrics["val"]["maintenance"]
        k_label = f"k{self.k_target}"

        if k_label not in val_maint:
            return float("inf")

        p90 = val_maint[k_label]["p90_error"]
        mae = val_maint[k_label]["mae_days"]
        mean_error = val_maint[k_label]["mean_error"]

        # Restrição dura: rejeitar se P90_k3 exceder limite
        if "k3" in val_maint and val_maint["k3"]["p90_error"] > self.max_p90_k3:
            trial.set_user_attr("pruned_reason", "P90_k3 excedeu limite")
            trial.set_user_attr("p90_k3", val_maint["k3"]["p90_error"])
            print(
                f"  Trial {trial.number:>3d} | REJEITADO | "
                f"P90_k3={val_maint['k3']['p90_error']:+.2f} > {self.max_p90_k3}"
            )
            return float("inf")

        # Função objetivo com penalização de viés de atraso
        bias_penalty = self.bias_weight * max(mean_error, 0.0)
        objective_value = p90 + self.mae_weight * mae + bias_penalty

        # Registrar todos os componentes para auditoria
        trial.set_user_attr("p90_k2", p90)
        trial.set_user_attr("mae_k2", mae)
        trial.set_user_attr("mean_error_k2", mean_error)
        trial.set_user_attr("bias_penalty", bias_penalty)
        trial.set_user_attr("pct_late_k2", val_maint[k_label]["pct_late"])

        # Métricas para k=3 (monitorar)
        if "k3" in val_maint:
            trial.set_user_attr("p90_k3", val_maint["k3"]["p90_error"])
            trial.set_user_attr("mae_k3", val_maint["k3"]["mae_days"])
            trial.set_user_attr("mean_error_k3", val_maint["k3"]["mean_error"])

        print(
            f"  Trial {trial.number:>3d} | "
            f"obj={objective_value:+.2f} | "
            f"P90_k2={p90:+.2f} | MAE_k2={mae:.2f} | "
            f"bias={bias_penalty:+.2f} | "
            f"P90_k3={val_maint.get('k3', {}).get('p90_error', 0):+.2f} | "
            f"late={val_maint[k_label]['pct_late']:.1f}% | "
            f"α={alpha_daily:.3f} lr={lr:.1e} hid={hidden_dim} "
            f"conv={conv_filters} drop={dropout:.2f}"
        )

        return objective_value

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def optimize(self) -> optuna.Study:
        """Executa a otimização e retorna o estudo Optuna."""
        print(f"\n{'=' * 60}")
        print(f"  OTIMIZAÇÃO DE HIPERPARÂMETROS — {self.target.upper()}")
        print(f"  {self.n_trials} trials | max_epochs={self.max_epochs_per_trial}")
        print(f"  Objetivo: P90_k{self.k_target} + {self.mae_weight} × MAE_k{self.k_target} + {self.bias_weight} × max(mean_error, 0)")
        print(f"  Restrição: P90_k3 ≤ {self.max_p90_k3} dias")
        print(f"  Split temporal: shuffle=False (forçado)")
        print(f"{'=' * 60}\n")

        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            study_name=f"optim_{self.target}",
        )

        # TensorBoard callback — loga hiperparâmetros e objetivo por trial
        callbacks = []
        tag = self.pipeline_cls.tag
        tb_log_dir = self.training_dir / f"optuna_tb_{tag}_{self.target}"
        try:
            from optuna.integration import TensorBoardCallback
            tb_callback = TensorBoardCallback(
                dirname=str(tb_log_dir),
                metric_name="objective",
            )
            callbacks.append(tb_callback)
            print(f"  TensorBoard: {tb_log_dir}")
        except ImportError:
            logger.info("optuna[tensorboard] não instalado — prosseguindo sem TensorBoard.")

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            callbacks=callbacks,
        )

        self._print_results()
        self._save_results()
        return self.study

    def retrain_best(self, max_epochs: Optional[int] = None) -> TrainingPipeline:
        """
        Retreina o modelo com os melhores hiperparâmetros encontrados.

        Usa todos os epochs configurados (ou ``max_epochs`` se fornecido),
        avalia em validação e teste, exporta ONNX e gera relatório.
        """
        if self.study is None:
            raise RuntimeError("Chame optimize() primeiro.")

        best = self.study.best_params
        print(f"\n{'=' * 60}")
        print(f"  RETREINO COM MELHORES HIPERPARÂMETROS")
        print(f"{'=' * 60}")
        for k, v in best.items():
            print(f"  {k}: {v}")
        print()

        model_hparams = deepcopy(self.base_model_hparams)
        model_hparams.update(best)

        training_hparams = deepcopy(self.production_training_hparams)
        if max_epochs is not None:
            training_hparams["max_epochs"] = max_epochs

        pipeline = self.pipeline_cls(
            target=self.target,
            dataset=self.dataset,
            normalizer=self.normalizer,
            val_size=self.val_size,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
            model_hparams=model_hparams,
            training_hparams=training_hparams,
            model_dir=self.model_dir,
            training_dir=self.training_dir,
        )

        # Reutilizar splits
        pipeline.train_ds = self._train_ds
        pipeline.val_ds = self._val_ds
        pipeline.test_ds = self._test_ds

        lightning.seed_everything(self.seed)
        pipeline.train()
        pipeline.metrics = pipeline.evaluate()
        pipeline.export_onnx()
        pipeline.generate_report()

        self.best_pipeline = pipeline
        self._save_results()
        return pipeline

    def _save_results(self):
        """Salva melhores hiperparâmetros e métricas em YAML."""
        import yaml

        if self.study is None:
            return

        best = self.study.best_trial
        output = {
            "target": self.target,
            "pipeline_cls": self.pipeline_cls.__name__,
            "n_trials": len(self.study.trials),
            "best_trial": best.number,
            "best_objective": float(best.value),
            "best_params": {k: _yaml_safe(v) for k, v in best.params.items()},
            "best_metrics": {k: _yaml_safe(v) for k, v in best.user_attrs.items()
                            if not k.startswith("pruned")},
        }

        tag = self.pipeline_cls.tag
        out_path = self.training_dir / f"best_hparams_{tag}_{self.target}.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"Hiperparâmetros salvos: {out_path}")

    def _print_results(self):
        """Imprime resumo dos resultados da otimização."""
        study = self.study
        best = study.best_trial

        print(f"\n{'=' * 60}")
        print(f"  RESULTADO DA OTIMIZAÇÃO — {self.target.upper()}")
        print(f"{'=' * 60}")
        print(f"  Melhor trial:    #{best.number}")
        print(f"  Objetivo:        {best.value:+.4f}")
        print(f"  P90_k2:          {best.user_attrs.get('p90_k2', '?'):+.2f}")
        print(f"  MAE_k2:          {best.user_attrs.get('mae_k2', '?'):.2f}")
        print(f"  % atrasos k2:    {best.user_attrs.get('pct_late_k2', '?'):.1f}%")

        if "p90_k3" in best.user_attrs:
            print(f"  P90_k3:          {best.user_attrs['p90_k3']:+.2f}")
            print(f"  MAE_k3:          {best.user_attrs['mae_k3']:.2f}")

        print(f"\n  Hiperparâmetros:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

        # Top 5
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top5 = sorted(completed, key=lambda t: t.value)[:5]
        print(f"\n  Top 5 trials:")
        for t in top5:
            print(
                f"    #{t.number:>3d}  obj={t.value:+.4f}  "
                f"P90={t.user_attrs.get('p90_k2', '?'):+.2f}  "
                f"MAE={t.user_attrs.get('mae_k2', '?'):.2f}  "
                f"late={t.user_attrs.get('pct_late_k2', '?'):.1f}%"
            )
        print(f"{'=' * 60}\n")
