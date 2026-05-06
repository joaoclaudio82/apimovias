"""
Pipelines de treinamento.

Classes
-------
- ``TrainingPipeline`` — base com fluxo completo (normalização, split,
  treino, avaliação, ONNX, relatório). Subclasses fornecem factories.
- ``MultiHeadTrainingPipeline`` — baseline multi-head (2 inputs).
- ``MoETrainingPipeline`` — SAFE MoE (3 inputs: struct + cluster + recent).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as lightning
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


from moviasai.data import (
    ForecastingDataModule,
    MoEForecastingDataModule,
    ProfileDataset,
    ProfileDatasetGenerator,
    ProfileDatasetNormalizer,
)
from moviasai.data.utils import train_test_split
from moviasai.forecasting.models import (
    VehicleForecastingMultiHeadModel,
    VehicleForecastingSafeMoEModel,
)
from moviasai.forecasting.metrics import compute_maintenance_metrics, save_predictions
from moviasai.profiling.profile import VersionedVehicleProfile

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orquestra geração de dados, treinamento e avaliação."""

    @classmethod
    def from_config(
        cls,
        df_daily: pd.DataFrame,
        dataset_cfg,
        training_cfg,
        model_cfg,
        output_config,
        target: str,
    ) -> "TrainingPipeline":
        """
        Cria pipeline a partir de configurações.

        Parameters
        ----------
        df_daily : pd.DataFrame
            DataFrame com dados diários de telemetria.
        dataset_cfg : api.config.dataset_config.DatasetConfig
            Configuração de geração de dataset.
        training_cfg : api.config.training_config.TrainingConfig
            Configuração de normalização e split.
        model_cfg : api.config.model_config.ModelConfig
            Configuração do modelo / trainer / ONNX.
        output_config : api.config.output_config.OutputConfig
            Configuração de diretórios de saída.
        target : str
            Target a processar: ``'km'`` ou ``'h'``.
        """
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

        # Sobrescrever hiperparâmetros com valores salvos de otimização
        if getattr(training_cfg, "use_saved_hyperparameters", False):
            tag = cls.tag
            hparams_file = Path(output_config.training_dir(target)) / f"best_hparams_{tag}_{target}.yaml"
            if hparams_file.exists():
                logger.info(
                    "Usando hiperparâmetros salvos: %s", hparams_file,
                )
                with open(hparams_file, "r", encoding="utf-8") as f:
                    saved = yaml.safe_load(f) or {}

                best_params = saved.get("best_params", {})

                # Sobrescrever atributos de model e optimizer
                for key, value in best_params.items():
                    if key in model_hparams:
                        logger.info(
                            "  %s: %s → %s (salvo)",
                            key, model_hparams[key], value,
                        )
                        model_hparams[key] = value
                    else:
                        logger.warning(
                            "  %s: ignorado (não reconhecido em model_hparams)",
                            key,
                        )
            else:
                logger.info(
                    "use_saved_hyperparameters=True mas arquivo não encontrado: %s. "
                    "Usando hiperparâmetros do model_config.yaml.",
                    hparams_file,
                )

        tc = training_cfg
        training_hparams = {
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
        return cls(
            target=target,
            dataset=dataset,
            normalizer=normalizer,
            val_size=split.val_size,
            test_size=split.test_size,
            shuffle=split.shuffle,
            random_state=split.random_state,
            model_hparams=model_hparams,
            training_hparams=training_hparams,
            model_dir=Path(output_config.models.forecasting),
            training_dir=output_config.training_dir(target),
        )

    def __init__(
        self,
        target: str,
        dataset: ProfileDataset,
        normalizer: ProfileDatasetNormalizer,
        val_size: float,
        test_size: float,
        shuffle: bool,
        random_state: int,
        model_hparams: Dict,
        training_hparams: Dict,
        model_dir: Path,
        training_dir: Path,
    ):
        self.target = target
        self.val_size = val_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.model_hparams = model_hparams
        self.training_hparams = training_hparams
        self.model_dir = model_dir
        self.training_dir = training_dir

        self.dataset = dataset
        self.normalizer = normalizer

        # Estado preenchido durante o pipeline
        self.train_ds: Optional[ProfileDataset] = None
        self.val_ds: Optional[ProfileDataset] = None
        self.test_ds: Optional[ProfileDataset] = None
        self.data_module = None
        self.model = None
        self.trainer: Optional[lightning.Trainer] = None
        self.metrics: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Factories (subclasses devem sobrescrever)
    # ------------------------------------------------------------------

    def _create_data_module(self):
        """Cria o DataModule adequado ao modelo."""
        raise NotImplementedError

    def _create_model(self, dm):
        """Cria o modelo adequado ao pipeline."""
        raise NotImplementedError

    @property
    def _onnx_filename(self) -> str:
        raise NotImplementedError

    tag: str
    """Sufixo para diferenciar artefactos (ex.: 'multihead', 'moe')."""

    def _predict_on_dataset(self, dataset: ProfileDataset):
        """Constrói tensores e executa inferência. Retorna (y_heads, y_daily) numpy."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Executa o pipeline completo e retorna métricas."""
        logger.info("=" * 60)
        logger.info("PIPELINE DE TREINAMENTO — target=%s", self.target)
        logger.info("=" * 60)

        self.normalize_and_split()
        self.train()
        self.metrics = self.evaluate()
        self.export_onnx()
        self.generate_report()

        logger.info("Pipeline concluído.")
        return self.metrics

    # ------------------------------------------------------------------
    # 1. Normalização + split
    # ------------------------------------------------------------------

    def normalize_and_split(self):
        """Normaliza o dataset e divide em treino / validação / teste."""
        dataset_norm = self.normalizer.fit_normalize_dataset(self.dataset)
        logger.info("Normalização concluída.")

        # Split: primeiro separa teste, depois valida a partir do treino
        has_val = self.val_size > 0
        has_test = self.test_size > 0

        remaining = dataset_norm

        if has_test:
            remaining, self.test_ds = train_test_split(
                remaining,
                test_size=self.test_size,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            logger.info("Teste: %d amostras", len(self.test_ds))

        if has_val:
            # Ajustar proporção de validação em relação ao restante
            val_ratio = self.val_size / (1.0 - self.test_size) if has_test else self.val_size
            remaining, self.val_ds = train_test_split(
                remaining,
                test_size=val_ratio,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            logger.info("Validação: %d amostras", len(self.val_ds))

        self.train_ds = remaining
        logger.info("Treino: %d amostras", len(self.train_ds))

    # ------------------------------------------------------------------
    # 3. Treinamento
    # ------------------------------------------------------------------

    def train(self):
        """Configura e executa o treinamento com PyTorch Lightning."""
        if self.train_ds is None:
            raise RuntimeError("Datasets não preparados. Chame normalize_and_split() primeiro.")

        tp = self.training_hparams
        lightning.seed_everything(tp["seed"])

        self.data_module = self._create_data_module()
        self.model = self._create_model(self.data_module)

        # Callbacks
        callbacks = []
        has_val = self.val_ds is not None

        if has_val:
            es = tp["early_stopping"]
            callbacks.append(
                EarlyStopping(
                    monitor=es["monitor"],
                    patience=es["patience"],
                    min_delta=es["min_delta"],
                    mode="min",
                )
            )
            ckpt = tp["checkpoint"]
            callbacks.append(
                ModelCheckpoint(
                    monitor=ckpt["monitor"],
                    save_top_k=ckpt["save_top_k"],
                    mode=ckpt["mode"],
                )
            )

        # Logger para curvas de treinamento
        csv_logger = CSVLogger(
            save_dir=str(self.training_dir),
            name=f"training_logs_{self.target}",
        )

        # Trainer
        self.trainer = lightning.Trainer(
            max_epochs=tp["max_epochs"],
            accelerator=tp["accelerator"],
            precision=tp["precision"],
            gradient_clip_val=tp["gradient_clip_val"],
            callbacks=callbacks,
            logger=csv_logger,
            enable_progress_bar=True,
            deterministic=True,
        )

        logger.info("Iniciando treinamento ...")
        self.trainer.fit(self.model, datamodule=self.data_module)

        # Carregar melhor checkpoint se disponível
        if has_val and self.trainer.checkpoint_callback.best_model_path:
            logger.info(
                "Carregando melhor checkpoint: %s",
                self.trainer.checkpoint_callback.best_model_path,
            )
            self.model = type(self.model).load_from_checkpoint(
                self.trainer.checkpoint_callback.best_model_path,
            )

        logger.info("Treinamento concluído.")

    # ------------------------------------------------------------------
    # 4. Avaliação
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict:
        """
        Avalia o modelo nos conjuntos de validação e teste.

        Retorna métricas denormalizadas (MAE, RMSE, MAPE) por cabeça.
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")

        results: Dict = {}

        if self.val_ds is not None:
            results["val"] = self._evaluate_split("val", self.val_ds)

        if self.test_ds is not None:
            results["test"] = self._evaluate_split("test", self.test_ds)

        return results

    def _evaluate_split(self, name: str, dataset: ProfileDataset) -> Dict:
        """Avalia um split e retorna métricas denormalizadas."""
        y_heads_pred, y_daily_pred = self._predict_on_dataset(dataset)

        # Garantir não-negatividade (pós-processamento)
        y_heads_pred = np.clip(y_heads_pred, a_min=0.0, a_max=None)
        y_daily_pred = np.clip(y_daily_pred, a_min=0.0, a_max=None)

        # Denormalizar predições e targets
        upper = dataset.metadata["upper"].values.astype(np.float64)
        head_days = self.dataset.config["head_days"]

        y_heads_true = ProfileDatasetNormalizer.inverse_transform_target_heads(
            dataset.y_heads.values, upper, head_days,
        )
        y_daily_true = ProfileDatasetNormalizer.inverse_transform_target_daily(
            dataset.y_daily.values, upper,
        )
        y_heads_pred_denorm = ProfileDatasetNormalizer.inverse_transform_target_heads(
            y_heads_pred, upper, head_days,
        )
        y_daily_pred_denorm = ProfileDatasetNormalizer.inverse_transform_target_daily(
            y_daily_pred, upper,
        )

        # Métricas de decisão de manutenção (horizonte misto)
        maintenance = compute_maintenance_metrics(
            y_daily_true, y_daily_pred_denorm,
            y_heads_true, y_heads_pred_denorm,
            head_days, upper,
        )

        # Métricas por head (MAE, RMSE)
        heads_mae = np.mean(np.abs(y_heads_true - y_heads_pred_denorm), axis=0)
        heads_rmse = np.sqrt(np.mean((y_heads_true - y_heads_pred_denorm) ** 2, axis=0))

        # Métricas daily (MAE, RMSE) — só para os D dias disponíveis
        daily_mae = np.mean(np.abs(y_daily_true - y_daily_pred_denorm), axis=0)
        daily_rmse = np.sqrt(np.mean((y_daily_true - y_daily_pred_denorm) ** 2, axis=0))

        self._log_metrics(name, maintenance, heads_mae, heads_rmse,
                          daily_mae, daily_rmse, head_days)

        # Salvar predições
        pred_dir = Path(self.training_dir) / "predictions"
        save_predictions(
            pred_dir, name, dataset.metadata,
            y_heads_true, y_heads_pred_denorm,
            y_daily_true, y_daily_pred_denorm,
        )
        logger.info(
            "Predições salvas: %s (%d amostras)",
            pred_dir, len(dataset.metadata),
        )

        return {
            "maintenance": maintenance,
            "y_heads_true": y_heads_true,
            "y_heads_pred": y_heads_pred_denorm,
            "y_daily_true": y_daily_true,
            "y_daily_pred": y_daily_pred_denorm,
            "upper": upper,
            "head_days": head_days,
            "metadata": dataset.metadata,
            "heads_mae": heads_mae,
            "heads_rmse": heads_rmse,
            "daily_mae": daily_mae,
            "daily_rmse": daily_rmse,
        }

    @staticmethod
    def _log_metrics(
        name: str,
        maintenance: Dict,
        heads_mae: np.ndarray,
        heads_rmse: np.ndarray,
        daily_mae: np.ndarray,
        daily_rmse: np.ndarray,
        head_days: list,
    ):
        """Imprime métricas de decisão e de previsão."""
        print(f"\n{'=' * 60}")
        print(f"  MÉTRICAS — {name.upper()}")
        print(f"{'=' * 60}")

        # Métricas por head
        print("\n  HEADS (agregado semanal):")
        for h, (mae, rmse, days) in enumerate(zip(heads_mae, heads_rmse, head_days)):
            print(f"    head_{h} ({days}d):  MAE={mae:.2f}  RMSE={rmse:.2f}")

        # Métricas daily
        D = len(daily_mae)
        print(f"\n  DAILY (dias 1–{D}):")
        print(f"    MAE médio:  {daily_mae.mean():.2f}")
        print(f"    RMSE médio: {daily_rmse.mean():.2f}")

        # Decisão de manutenção
        print("\n  DECISÃO DE MANUTENÇÃO:")
        for k_label in sorted(maintenance.keys()):
            m = maintenance[k_label]
            k = int(k_label[1:])
            print(f"\n  Limite: k={k} ({k_label})")
            print(f"    Erro médio:          {m['mean_error']:+.2f} dias")
            print(f"    Erro absoluto médio: {m['mae_days']:.2f} dias")
            print(f"    P90 erro:            {m['p90_error']:+.2f} dias")
            print(f"    % atrasos:           {m['pct_late']:.1f}%")
            print(f"    % adiantamentos:     {m['pct_early']:.1f}%")

        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # 5. Exportação ONNX
    # ------------------------------------------------------------------

    def export_onnx(self) -> Optional[Path]:
        """
        Exporta o modelo treinado para ONNX, se habilitado.

        Salva com sufixo de data (ex.: forecasting_moe_km_20260427.onnx).
        O path exportado fica disponível em ``self.exported_onnx_path``.

        Returns
        -------
        Path para o ficheiro exportado, ou None se desabilitado.
        """
        from datetime import date as _date

        onnx_cfg = self.training_hparams["onnx"]
        if not onnx_cfg["enabled"]:
            logger.info("Exportação ONNX desabilitada.")
            return None

        if self.model is None or self.data_module is None:
            raise RuntimeError("Modelo não treinado.")

        # Ficheiro com data (versionado)
        stem = Path(self._onnx_filename).stem  # ex: forecasting_moe_km
        today = _date.today().strftime("%Y%m%d")
        dated_filename = f"{stem}_{today}.onnx"
        dated_path = self.model_dir / dated_filename
        dated_path.parent.mkdir(parents=True, exist_ok=True)

        onnx_path = self.model.export_onnx(
            path=dated_path,
            n_recent_steps=self.data_module.n_recent_steps,
            opset_version=onnx_cfg["opset_version"],
            verify=onnx_cfg["verify"],
        )
        logger.info("Modelo ONNX exportado: %s", onnx_path)

        self.exported_onnx_path = onnx_path
        return onnx_path

    # ------------------------------------------------------------------
    # 6. Relatório PDF
    # ------------------------------------------------------------------

    def generate_report(self) -> Optional[Path]:
        """Gera relatório PDF com gráficos de avaliação."""
        if self.metrics is None:
            logger.info("Sem métricas para gerar relatório.")
            return None

        from moviasai.report import TrainingReportGenerator

        output_dir = self.training_dir
        report = TrainingReportGenerator(output_dir)
        return report.generate(self)


# ======================================================================
# MultiHead — baseline com 2 inputs (X_general, X_recent)
# ======================================================================


class MultiHeadTrainingPipeline(TrainingPipeline):
    """Pipeline para ``VehicleForecastingMultiHeadModel`` (2 inputs)."""

    tag = "multihead"

    def _create_data_module(self):
        tp = self.training_hparams
        return ForecastingDataModule(
            train_dataset=self.train_ds,
            val_dataset=self.val_ds,
            test_dataset=self.test_ds,
            batch_size=tp["batch_size"],
            num_workers=tp["num_workers"],
        )

    def _create_model(self, dm):
        return VehicleForecastingMultiHeadModel(
            n_general_features=dm.n_general_features,
            n_recent_features=dm.n_recent_features,
            n_heads=dm.n_heads,
            n_daily=dm.n_daily,
            **self.model_hparams,
        )

    @property
    def _onnx_filename(self) -> str:
        return f"forecasting_model_{self.target}.onnx"



    def _predict_on_dataset(self, dataset: ProfileDataset):
        self.model.eval()
        device = next(self.model.parameters()).device

        struct_cols, day_features = ForecastingDataModule._parse_columns(
            dataset.X_general.columns,
        )
        X_struct = torch.from_numpy(
            dataset.X_general[struct_cols].values.astype(np.float32)
        ).to(device)
        X_recent_3d = torch.from_numpy(
            ForecastingDataModule._build_X_recent_3d(
                dataset.X_recent, dataset.X_general, day_features,
            )
        ).to(device)

        with torch.no_grad():
            y_heads_pred, y_daily_pred = self.model(X_struct, X_recent_3d)

        return y_heads_pred.cpu().numpy(), y_daily_pred.cpu().numpy()


# ======================================================================
# SAFE MoE — 3 inputs (X_general_struct, X_cluster, X_recent)
# ======================================================================


class MoETrainingPipeline(TrainingPipeline):
    """Pipeline para ``VehicleForecastingSafeMoEModel`` (3 inputs)."""

    tag = "moe"

    def _create_data_module(self):
        tp = self.training_hparams
        return MoEForecastingDataModule(
            train_dataset=self.train_ds,
            val_dataset=self.val_ds,
            test_dataset=self.test_ds,
            batch_size=tp["batch_size"],
            num_workers=tp["num_workers"],
        )

    def _create_model(self, dm):
        return VehicleForecastingSafeMoEModel(
            n_general_features=dm.n_general_features,
            n_clusters=dm.n_clusters,
            n_recent_features=dm.n_recent_features,
            n_heads=dm.n_heads,
            n_daily=dm.n_daily,
            **self.model_hparams,
        )

    @property
    def _onnx_filename(self) -> str:
        return f"forecasting_moe_{self.target}.onnx"



    def _predict_on_dataset(self, dataset: ProfileDataset):
        self.model.eval()
        dm: MoEForecastingDataModule = self.data_module
        device = next(self.model.parameters()).device

        X_struct = torch.from_numpy(
            dataset.X_general[dm.general_struct_cols].values.astype(np.float32)
        ).to(device)
        X_cluster = torch.from_numpy(
            dataset.X_general[dm.cluster_cols].values.astype(np.float32)
        ).to(device)

        _, day_features = ForecastingDataModule._parse_columns(dataset.X_general.columns)
        X_recent_3d = torch.from_numpy(
            ForecastingDataModule._build_X_recent_3d(
                dataset.X_recent, dataset.X_general, day_features,
            )
        ).to(device)

        with torch.no_grad():
            y_heads_pred, y_daily_pred = self.model(X_struct, X_cluster, X_recent_3d)

        return y_heads_pred.cpu().numpy(), y_daily_pred.cpu().numpy()
