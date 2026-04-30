from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import pytorch_lightning as pl


class VehicleForecastingBaseModel(pl.LightningModule):
    """
    Classe base para modelos de previsão de consumo de veículos.

    Contém a arquitectura partilhada:

    - **Encoder estrutural** (MLP): processa features estruturais
    - **Encoder temporal** (Conv1D): processa histórico recente
    - **Fusão**: concatena embeddings
    - **Cabeças globais**: ``y_heads`` (agregado) e ``y_daily`` (diário)
    - **Funções de perda**, optimizer e steps de treino/validação/teste

    Subclasses implementam ``forward()``, ``_compute_loss()``,
    ``predict_step()`` e ``export_onnx()``.
    """

    def __init__(
        self,
        n_general_features: int,
        n_recent_features: int,
        n_heads: int,
        n_daily: int,
        hidden_dim: int = 64,
        conv_filters: int = 32,
        conv_kernel: int = 3,
        conv_layers: int = 2,
        dropout: float = 0.1,
        loss_heads_type: Literal['huber', 'mse', 'mae'] = 'huber',
        loss_daily_type: Literal['mae', 'mse', 'huber'] = 'mae',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha_daily: float = 0.3,
    ):
        super().__init__()
        # NOTE: subclasses chamam save_hyperparameters()

        # ---- Encoder estrutural (MLP) ----
        self.encoder_general = nn.Sequential(
            nn.Linear(n_general_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Encoder temporal (Conv1D) ----
        padding = (conv_kernel - 1) // 2
        conv_blocks = []
        in_ch = n_recent_features
        for _ in range(conv_layers):
            conv_blocks.extend([
                nn.Conv1d(in_ch, conv_filters, kernel_size=conv_kernel, padding=padding),
                nn.ReLU(),
            ])
            in_ch = conv_filters
        conv_blocks.append(nn.AdaptiveAvgPool1d(1))
        self.encoder_temporal = nn.Sequential(*conv_blocks)

        # ---- Fusão ----
        fusion_dim = hidden_dim + conv_filters
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Cabeça A: targets agregados ----
        self.head_agg = nn.Linear(hidden_dim, n_heads)

        # ---- Cabeça B: targets diários ----
        self.head_daily = nn.Linear(hidden_dim, n_daily)

        # ---- Funções de perda ----
        self.loss_heads_fn = self._build_loss(loss_heads_type)
        self.loss_daily_fn = self._build_loss(loss_daily_type)

    @staticmethod
    def _build_loss(name: str) -> nn.Module:
        if name == 'huber':
            return nn.HuberLoss(reduction="none")
        if name == 'mse':
            return nn.MSELoss(reduction="none")
        if name == 'mae':
            return nn.L1Loss(reduction="none")
        raise ValueError(f"Loss desconhecida: {name!r}. Usar 'huber', 'mse' ou 'mae'.")

    # ------------------------------------------------------------------
    # Encoding partilhado
    # ------------------------------------------------------------------

    def _encode(self, X_general: torch.Tensor, X_recent: torch.Tensor) -> torch.Tensor:
        """
        Encoding partilhado: MLP estrutural + Conv1D temporal + fusão.

        Parameters
        ----------
        X_general : (B, Dg)
            Features estruturais (sem ``cluster_*`` no MoE).
        X_recent : (B, T, F)
            Histórico recente com contexto diário.

        Returns
        -------
        fused : (B, hidden_dim)
        """
        emb_general = self.encoder_general(X_general)
        x_conv = X_recent.transpose(1, 2)
        emb_recent = self.encoder_temporal(x_conv).squeeze(-1)
        fused = torch.cat([emb_general, emb_recent], dim=1)
        return self.fusion(fused)

    def _compute_combined_loss(
        self,
        y_heads_pred: torch.Tensor,
        y_daily_pred: torch.Tensor,
        y_heads: torch.Tensor,
        y_daily: torch.Tensor,
        weights: torch.Tensor,
    ):
        """Calcula loss combinada ``loss_heads + α × loss_daily``."""
        loss_h = self.loss_heads_fn(y_heads_pred, y_heads).mean(dim=1)
        loss_d = self.loss_daily_fn(y_daily_pred, y_daily).mean(dim=1)
        alpha = self.hparams.alpha_daily
        per_sample = loss_h + alpha * loss_d
        loss = (per_sample * weights).mean()
        return loss, loss_h.mean(), loss_d.mean()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, loss_h, loss_d = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_heads", loss_h)
        self.log("train_loss_daily", loss_d)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_h, loss_d = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_heads", loss_h)
        self.log("val_loss_daily", loss_d)

    def test_step(self, batch, batch_idx):
        loss, loss_h, loss_d = self._compute_loss(batch)
        self.log("test_loss", loss)
        self.log("test_loss_heads", loss_h)
        self.log("test_loss_daily", loss_d)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        monitor = "val_loss" if self.trainer and self.trainer.val_dataloaders else "train_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor,
            },
        }

    # ------------------------------------------------------------------
    # Exportação ONNX
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        path: str | Path,
        n_recent_steps: int = 7,
        opset_version: int = 17,
        verify: bool = True,
    ) -> Path:
        """
        Exporta o modelo para formato ONNX.

        Subclasses com assinaturas de forward diferentes devem sobrescrever
        este método.
        """
        raise NotImplementedError("Subclasses devem implementar export_onnx().")


class VehicleForecastingMultiHeadModel(VehicleForecastingBaseModel):
    """
    Modelo baseline multi-saída para previsão de consumo de veículos.

    Herda encoders, fusão, cabeças e optimizer de ``VehicleForecastingBaseModel``.
    Recebe 2 inputs: ``(X_general, X_recent)``.

    Perda: ``loss = loss_heads + α × loss_daily``, ponderada por ``sample_weight``.
    """

    def __init__(
        self,
        n_general_features: int,
        n_recent_features: int,
        n_heads: int,
        n_daily: int,
        hidden_dim: int = 64,
        conv_filters: int = 32,
        conv_kernel: int = 3,
        conv_layers: int = 2,
        dropout: float = 0.1,
        loss_heads_type: Literal['huber', 'mse', 'mae'] = 'huber',
        loss_daily_type: Literal['mae', 'mse', 'huber'] = 'mae',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha_daily: float = 0.3,
    ):
        super().__init__(
            n_general_features=n_general_features,
            n_recent_features=n_recent_features,
            n_heads=n_heads,
            n_daily=n_daily,
            hidden_dim=hidden_dim,
            conv_filters=conv_filters,
            conv_kernel=conv_kernel,
            conv_layers=conv_layers,
            dropout=dropout,
            loss_heads_type=loss_heads_type,
            loss_daily_type=loss_daily_type,
            lr=lr,
            weight_decay=weight_decay,
            alpha_daily=alpha_daily,
        )
        self.save_hyperparameters()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X_general: torch.Tensor, X_recent: torch.Tensor):
        """
        Parameters
        ----------
        X_general : (B, Dg)  — perfil estrutural
        X_recent  : (B, T, F) — histórico recente

        Returns
        -------
        y_heads_pred : (B, H)
        y_daily_pred : (B, D)
        """
        fused = self._encode(X_general, X_recent)
        return self.head_agg(fused), self.head_daily(fused)

    def _compute_loss(self, batch):
        X_general, X_recent, y_heads, y_daily, weights = batch
        y_heads_pred, y_daily_pred = self(X_general, X_recent)
        return self._compute_combined_loss(
            y_heads_pred, y_daily_pred, y_heads, y_daily, weights,
        )

    def predict_step(self, batch, batch_idx):
        X_general, X_recent, _y_heads, _y_daily, _weights = batch
        return self(X_general, X_recent)

    # ------------------------------------------------------------------
    # Exportação ONNX (2 inputs)
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        path: str | Path,
        n_recent_steps: int = 7,
        opset_version: int = 17,
        verify: bool = True,
    ) -> Path:
        """Exporta o modelo baseline para ONNX (2 inputs)."""
        self.eval()

        device = next(self.parameters()).device
        X_general = torch.randn(2, self.hparams.n_general_features, device=device)
        X_recent = torch.randn(2, n_recent_steps, self.hparams.n_recent_features, device=device)

        onnx_path = Path(path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self,
            (X_general, X_recent),
            str(onnx_path),
            input_names=["X_general", "X_recent"],
            output_names=["y_heads", "y_daily"],
            dynamic_axes={
                "X_general": {0: "batch"},
                "X_recent": {0: "batch", 1: "seq_len"},
                "y_heads": {0: "batch"},
                "y_daily": {0: "batch"},
            },
            opset_version=opset_version,
        )

        if verify:
            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path))
            ort_out = sess.run(
                None,
                {
                    "X_general": X_general.cpu().numpy(),
                    "X_recent": X_recent.cpu().numpy(),
                },
            )

            with torch.no_grad():
                pt_heads, pt_daily = self(X_general, X_recent)

            np.testing.assert_allclose(
                ort_out[0], pt_heads.cpu().numpy(), rtol=1e-4, atol=1e-5,
            )
            np.testing.assert_allclose(
                ort_out[1], pt_daily.cpu().numpy(), rtol=1e-4, atol=1e-5,
            )

        return onnx_path


# ======================================================================
# SAFE MoE — correções por cluster apenas para y_heads
# ======================================================================


class VehicleForecastingSafeMoEModel(VehicleForecastingBaseModel):
    """
    SAFE MoE — base global + correções por cluster apenas para ``y_heads``.

    Herda encoders, fusão, cabeças globais e optimizer de
    ``VehicleForecastingBaseModel``. Acrescenta ``delta_heads``
    (K camadas lineares inicializadas a zero) cujas saídas são
    ponderadas por ``cluster_proba`` (gate fixo).

    Parameters
    ----------
    n_general_features : int
        Dimensão de ``X_general_struct`` (sem ``cluster_*``).
    n_clusters : int
        Número de clusters (K).
    n_recent_features, n_heads, n_daily, hidden_dim, conv_filters,
    conv_kernel, conv_layers, dropout, loss_heads_type, loss_daily_type,
    lr, weight_decay, alpha_daily :
        Idênticos ao ``VehicleForecastingBaseModel``.
    """

    def __init__(
        self,
        n_general_features: int,
        n_clusters: int,
        n_recent_features: int,
        n_heads: int,
        n_daily: int,
        hidden_dim: int = 64,
        conv_filters: int = 32,
        conv_kernel: int = 3,
        conv_layers: int = 2,
        dropout: float = 0.1,
        loss_heads_type: Literal['huber', 'mse', 'mae'] = 'huber',
        loss_daily_type: Literal['mae', 'mse', 'huber'] = 'mae',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha_daily: float = 0.3,
    ):
        super().__init__(
            n_general_features=n_general_features,
            n_recent_features=n_recent_features,
            n_heads=n_heads,
            n_daily=n_daily,
            hidden_dim=hidden_dim,
            conv_filters=conv_filters,
            conv_kernel=conv_kernel,
            conv_layers=conv_layers,
            dropout=dropout,
            loss_heads_type=loss_heads_type,
            loss_daily_type=loss_daily_type,
            lr=lr,
            weight_decay=weight_decay,
            alpha_daily=alpha_daily,
        )
        self.save_hyperparameters()

        # ---- Cabeças de correção por cluster (SAFE MoE, apenas y_heads) ----
        self.delta_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_heads) for _ in range(n_clusters)
        ])
        # Inicialização zero → começa como baseline puro
        for delta in self.delta_heads:
            nn.init.zeros_(delta.weight)
            nn.init.zeros_(delta.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        X_general_struct: torch.Tensor,
        X_cluster_proba: torch.Tensor,
        X_recent: torch.Tensor,
    ):
        """
        Parameters
        ----------
        X_general_struct : (B, Dg) — features estruturais (sem cluster_*)
        X_cluster_proba  : (B, K)  — probabilidades de cluster (gate fixo)
        X_recent         : (B, T, F) — histórico recente

        Returns
        -------
        y_heads_pred : (B, H)
        y_daily_pred : (B, D)
        """
        fused = self._encode(X_general_struct, X_recent)

        # Saídas globais
        y_heads_base = self.head_agg(fused)
        y_daily = self.head_daily(fused)

        # SAFE MoE: correções por cluster (apenas y_heads)
        deltas = torch.stack([delta(fused) for delta in self.delta_heads], dim=1)  # (B, K, H)
        gates = X_cluster_proba.unsqueeze(-1)  # (B, K, 1)
        corrections = (gates * deltas).sum(dim=1)  # (B, H)

        return y_heads_base + corrections, y_daily

    # ------------------------------------------------------------------
    # Loss e predict
    # ------------------------------------------------------------------

    def _compute_loss(self, batch):
        X_general_struct, X_cluster_proba, X_recent, y_heads, y_daily, weights = batch
        y_heads_pred, y_daily_pred = self(X_general_struct, X_cluster_proba, X_recent)
        return self._compute_combined_loss(
            y_heads_pred, y_daily_pred, y_heads, y_daily, weights,
        )

    def predict_step(self, batch, batch_idx):
        X_general_struct, X_cluster_proba, X_recent, _yh, _yd, _w = batch
        return self(X_general_struct, X_cluster_proba, X_recent)

    # ------------------------------------------------------------------
    # Exportação ONNX (3 inputs)
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        path: str | Path,
        n_recent_steps: int = 7,
        opset_version: int = 18,
        verify: bool = True,
    ) -> Path:
        """Exporta o modelo SAFE MoE para ONNX (3 inputs)."""
        self.eval()

        device = next(self.parameters()).device
        X_general = torch.randn(2, self.hparams.n_general_features, device=device)
        X_cluster = torch.randn(2, self.hparams.n_clusters, device=device)
        X_recent = torch.randn(2, n_recent_steps, self.hparams.n_recent_features, device=device)

        onnx_path = Path(path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self,
            (X_general, X_cluster, X_recent),
            str(onnx_path),
            input_names=["X_general_struct", "X_cluster_proba", "X_recent"],
            output_names=["y_heads", "y_daily"],
            dynamic_axes={
                "X_general_struct": {0: "batch"},
                "X_cluster_proba": {0: "batch"},
                "X_recent": {0: "batch", 1: "seq_len"},
                "y_heads": {0: "batch"},
                "y_daily": {0: "batch"},
            },
            opset_version=opset_version,
        )

        if verify:
            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path))
            feeds = {
                "X_general_struct": X_general.cpu().numpy(),
                "X_cluster_proba": X_cluster.cpu().numpy(),
                "X_recent": X_recent.cpu().numpy(),
            }
            ort_out = sess.run(None, feeds)

            with torch.no_grad():
                pt_heads, pt_daily = self(X_general, X_cluster, X_recent)

            np.testing.assert_allclose(
                ort_out[0], pt_heads.cpu().numpy(), rtol=1e-4, atol=1e-5,
            )
            np.testing.assert_allclose(
                ort_out[1], pt_daily.cpu().numpy(), rtol=1e-4, atol=1e-5,
            )

        return onnx_path
