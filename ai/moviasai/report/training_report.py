"""
Gerador de relatório PDF para o pipeline de treinamento.

Respeita o horizonte misto do modelo:
- y_daily  → apenas os primeiros ``daily_horizon`` dias (ex.: 7)
- y_heads  → agregados semanais (ex.: semanas 1–4)

Páginas por split (val / test)
------------------------------
1. Capa com configurações
2. Dataset — distribuição temporal, split, pesos
3. Curvas de treinamento (loss por época)
4. Tabela de métricas (MAE/RMSE por head + daily + decisão)
5. PLOT 1 — Histograma do erro em dias (DECISÃO)
6. PLOT 2 — Boxplot do erro em dias (RISCO) por k
7. PLOT 3 — Scatter consumo agregado REAL vs PREDITO por semana
8. PLOT 4 — Curvas reais vs preditas (CASOS) — daily + barras semanais
9. PLOT 5 — Erro acumulado por semana
10. PLOT 8 — Erro em dias vs p95 (viés por porte)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


class TrainingReportGenerator:
    """Gera relatório PDF a partir de um TrainingPipeline executado."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(self, pipeline) -> Path | None:
        pdf_path = self.output_dir / f"training_report_{pipeline.tag}_{pipeline.target}.pdf"

        try:
            with PdfPages(str(pdf_path)) as pdf:
                self._page_cover(pdf, pipeline)
                self._page_dataset(pdf, pipeline)
                self._page_training_curves(pdf, pipeline)

                for split_name in ("val", "test"):
                    split_data = (pipeline.metrics or {}).get(split_name)
                    if split_data is None:
                        continue
                    # Ordem conforme spec
                    self._page_metrics_table(pdf, split_name, split_data)
                    self._page_error_histogram(pdf, split_name, split_data)       # Plot 1
                    self._page_error_boxplot(pdf, split_name, split_data)         # Plot 2
                    self._page_scatter_heads_weekly(pdf, split_name, split_data)  # Plot 3
                    self._page_vehicle_curves(pdf, split_name, split_data)        # Plot 4
                    self._page_weekly_cumulative_error(pdf, split_name, split_data)  # Plot 5
                    self._page_error_vs_p95(pdf, split_name, split_data)          # Plot 8
        except PermissionError:
            print(f"⚠ Não foi possível gerar o PDF: sem permissão de escrita em {pdf_path}")
            return None

        print(f"Relatório gerado: {pdf_path}")
        return pdf_path

    # ------------------------------------------------------------------
    # Páginas fixas
    # ------------------------------------------------------------------

    def _page_cover(self, pdf: PdfPages, pipeline):
        """Capa com resumo de configurações."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")

        mp = pipeline.model_hparams
        tp = pipeline.training_hparams
        dm = pipeline.data_module

        lines = [
            "",
            "RELATÓRIO DE TREINAMENTO",
            f"{type(pipeline.model).__name__} — {pipeline.target.upper()}",
            "",
            f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "",
            "=" * 52,
            "",
            "CONFIGURAÇÃO DO MODELO",
            f"  hidden_dim:      {mp.get('hidden_dim')}",
            f"  conv_filters:    {mp.get('conv_filters')}",
            f"  conv_kernel:     {mp.get('conv_kernel')}",
            f"  conv_layers:     {mp.get('conv_layers')}",
            f"  dropout:         {mp.get('dropout')}",
            f"  loss_heads:      {mp.get('loss_heads_type')}",
            f"  loss_daily:      {mp.get('loss_daily_type')}",
            f"  alpha_daily:     {mp.get('alpha_daily')}",
            "",
            "TREINAMENTO",
            f"  max_epochs:      {tp.get('max_epochs')}",
            f"  batch_size:      {tp.get('batch_size')}",
            f"  lr:              {mp.get('lr')}",
            f"  weight_decay:    {mp.get('weight_decay')}",
            f"  early_stopping:  patience={tp.get('early_stopping', {}).get('patience')}",
            f"  seed:            {tp.get('seed')}",
        ]

        if dm is not None:
            lines += [
                "",
                "DIMENSÕES",
                f"  X_general:       {dm.n_general_features} features",
                f"  X_recent:        ({dm.n_recent_steps}, {dm.n_recent_features})",
                f"  y_heads:         {dm.n_heads} cabeças",
                f"  y_daily:         {dm.n_daily} dias",
            ]

        ax.text(
            0.5, 0.5, "\n".join(lines),
            transform=ax.transAxes, fontsize=11,
            ha="center", va="center", fontfamily="monospace",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _page_dataset(self, pdf: PdfPages, pipeline):
        """Distribuição temporal, proporções do split e pesos."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Dataset", fontsize=14, fontweight="bold")
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        ax = fig.add_subplot(gs[0, :])
        if pipeline.dataset is not None:
            meta = pipeline.dataset.metadata
            weekly = meta.groupby("ref_date").size()
            ax.bar(range(len(weekly)), weekly.values, color="steelblue", alpha=0.8)
            ax.set_xticks(range(0, len(weekly), max(1, len(weekly) // 10)))
            ax.set_xticklabels(
                [str(d)[:10] for d in weekly.index[::max(1, len(weekly) // 10)]],
                rotation=45, fontsize=8,
            )
            ax.set_ylabel("Amostras")
            ax.set_title("Amostras por semana de referência (dataset completo)")
            ax.grid(True, alpha=0.3, axis="y")

        ax = fig.add_subplot(gs[1, 0])
        sizes, labels, colors = [], [], []
        for name, ds, c in [
            ("Treino", pipeline.train_ds, "#4472C4"),
            ("Validação", pipeline.val_ds, "#ED7D31"),
            ("Teste", pipeline.test_ds, "#A5A5A5"),
        ]:
            if ds is not None:
                sizes.append(len(ds))
                labels.append(f"{name}\n{len(ds):,}")
                colors.append(c)
        if sizes:
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 9})
            ax.set_title("Divisão treino / val / teste")

        ax = fig.add_subplot(gs[1, 1])
        if pipeline.dataset is not None:
            ax.hist(pipeline.dataset.weights, bins=40, color="steelblue",
                    alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_xlabel("Peso")
            ax.set_ylabel("Frequência")
            ax.set_title("Distribuição dos pesos amostrais")
            ax.grid(True, alpha=0.3, axis="y")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _page_training_curves(self, pdf: PdfPages, pipeline):
        """Loss de treino e validação por época."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Curvas de Treinamento", fontsize=14, fontweight="bold")
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        trainer = pipeline.trainer
        if trainer is None:
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, "Trainer não disponível",
                    ha="center", va="center", fontsize=14)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return

        logged = _extract_logged_metrics(trainer)

        ax = fig.add_subplot(gs[0, :])
        if "train_loss" in logged:
            ax.plot(logged["train_loss"], label="train_loss",
                    color="#4472C4", linewidth=1.2)
        if "val_loss" in logged:
            ax.plot(logged["val_loss"], label="val_loss",
                    color="#ED7D31", linewidth=1.2)
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.set_title("Loss por época"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 0])
        for key, color in [("train_loss_heads", "#4472C4"),
                           ("val_loss_heads", "#ED7D31")]:
            if key in logged:
                ax.plot(logged[key], label=key, color=color, linewidth=1.0)
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.set_title("Loss — Heads"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        for key, color in [("train_loss_daily", "#4472C4"),
                           ("val_loss_daily", "#ED7D31")]:
            if key in logged:
                ax.plot(logged[key], label=key, color=color, linewidth=1.0)
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.set_title("Loss — Daily"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Tabela de métricas completa
    # ------------------------------------------------------------------

    def _page_metrics_table(self, pdf: PdfPages, name: str, split_data: Dict):
        """Tabela com MAE/RMSE por head, daily, e métricas de decisão."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(f"Métricas — {name.upper()}", fontsize=14, fontweight="bold")

        head_days = split_data["head_days"]
        heads_mae = split_data["heads_mae"]
        heads_rmse = split_data["heads_rmse"]
        daily_mae = split_data["daily_mae"]
        daily_rmse = split_data["daily_rmse"]
        maint = split_data["maintenance"]
        H = len(heads_mae)
        D = len(daily_mae)

        # --- Sub-tabela 1: heads ---
        ax1 = fig.add_axes([0.08, 0.62, 0.84, 0.28])
        ax1.axis("off")
        ax1.set_title("y_heads (agregado semanal)", fontsize=11, fontweight="bold",
                       loc="left", pad=10)
        rows_h = []
        for h in range(H):
            rows_h.append([
                f"head_{h} ({head_days[h]}d)",
                f"{heads_mae[h]:.2f}",
                f"{heads_rmse[h]:.2f}",
            ])
        t1 = ax1.table(cellText=rows_h,
                        colLabels=["Head", "MAE", "RMSE"],
                        cellLoc="center", loc="upper center",
                        colWidths=[0.3, 0.2, 0.2])
        t1.auto_set_font_size(False); t1.set_fontsize(10); t1.scale(1, 1.8)
        for j in range(3):
            t1[(0, j)].set_facecolor("#4472C4")
            t1[(0, j)].set_text_props(weight="bold", color="white")

        # --- Sub-tabela 2: daily ---
        ax2 = fig.add_axes([0.08, 0.48, 0.84, 0.12])
        ax2.axis("off")
        ax2.set_title(f"y_daily (dias 1–{D})", fontsize=11, fontweight="bold",
                       loc="left", pad=10)
        t2 = ax2.table(
            cellText=[[f"{daily_mae.mean():.2f}", f"{daily_rmse.mean():.2f}"]],
            colLabels=["MAE médio", "RMSE médio"],
            cellLoc="center", loc="upper center",
            colWidths=[0.25, 0.25])
        t2.auto_set_font_size(False); t2.set_fontsize(10); t2.scale(1, 1.8)
        for j in range(2):
            t2[(0, j)].set_facecolor("#ED7D31")
            t2[(0, j)].set_text_props(weight="bold", color="white")

        # --- Sub-tabela 3: decisão ---
        ax3 = fig.add_axes([0.05, 0.08, 0.9, 0.35])
        ax3.axis("off")
        ax3.set_title("Decisão de manutenção", fontsize=11, fontweight="bold",
                       loc="left", pad=10)
        rows_m = []
        for k_label in ("k2", "k3", "k4"):
            m = maint[k_label]
            k = int(k_label[1])
            rows_m.append([
                f"{k} semanas",
                f"{m['mean_error']:+.2f}",
                f"{m['mae_days']:.2f}",
                f"{m['p90_error']:+.2f}",
                f"{m['pct_late']:.1f}%",
                f"{m['pct_early']:.1f}%",
            ])
        t3 = ax3.table(
            cellText=rows_m,
            colLabels=["Limite", "Erro Médio\n(dias)", "MAE\n(dias)",
                        "P90 Erro\n(dias)", "% Atrasos", "% Adiantam."],
            cellLoc="center", loc="upper center",
            colWidths=[0.14, 0.14, 0.14, 0.14, 0.14, 0.14])
        t3.auto_set_font_size(False); t3.set_fontsize(10); t3.scale(1, 2.2)
        for j in range(6):
            t3[(0, j)].set_facecolor("#70AD47")
            t3[(0, j)].set_text_props(weight="bold", color="white")

        ax3.text(0.5, -0.02,
                 "Limite = k × P95_diário × 7  |  "
                 "Erro > 0 = atraso (risco)  |  Erro < 0 = adiantamento (custo)",
                 transform=ax3.transAxes, fontsize=9,
                 ha="center", style="italic", color="#555555")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ==================================================================
    # PLOT 1 — Histograma do erro em dias (DECISÃO)
    # ==================================================================

    def _page_error_histogram(self, pdf: PdfPages, name: str, split_data: Dict):
        maint = split_data["maintenance"]
        colors = {"k2": "#4472C4", "k3": "#ED7D31", "k4": "#70AD47"}

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"PLOT 1 — Histograma do Erro em Dias (DECISÃO) — {name.upper()}",
            fontsize=13, fontweight="bold",
        )

        for idx, k_label in enumerate(("k2", "k3", "k4")):
            ax = axes[idx]
            errors = maint[k_label]["errors"]
            k = int(k_label[1])

            ax.hist(errors, bins=50, color=colors[k_label],
                    alpha=0.75, edgecolor="black", linewidth=0.3)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="zero")
            ax.axvline(np.nanmean(errors), color="black", linestyle=":",
                       linewidth=1, label=f"média={np.nanmean(errors):+.1f}")
            ax.axvline(np.nanmedian(errors), color="purple", linestyle="-.",
                       linewidth=1, label=f"mediana={np.nanmedian(errors):+.1f}")
            ax.set_xlabel("Erro (dias): predito − real")
            ax.set_ylabel("Frequência")
            ax.set_title(f"k = {k} semanas")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ==================================================================
    # PLOT 2 — Boxplot do erro em dias (RISCO) por k
    # ==================================================================

    def _page_error_boxplot(self, pdf: PdfPages, name: str, split_data: Dict):
        maint = split_data["maintenance"]
        colors_list = ["#4472C4", "#ED7D31", "#70AD47"]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(
            f"PLOT 2 — Boxplot do Erro em Dias (RISCO) — {name.upper()}",
            fontsize=13, fontweight="bold",
        )

        all_errors = [maint[f"k{k}"]["errors"] for k in (2, 3, 4)]
        bp = ax.boxplot(all_errors,
                        labels=[f"k={k}" for k in (2, 3, 4)],
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, c in zip(bp["boxes"], colors_list):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_ylabel("Erro (dias): predito − real")
        ax.set_title("Comparação entre limites — mediana, IQR, P90")
        ax.grid(True, alpha=0.3, axis="y")

        for i, k in enumerate((2, 3, 4)):
            errs = maint[f"k{k}"]["errors"]
            med = np.nanmedian(errs)
            p90 = np.nanpercentile(errs, 90)
            ax.annotate(f"med={med:+.1f}\nP90={p90:+.1f}",
                        xy=(i + 1, p90), fontsize=7, ha="center",
                        va="bottom", color="#333333")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ==================================================================
    # PLOT 3 — Scatter consumo agregado REAL vs PREDITO por semana
    # ==================================================================

    def _page_scatter_heads_weekly(self, pdf: PdfPages, name: str, split_data: Dict):
        """Um scatter por head: y_heads_true vs y_heads_pred (sem fabricar dias)."""
        y_true = split_data["y_heads_true"]   # (N, H)
        y_pred = split_data["y_heads_pred"]
        head_days = split_data["head_days"]
        H = y_true.shape[1]

        ncols = min(H, 4)
        nrows = (H + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows))
        fig.suptitle(
            f"PLOT 3 — Σ Real vs Σ Predito por Semana — {name.upper()}",
            fontsize=13, fontweight="bold",
        )
        axes = np.atleast_2d(axes)

        for h in range(H):
            ax = axes[h // ncols, h % ncols]
            yt, yp = y_true[:, h], y_pred[:, h]
            ax.scatter(yt, yp, alpha=0.3, s=12, color="steelblue", edgecolors="none")
            lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)
            mae = np.mean(np.abs(yt - yp))
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.text(0.05, 0.92, f"R²={r2:.3f}\nMAE={mae:.1f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.set_xlabel("Real"); ax.set_ylabel("Predito")
            ax.set_title(f"Semana {h+1} ({head_days[h]}d)")
            ax.grid(True, alpha=0.3)

        for h in range(H, nrows * ncols):
            axes[h // ncols, h % ncols].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ==================================================================
    # PLOT 4 — Curvas reais vs preditas (CASOS) — horizonte misto
    # ==================================================================

    def _page_vehicle_curves(self, pdf: PdfPages, name: str, split_data: Dict):
        """
        Para 12 veículos aleatórios:
        - Dias 1–D: curva diária real vs y_daily
        - Semanas 2–H: barras do agregado real vs y_heads
        """
        y_daily_true = split_data["y_daily_true"]   # (N, D)
        y_daily_pred = split_data["y_daily_pred"]
        y_heads_true = split_data["y_heads_true"]   # (N, H)
        y_heads_pred = split_data["y_heads_pred"]
        head_days = split_data["head_days"]
        metadata = split_data["metadata"]

        N = y_daily_true.shape[0]
        D = y_daily_true.shape[1]
        H = y_heads_true.shape[1]
        n_vehicles = min(12, N)
        rng = np.random.RandomState(42)
        idxs = rng.choice(N, size=n_vehicles, replace=False)

        ncols = 3
        nrows = (n_vehicles + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(
            f"PLOT 4 — Curvas Reais vs Preditas (CASOS) — {name.upper()}",
            fontsize=13, fontweight="bold",
        )

        for pos, i in enumerate(idxs):
            ax = axes[pos // ncols, pos % ncols]
            vid = metadata.iloc[i].get("veiculo_id", i) if metadata is not None else i

            # Parte 1: curva diária (dias 1–D)
            days = np.arange(1, D + 1)
            ax.plot(days, y_daily_true[i], color="#4472C4", linewidth=1.0,
                    label="real (daily)", marker=".", markersize=3)
            ax.plot(days, y_daily_pred[i], color="#ED7D31", linewidth=1.0,
                    label="pred (daily)", linestyle="--", marker=".", markersize=3)

            # Parte 2: barras semanais (heads 1–H-1, ou seja semanas 2+)
            bar_width = 2.0
            offset = head_days[0]  # fim da semana 1
            for h in range(1, H):
                mid = offset + head_days[h] / 2.0
                ax.bar(mid - bar_width / 4, y_heads_true[i, h],
                       width=bar_width / 2, color="#4472C4", alpha=0.4,
                       label="real (head)" if h == 1 else None)
                ax.bar(mid + bar_width / 4, y_heads_pred[i, h],
                       width=bar_width / 2, color="#ED7D31", alpha=0.4,
                       label="pred (head)" if h == 1 else None)
                offset += head_days[h]

            ax.set_xlabel("dia / posição", fontsize=7)
            ax.set_ylabel("consumo", fontsize=7)
            ax.set_title(f"v={vid}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, loc="upper left")
            ax.grid(True, alpha=0.3)

        for pos in range(n_vehicles, nrows * ncols):
            axes[pos // ncols, pos % ncols].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ==================================================================
    # PLOT 5 — Erro acumulado por semana (y_heads)
    # ==================================================================

    def _page_weekly_cumulative_error(self, pdf: PdfPages, name: str, split_data: Dict):
        """
        Σ_predito − Σ_real acumulado ao longo das semanas.
        Usa exclusivamente y_heads (sem fabricar dias).
        """
        y_heads_true = split_data["y_heads_true"]   # (N, H)
        y_heads_pred = split_data["y_heads_pred"]
        head_days = split_data["head_days"]
        H = y_heads_true.shape[1]

        # Acumulado semanal
        cum_true = np.cumsum(y_heads_true, axis=1)   # (N, H)
        cum_pred = np.cumsum(y_heads_pred, axis=1)
        cum_error = cum_pred - cum_true               # (N, H)

        mean_err = np.mean(cum_error, axis=0)
        p10 = np.percentile(cum_error, 10, axis=0)
        p25 = np.percentile(cum_error, 25, axis=0)
        p75 = np.percentile(cum_error, 75, axis=0)
        p90 = np.percentile(cum_error, 90, axis=0)

        weeks = np.arange(1, H + 1)
        week_labels = [f"Sem {w}\n({head_days[w-1]}d)" for w in weeks]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.suptitle(
            f"PLOT 5 — Erro Acumulado por Semana — {name.upper()}",
            fontsize=13, fontweight="bold",
        )

        ax.fill_between(weeks, p10, p90, alpha=0.15, color="#4472C4", label="P10–P90")
        ax.fill_between(weeks, p25, p75, alpha=0.3, color="#4472C4", label="P25–P75")
        ax.plot(weeks, mean_err, color="#4472C4", linewidth=1.5,
                marker="o", markersize=5, label="Erro médio")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xticks(weeks)
        ax.set_xticklabels(week_labels)
        ax.set_xlabel("Semana"); ax.set_ylabel("Σ predito − Σ real")
        ax.set_title("Evolução do erro acumulado — cresce lentamente = bom")
        ax.legend(); ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ==================================================================
    # PLOT 8 (opcional) — Erro em dias vs p95
    # ==================================================================

    def _page_error_vs_p95(self, pdf: PdfPages, name: str, split_data: Dict):
        """Scatter: erro_em_dias vs p95 — verificar viés por porte."""
        maint = split_data["maintenance"]
        upper = split_data["upper"]
        colors = {"k2": "#4472C4", "k3": "#ED7D31", "k4": "#70AD47"}

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"PLOT 8 — Erro em Dias vs p95 — {name.upper()}",
            fontsize=13, fontweight="bold",
        )

        for idx, k_label in enumerate(("k2", "k3", "k4")):
            ax = axes[idx]
            errors = maint[k_label]["errors"]
            k = int(k_label[1])

            ax.scatter(upper, errors, alpha=0.25, s=10,
                       color=colors[k_label], edgecolors="none")
            ax.axhline(0, color="red", linestyle="--", linewidth=1)

            if len(upper) > 2:
                z = np.polyfit(upper, errors, 1)
                p = np.poly1d(z)
                x_line = np.linspace(upper.min(), upper.max(), 100)
                ax.plot(x_line, p(x_line), color="black", linewidth=1,
                        linestyle=":", label=f"slope={z[0]:.3f}")
                ax.legend(fontsize=7)

            ax.set_xlabel("p95 (consumo diário)")
            ax.set_ylabel("Erro (dias)")
            ax.set_title(f"k = {k} semanas")
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ======================================================================
# Helpers
# ======================================================================


def _extract_logged_metrics(trainer) -> Dict[str, List[float]]:
    """
    Extrai métricas epoch-level do CSVLogger do Lightning.

    Retorna dict de listas: ``{'train_loss': [...], 'val_loss': [...], ...}``
    """
    result: Dict[str, List[float]] = {}

    # CSVLogger grava em metrics.csv
    # Train e val são logados em linhas distintas (step vs epoch),
    # por isso agrupamos por época para alinhar as curvas.
    try:
        for lg in (trainer.loggers if hasattr(trainer, "loggers") else []):
            if isinstance(lg, __import__("pytorch_lightning.loggers", fromlist=["CSVLogger"]).CSVLogger):
                metrics_path = Path(lg.log_dir) / "metrics.csv"
                if metrics_path.exists():
                    df = pd.read_csv(metrics_path)
                    if "epoch" in df.columns:
                        grouped = df.groupby("epoch").mean(numeric_only=True)
                        for col in grouped.columns:
                            if col == "step":
                                continue
                            vals = grouped[col].dropna().tolist()
                            if vals:
                                result[col] = vals
                    else:
                        for col in df.columns:
                            if col in ("step", "epoch"):
                                continue
                            vals = df[col].dropna().tolist()
                            if vals:
                                result[col] = vals
                    if result:
                        return result
    except Exception:
        pass

    # Fallback: logged_metrics (último valor apenas)
    try:
        logged = trainer.logged_metrics or {}
        for k, v in logged.items():
            if isinstance(v, (int, float)):
                result.setdefault(k, []).append(float(v))
            elif hasattr(v, "item"):
                result.setdefault(k, []).append(float(v.item()))
    except Exception:
        pass

    return result
