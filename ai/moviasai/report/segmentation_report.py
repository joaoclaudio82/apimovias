"""
Gerador de relatório PDF para o pipeline de segmentação de veículos.

Páginas
-------
1. Capa
2-N. Console output
3. Análise de anomalias
4. Clustering etapa 1 (PCA + t-SNE + distribuição + métricas)
5. Classificação etapa 1 (confusão + importância + comparação + métricas)
6. Regiões de decisão + ECDF
7. Clustering etapa 2 — KM
8. Classificação etapa 2 — KM
9. Clustering etapa 2 — H
10. Classificação etapa 2 — H
11-N. Estatísticas dos segmentos
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix

from moviasai.data.data_quality import SeriesQuality


class SegmentationReportGenerator:
    """Gera relatório PDF a partir de um VehicleSegmentationPipeline executado."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(self, pipeline, console_output: str) -> Path:
        pdf_path = self.output_dir / "relatorio_completo.pdf"

        print(f"\n{'='*80}")
        print("GERANDO RELATÓRIO PDF")
        print(f"{'='*80}\n")

        with PdfPages(pdf_path) as pdf:
            self._page_cover(pdf)
            self._page_console(pdf, console_output)

            if (pipeline.df_features["quality"] == SeriesQuality.OUTLIER).sum() > 0:
                self._page_anomalies(pdf, pipeline)

            if pipeline.metric_clusterer:
                self._page_clustering(
                    pdf, pipeline.metric_clusterer, "Etapa 1: Identificação de Métrica"
                )

            if pipeline.metric_classifier and pipeline.metric_classifier.results:
                self._page_classification(pdf, pipeline.metric_classifier)
                self._page_decision_boundaries(pdf, pipeline)

            if pipeline.km_clusterer:
                self._page_clustering(
                    pdf, pipeline.km_clusterer, "Etapa 2: Segmentação - KM"
                )
            if pipeline.km_classifier and pipeline.km_classifier.results:
                self._page_classification(pdf, pipeline.km_classifier)

            if pipeline.h_clusterer:
                self._page_clustering(
                    pdf, pipeline.h_clusterer, "Etapa 2: Segmentação - H"
                )
            if pipeline.h_classifier and pipeline.h_classifier.results:
                self._page_classification(pdf, pipeline.h_classifier)

            self._page_segment_stats(pdf, pipeline)

        print(f"✓ PDF gerado: {pdf_path}")
        print(f"  Tamanho: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
        print()

        return pdf_path

    # ------------------------------------------------------------------
    # Capa
    # ------------------------------------------------------------------

    def _page_cover(self, pdf: PdfPages):
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")

        lines = [
            "", "", "",
            "RELATÓRIO DE ANÁLISE",
            "DE VEÍCULOS",
            "",
            "Detecção de Anomalias, Clusterização e Classificação",
            "em Duas Etapas",
            "", "",
            f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            "", "",
            "=" * 50,
            "",
            "Gerado automaticamente pelo",
            "Vehicle Segmentation Pipeline",
        ]
        ax.text(
            0.5, 0.5, "\n".join(lines),
            transform=ax.transAxes, fontsize=16,
            ha="center", va="center", fontfamily="sans-serif", fontweight="bold",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _page_console(self, pdf: PdfPages, console_output: str):
        lines = console_output.split("\n")
        per_page = 60

        for page_num, i in enumerate(range(0, len(lines), per_page), start=1):
            batch = lines[i : i + per_page]
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.05, 0.95, "\n".join(batch),
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top", fontfamily="monospace", wrap=True,
            )
            ax.text(
                0.95, 0.02, f"Página {page_num}",
                transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom", style="italic",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    # ------------------------------------------------------------------
    # Anomalias
    # ------------------------------------------------------------------

    def _page_anomalies(self, pdf: PdfPages, pipeline):
        df = pipeline.df_features
        outliers = df[df["quality"] == SeriesQuality.OUTLIER]
        normals = df[df["quality"] == SeriesQuality.VALID]
        anomalies = df[df["quality"] != SeriesQuality.VALID]

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Análise de Anomalias", fontsize=14, fontweight="bold")
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Tabela de anomalias
        ax = fig.add_subplot(gs[0, :])
        ax.axis("off")

        if len(anomalies) > 0:
            counts = anomalies["quality_reason"].value_counts()
            total = len(df)
            table_data = []
            for reason, count in counts.items():
                label = _wrap_text(reason or "Não especificado", 80)
                table_data.append([label, f"{count:,}", f"{count/total*100:.1f}%"])
            table_data.sort(key=lambda x: float(x[2].rstrip("%")), reverse=True)

            table = ax.table(
                cellText=table_data,
                colLabels=["Motivo", "Qtd.", "Prop. (%)"],
                cellLoc="left", loc="center",
                colWidths=[0.70, 0.15, 0.15],
            )
            _style_table(table, len(table_data))
            ax.set_title("Distribuição de Anomalias", fontsize=11, fontweight="bold", pad=15)
        else:
            ax.text(
                0.5, 0.5, "Nenhuma anomalia detectada",
                ha="center", va="center", fontsize=12, color="green", weight="bold",
            )

        # Scatter
        if "seg_mediana_km" in normals.columns and "seg_mediana_h" in normals.columns:
            ax = fig.add_subplot(gs[1, :])
            ax.scatter(
                normals["seg_mediana_km"], normals["seg_mediana_h"],
                alpha=0.5, s=20, label="Normal", color="blue",
            )
            if len(outliers) > 0:
                ax.scatter(
                    outliers["seg_mediana_km"], outliers["seg_mediana_h"],
                    alpha=0.8, s=40, label="Outlier", color="red", marker="x",
                )
            ax.set_xlabel("KM/dia (mediana)")
            ax.set_ylabel("Horas/dia (mediana)")
            ax.set_title("Detecção de Outliers - Espaço KM vs Horas", fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Clustering (genérico para etapa 1 e 2)
    # ------------------------------------------------------------------

    def _page_clustering(self, pdf: PdfPages, clusterer, title: str):
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.4)

        unique_labels = np.unique(clusterer.labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        # PCA
        ax = fig.add_subplot(gs[0, 0])
        _plot_projection(ax, clusterer.pca_coords, clusterer.labels, unique_labels, colors)
        var = clusterer.pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.set_title(f"PCA (Var: {var.sum():.1%})")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        # t-SNE
        ax = fig.add_subplot(gs[0, 1])
        _plot_projection(ax, clusterer.tsne_coords, clusterer.labels, unique_labels, colors)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        # Distribuição
        ax = fig.add_subplot(gs[1, 0])
        unique, counts = np.unique(clusterer.labels, return_counts=True)
        bars = ax.bar(unique, counts, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Quantidade de Veículos")
        ax.set_title("Distribuição dos Clusters")
        ax.grid(True, alpha=0.3, axis="y")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{int(h)}",
                    ha="center", va="bottom", fontsize=8)

        # Métricas
        ax = fig.add_subplot(gs[1, 1])
        ax.axis("off")
        m = clusterer.metrics
        text = [
            "Métricas de Qualidade:", "",
            f"Silhouette:        {m['silhouette']:8.4f}",
            f"Calinski-Harabasz: {m['calinski_harabasz']:8.2f}",
            f"Davies-Bouldin:    {m['davies_bouldin']:8.4f}",
            f"Inércia:           {m['inertia']:8.2f}",
            "", "Informações:",
            f"Clusters:          {m['n_clusters']:8d}",
            f"Amostras:          {m['n_samples']:8d}",
            f"Features:          {m['n_features']:8d}",
        ]
        ax.text(
            0.1, 0.9, "\n".join(text), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Classificação (genérico)
    # ------------------------------------------------------------------

    def _page_classification(self, pdf: PdfPages, classifier):
        best = classifier.results[classifier.best_model_name]
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(f"Classificação - {classifier.name}", fontsize=14, fontweight="bold")
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Confusion matrix
        ax = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(best["y_test"], best["y_pred_test"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar_kws={"label": "Contagem"})
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão - {classifier.best_model_name}",
                      fontsize=11, fontweight="bold")

        # 2. Feature importance
        ax = fig.add_subplot(gs[0, 1])
        importances, imp_type = _get_feature_importance(best["model"])
        if importances is not None:
            n_top = min(12, len(importances))
            indices = np.argsort(importances)[::-1][:n_top]
            ax.barh(range(len(indices)), importances[indices],
                    color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([classifier.feature_names[i] for i in indices], fontsize=8)
            ax.set_xlabel(imp_type)
            ax.set_title(f"Top {n_top} Features", fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis="x")
            for i, imp in enumerate(importances[indices]):
                ax.text(imp, i, f" {imp:.3f}", va="center", fontsize=7)
        else:
            ax.text(
                0.5, 0.5,
                "Feature Importance\nnão disponível\npara este modelo",
                ha="center", va="center", fontsize=12, color="gray",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
            )
            ax.axis("off")

        # 3. Comparação de modelos
        ax = fig.add_subplot(gs[1, 0])
        models = list(classifier.results.keys())
        f1_scores = [classifier.results[m]["f1_test"] for m in models]
        colors = ["gold" if m == classifier.best_model_name else "steelblue" for m in models]
        bars = ax.bar(range(len(models)), f1_scores, color=colors,
                      alpha=0.7, edgecolor="black", linewidth=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
        ax.set_ylabel("F1 Score (Teste)")
        ax.set_title("Comparação de Modelos", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{score:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # 4. Métricas detalhadas
        ax = fig.add_subplot(gs[1, 1])
        ax.axis("off")
        text = [
            f"Melhor Modelo:",
            f"{classifier.best_model_name}", "",
            "Métricas no Teste:",
            f"  Acurácia:  {best['acc_test']:.4f}",
            f"  F1 Score:  {best['f1_test']:.4f}", "",
            "Métricas no Treino:",
            f"  Acurácia:  {best['acc_train']:.4f}",
            f"  F1 Score:  {best['f1_train']:.4f}", "",
            "Cross-Validation (5-fold):",
            f"  F1 Médio:  {best['f1_cv_mean']:.4f}",
            f"  F1 Desvio: {best['f1_cv_std']:.4f}",
        ]
        ax.text(
            0.1, 0.9, "\n".join(text), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Regiões de decisão + ECDF (etapa 1)
    # ------------------------------------------------------------------

    def _page_decision_boundaries(self, pdf: PdfPages, pipeline):
        from matplotlib import gridspec
        from matplotlib.colors import ListedColormap
        from sklearn.base import clone
        from sklearn.decomposition import PCA

        classifier = pipeline.metric_classifier
        df_train = pipeline.df_clean.copy()
        df_all = pipeline.df_no_anomalies.copy()
        feature_cols = pipeline.type_extractor.feature_names
        idx_to_metric = pipeline.class_idx_to_metric

        df_train = df_train[df_train["metrica_predominante"].isin(["KM", "H"])]
        df_all = df_all[df_all["metrica_predominante"].isin(["KM", "H"])]

        if len(df_all) == 0 or len(df_train) == 0:
            return

        X_all = classifier.scaler.transform(df_all[feature_cols].values)
        y_all = df_all["cluster_metrica"].astype(int).values
        y_train = df_train["cluster_metrica"].astype(int).values
        train_mask = df_all.index.isin(df_train.index)

        pca = PCA(n_components=2, random_state=42)
        X_all_pca = pca.fit_transform(X_all)
        X_train_pca = X_all_pca[train_mask]

        fig = plt.figure(figsize=(10, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1.5], hspace=0.25)
        ax_main = fig.add_subplot(gs[0])
        ax_ecdf = fig.add_subplot(gs[1])

        palette = ["#1f77b4", "#ff7f0e"]
        cmap_light = ListedColormap(["#aec7e8", "#ffbb78"])

        # Grid de decisão
        h = 0.02
        x_min, x_max = X_all_pca[:, 0].min() - 1, X_all_pca[:, 0].max() + 1
        y_min, y_max = X_all_pca[:, 1].min() - 1, X_all_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        try:
            model_pca = clone(classifier.best_model)
            model_pca.fit(X_train_pca, y_train)
            Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax_main.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
            ax_main.contour(xx, yy, Z, colors="black", linewidths=1, alpha=0.5)
        except Exception:
            ax_main.text(
                0.5, 0.95, "Não foi possível gerar\nregiões de decisão",
                transform=ax_main.transAxes, ha="center", va="top",
                fontsize=9, color="red",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        # Pontos
        for class_idx in np.unique(y_all):
            class_mask = y_all == class_idx
            not_train = class_mask & (~train_mask)
            in_train = class_mask & train_mask

            if not_train.sum() > 0:
                ax_main.scatter(
                    X_all_pca[not_train, 0], X_all_pca[not_train, 1],
                    c=[palette[class_idx]], edgecolors="lightgray", linewidth=0.5,
                    s=30, alpha=0.3, label=f"{idx_to_metric[class_idx]} - outros",
                )
            if in_train.sum() > 0:
                ax_main.scatter(
                    X_all_pca[in_train, 0], X_all_pca[in_train, 1],
                    c=[palette[class_idx]], edgecolors="black", linewidth=0.8,
                    s=60, alpha=0.9, label=f"{idx_to_metric[class_idx]} - treino",
                )

        var = pca.explained_variance_ratio_
        ax_main.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax_main.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax_main.set_title("Regiões de Decisão (PCA)", fontweight="bold")
        ax_main.legend(fontsize=9)
        ax_main.grid(True, alpha=0.3)

        # ECDF
        for i, (label, name) in enumerate(idx_to_metric.items()):
            x = np.sort(df_train[df_train["cluster_metrica"] == label]["p_km"])
            if len(x) == 0:
                continue
            y = np.arange(1, len(x) + 1) / len(x)
            ax_ecdf.plot(x, y, label=name, color=palette[i])
        ax_ecdf.set_xlabel("p_km")
        ax_ecdf.set_ylabel("ECDF")
        ax_ecdf.set_title("ECDF de p_km no núcleo (df_clean)")
        ax_ecdf.legend()
        ax_ecdf.grid(True, alpha=0.3)

        fig.text(
            0.5, 0.01,
            f"PCA: {var.sum():.1%} var. explicada | "
            f"Total: {len(df_all):,} | Treino: {train_mask.sum():,}",
            ha="center", fontsize=9, style="italic", color="gray",
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Estatísticas dos segmentos
    # ------------------------------------------------------------------

    def _page_segment_stats(self, pdf: PdfPages, pipeline):
        df = pipeline.df_final
        if df is None:
            return

        segmentos = sorted(df["segmento"].unique())
        km_features = pipeline.km_extractor.feature_names
        h_features = pipeline.h_extractor.feature_names

        DISPLAY = {
            "media": "Média", "max": "Máximo", "mediana": "Mediana",
            "std": "Desvio Padrão", "score_continuidade": "Score Continuidade",
            "taxa_semanas_ativas": "Taxa Semanas Ativas",
            "taxa_dias_ativos": "Taxa Dias Ativos",
            "cv_gaps": "CV Gaps", "gap_medio": "Gap Médio", "gap_max": "Gap Máximo",
            "cv": "Coef. Variação", "p25": "Percentil 25", "p75": "Percentil 75",
            "iqr": "IQR", "tamanho_periodo": "Tamanho do Período",
        }

        for i in range(0, len(segmentos), 5):
            batch = segmentos[i : i + 5]
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle("Estatísticas dos Segmentos", fontsize=14, fontweight="bold")

            ax = fig.add_subplot(111)
            ax.axis("off")

            text = []
            for seg in batch:
                seg_data = df[df["segmento"] == seg]
                tipo = seg_data["metrica_predominante"].iloc[0]
                n = len(seg_data)
                pct = n / len(df) * 100

                text.append("")
                text.append("=" * 70)
                text.append(f"{seg} (n={n}, {pct:.1f}%)")
                text.append("=" * 70)

                features = km_features if tipo == "KM" else h_features
                for feat in features:
                    if feat not in seg_data.columns:
                        continue
                    label = DISPLAY.get(feat, feat.replace("_", " ").title())
                    value = seg_data[feat].mean()

                    if "taxa" in feat or feat in ["score_continuidade"]:
                        text.append(f"{label:25s}: {value:8.2%}")
                    elif "gap" in feat or feat == "tamanho_periodo":
                        text.append(f"{label:25s}: {value:8.1f} dias")
                    else:
                        text.append(f"{label:25s}: {value:8.2f}")

            ax.text(
                0.05, 0.95, "\n".join(text), transform=ax.transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close()


# ======================================================================
# Helpers
# ======================================================================


def _plot_projection(ax, coords, labels, unique_labels, colors):
    """Plota projeção 2D (PCA ou t-SNE) com centroides."""
    for i, label in enumerate(unique_labels):
        mask = labels == label
        n = mask.sum()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=f"C{label} (n={n})", alpha=0.6, s=40,
        )
        centroid = coords[mask].mean(axis=0)
        ax.scatter(
            centroid[0], centroid[1],
            c="black", marker="X", s=150, edgecolors="white", linewidth=2,
        )
        ax.text(
            centroid[0], centroid[1], f"{label}",
            fontsize=9, fontweight="bold", ha="center", va="center", color="white",
        )


def _get_feature_importance(model) -> Tuple[Optional[np.ndarray], str]:
    """Extrai importância de features de modelos sklearn."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_, "Importância (Gini/Entropy)"
    if hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) > 1:
            return np.abs(coef).mean(axis=0), "Coeficiente Absoluto"
        return np.abs(coef[0]), "Coeficiente Absoluto"
    return None, ""


def _wrap_text(text: str, max_chars: int = 80) -> str:
    """Quebra texto longo em múltiplas linhas."""
    if not text or len(text) <= max_chars:
        return text
    words = text.split()
    lines, current = [], []
    length = 0
    for word in words:
        if length + len(word) + 1 <= max_chars:
            current.append(word)
            length += len(word) + 1
        else:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _style_table(table, n_rows: int):
    """Aplica estilo consistente a tabelas do relatório."""
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    for j in range(3):
        cell = table[(0, j)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white", fontsize=10)

    for i in range(1, n_rows + 1):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor("#F2F2F2" if i % 2 == 0 else "white")
            if j > 0:
                cell.set_text_props(ha="right")
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.5)
