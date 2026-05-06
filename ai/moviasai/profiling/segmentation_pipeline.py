"""
Pipeline de segmentação de veículos em duas etapas.

Etapas
------
1. Feature engineering (tipo + segmentação KM/H)
2. Detecção de anomalias (outliers + séries não-modeláveis)
3. Clustering de métrica predominante (KM vs H)
4. Classificação de métrica predominante
5. Clustering de segmentação por métrica
6. Classificação de segmentos
7. Relatório final + exportação
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from moviasai.profiling.classification import SegmentationClassifier, TypeClassifier
from moviasai.profiling.clustering import SegmentationClusterer, TypeClusterer
from moviasai.data.data_quality import SeriesQuality
from moviasai.versioning import PipelineManifest

from moviasai.profiling.profile import VehicleProfile


class VehicleSegmentationPipeline:
    """Pipeline completo de segmentação de veículos em duas etapas."""

    REQUIRED_COLUMNS = {"veiculo_id", "data", "h_dia_clean", "km_dia_clean"}

    def __init__(
        self,
        df_daily: pl.DataFrame,
        type_features: List[str],
        km_features: List[str],
        h_features: List[str],
        filter_thresholds: Dict[str, float],
        quantiles: Optional[Dict[str, float]] = None,
        # Parâmetros do pipeline
        k_stage1: Optional[int] = 2,
        k_range_stage1: Tuple[int, int] = (2, 10),
        k_stage2: Optional[int] = None,
        k_range_stage2: Tuple[int, int] = (2, 12),
        train_classifiers: bool = True,
        test_size: float = 0.3,
        percentile_clean: float = 0.10,
        low_pct: float = 0.1,
        high_pct: float = 0.9,
        generate_pdf: bool = True,
        # Diretórios
        output_base_dir: str = "./output",
        dataset_dir: Optional[str] = None,
    ):
        # Validar colunas obrigatórias
        missing = self.REQUIRED_COLUMNS - set(df_daily.columns)
        if missing:
            raise ValueError(
                f"DataFrame sem colunas obrigatórias: {sorted(missing)}"
            )

        self.df_daily = df_daily
        self.filter_thresholds = filter_thresholds
        self.output_base_dir = Path(output_base_dir)
        self.classification_dir = self.output_base_dir / "models"
        self.dataset_dir = Path(dataset_dir) if dataset_dir else self.output_base_dir / "train_data"
        self.console_output = ""

        # Parâmetros do pipeline
        self.k_stage1 = k_stage1
        self.k_range_stage1 = k_range_stage1
        self.k_stage2 = k_stage2
        self.k_range_stage2 = k_range_stage2
        self.train_classifiers = train_classifiers
        self.test_size = test_size
        self.percentile_clean = percentile_clean
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.generate_pdf = generate_pdf
        self.quantiles = quantiles

        if self.df_daily["data"].dtype not in [pl.Date, pl.Datetime]:
            self.df_daily = self.df_daily.with_columns(
                pl.col("data").str.strptime(pl.Date, "%Y-%m-%d")
            )

        # DataFrames de estado
        self.df_features: Optional[pd.DataFrame] = None
        self.df_no_anomalies: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_final: Optional[pd.DataFrame] = None

        # Profile (feature engineering + anomaly detection)
        self.profile = VehicleProfile(
            type_features=type_features,
            km_features={'seg': km_features} if km_features else None,
            h_features={'seg': h_features} if h_features else None,
            filter_thresholds=filter_thresholds,
            quantiles=quantiles,
        )
        self.type_extractor = self.profile.type_extractor
        self.km_extractor = self.profile.km_extractors.get('seg')
        self.h_extractor = self.profile.h_extractors.get('seg')

        # Clusterers
        self.metric_clusterer: Optional[TypeClusterer] = None
        self.km_clusterer: Optional[SegmentationClusterer] = None
        self.h_clusterer: Optional[SegmentationClusterer] = None

        # Classifiers
        self.metric_classifier: Optional[TypeClassifier] = None
        self.km_classifier: Optional[SegmentationClassifier] = None
        self.h_classifier: Optional[SegmentationClassifier] = None

        # Mapeamentos de classes
        self.class_idx_to_metric: Optional[Dict[int, str]] = None
        self.metric_to_class_idx: Optional[Dict[str, int]] = None

        print("=" * 80)
        print("PIPELINE INICIALIZADO")
        print("=" * 80)
        print(f"Registros: {len(self.df_daily):,}")
        print(f"Veículos: {self.df_daily['veiculo_id'].n_unique()}")
        print(f"Output: {self.output_base_dir}")
        print()
        print("Features selecionadas:")
        print(f"  • Tipo: {len(self.type_extractor.feature_names)}")
        print(f"  • Segmentação KM: {len(self.km_extractor.feature_names)}")
        print(f"  • Segmentação H: {len(self.h_extractor.feature_names)}")
        print()

    @classmethod
    def from_config(
        cls, df_daily: pl.DataFrame, config, output_config, data_quality_config,
    ) -> "VehicleSegmentationPipeline":
        """
        Cria pipeline a partir de um SegmentationConfig.

        Parameters
        ----------
        df_daily : pl.DataFrame
            DataFrame com dados diários de telemetria.
        config : api.config.segmentation_config.SegmentationConfig
            Configuração carregada do YAML.
        output_config : api.config.output_config.OutputConfig
            Configuração de diretórios de saída.
        data_quality_config : api.config.data_quality_config.DataQualityConfig
            Configuração de qualidade de dados.
        """
        s1 = config.stage1
        s2 = config.stage2

        fmt = data_quality_config.thresholds.model_dump()
        quantiles = data_quality_config.quantiles

        return cls(
            df_daily=df_daily,
            type_features=config.features.type,
            km_features=config.features.km.seg,
            h_features=config.features.h.seg,
            filter_thresholds=fmt,
            quantiles=quantiles,
            k_stage1=s1.clustering.k,
            k_range_stage1=s1.clustering.k_range,
            k_stage2=s2.clustering.k,
            k_range_stage2=s2.clustering.k_range,
            test_size=s1.classification.test_size,
            percentile_clean=s1.classification.percentile_clean,
            low_pct=s1.uncertainty.low_pct,
            high_pct=s1.uncertainty.high_pct,
            generate_pdf=config.report.enabled,
            output_base_dir=output_config.logs.segmentation,
            dataset_dir=output_config.data.train_dataset,
        )

    # ------------------------------------------------------------------
    # 1. Feature engineering
    # ------------------------------------------------------------------

    def feature_engineering(self) -> pd.DataFrame:
        """Gera e combina features de todos os extractors."""
        self.profile.feature_engineering(self.df_daily)
        self.df_features = self.profile.df_features
        return self.df_features

    # ------------------------------------------------------------------
    # 2. Detecção de anomalias
    # ------------------------------------------------------------------

    def detect_anomalies(self) -> pd.DataFrame:
        """Detecta outliers físicos e séries não-modeláveis."""
        self.profile.detect_anomalies(
            df_daily=self.df_daily,
            output_dir=self.output_base_dir,
        )
        self.df_features = self.profile.df_features
        self.df_no_anomalies = self.profile.df_no_anomalies
        return self.df_features

    # ------------------------------------------------------------------
    # 3. Etapa 1 — Clustering de métrica predominante
    # ------------------------------------------------------------------

    def run_stage1_clustering(
        self, k: Optional[int] = 2, k_range: Tuple[int, int] = (2, 10)
    ):
        """Clustering para separar KM vs H."""
        print("\n" + "=" * 80)
        print("ETAPA 1: CLUSTERING DE MÉTRICA PREDOMINANTE")
        print("=" * 80 + "\n")

        df = self.df_no_anomalies.copy()
        km_feature = "razao_km_h" if "razao_km_h" in df.columns else "proporcao_km"

        print(f"Clustering para {len(df)} veículos...")

        self.metric_clusterer = TypeClusterer(
            str(self.output_base_dir / "clustering" / "stage1")
        )
        X_type = df[self.type_extractor.feature_names].values
        labels = self.metric_clusterer.fit(
            X_type, self.type_extractor.feature_names, k, k_range
        )

        df["cluster_metrica"] = labels.astype(int)
        self.metric_clusterer.plot_projections()

        # Mapear clusters para nomes
        cluster_stats = df.groupby("cluster_metrica")[[km_feature]].mean()
        if cluster_stats.loc[0, km_feature] > cluster_stats.loc[1, km_feature]:
            self.class_idx_to_metric = {0: "KM", 1: "H"}
            self.metric_to_class_idx = {"KM": 0, "H": 1}
        else:
            self.class_idx_to_metric = {1: "KM", 0: "H"}
            self.metric_to_class_idx = {"KM": 1, "H": 0}

        df["metrica_predominante"] = df["cluster_metrica"].map(self.class_idx_to_metric)

        clustering_dir = self.output_base_dir / "clustering" / "stage1"
        clustering_dir.mkdir(parents=True, exist_ok=True)
        df[["veiculo_id", "cluster_metrica", "metrica_predominante"]].to_csv(
            clustering_dir / "clusters_metrica.csv", index=False
        )
        print(f"✓ Clusters salvos: {clustering_dir / 'clusters_metrica.csv'}\n")

        self.df_no_anomalies = df
        return df

    # ------------------------------------------------------------------
    # 4. Etapa 1 — Classificação de métrica
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_filter_thresholds(df: pd.DataFrame, percentile_clean: float = 0.10) -> dict:
        """Sugere limiares de filtragem baseados em percentis."""
        km_data = df[df["metrica_predominante"] == "KM"]
        h_data = df[df["metrica_predominante"] == "H"]
        thresholds = {}

        if "proporcao_km" in df.columns:
            thresholds["km_prop_min"] = km_data["proporcao_km"].quantile(percentile_clean)
            thresholds["h_prop_max"] = h_data["proporcao_km"].quantile(1 - percentile_clean)
        if "razao_km_h" in df.columns:
            thresholds["km_ratio_min"] = km_data["razao_km_h"].quantile(percentile_clean)
            thresholds["h_ratio_max"] = h_data["razao_km_h"].quantile(1 - percentile_clean)
        if "corr_km_h" in df.columns:
            thresholds["corr_min"] = min(
                km_data["corr_km_h"].quantile(percentile_clean),
                h_data["corr_km_h"].quantile(percentile_clean),
            )

        print("Limiares sugeridos:")
        for k, v in thresholds.items():
            print(f"  {k}: {v:.3f}")

        return thresholds

    @staticmethod
    def filter_noisy_type_samples(
        df: pd.DataFrame,
        km_prop_min: float,
        h_prop_max: float,
        km_ratio_min: float,
        h_ratio_max: float,
        corr_min: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove amostras ruidosas após clustering de tipo."""
        required = ["cluster_metrica", "metrica_predominante"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória ausente: {col}")

        has_prop = "proporcao_km" in df.columns
        has_razao = "razao_km_h" in df.columns
        has_corr = "corr_km_h" in df.columns

        if not any([has_prop, has_razao, has_corr]):
            raise ValueError(
                "Nenhuma feature disponível para filtragem. "
                "É necessário ao menos uma de: proporcao_km, razao_km_h, corr_km_h"
            )

        print("\nFiltragem de amostras ruidosas:")
        print("=" * 60)
        print("Features disponíveis:")
        print(f"  • proporcao_km: {'✓' if has_prop else '✗'}")
        print(f"  • razao_km_h:   {'✓' if has_razao else '✗'}")
        print(f"  • corr_km_h:    {'✓' if has_corr else '✗'}")
        print()

        df = df.copy()
        df["is_noisy"] = False
        df["noise_reason"] = None

        # Filtrar KM ruidoso
        km_conds, km_reasons = [], []
        if has_prop:
            km_conds.append(df["proporcao_km"] < km_prop_min)
            km_reasons.append(f"prop_km < {km_prop_min}")
        if has_razao:
            km_conds.append(df["razao_km_h"] < km_ratio_min)
            km_reasons.append(f"razao < {km_ratio_min}")
        if has_corr:
            km_conds.append(df["corr_km_h"] < corr_min)
            km_reasons.append(f"corr < {corr_min}")

        mask_km_noise = pd.Series(False, index=df.index)
        if km_conds:
            mask_km_noise = (df["metrica_predominante"] == "KM") & np.logical_or.reduce(
                km_conds
            )
            df.loc[mask_km_noise, "is_noisy"] = True
            df.loc[mask_km_noise, "noise_reason"] = "KM inconsistente"

        # Filtrar H ruidoso
        h_conds, h_reasons = [], []
        if has_prop:
            h_conds.append(df["proporcao_km"] > h_prop_max)
            h_reasons.append(f"prop_km > {h_prop_max}")
        if has_razao:
            h_conds.append(df["razao_km_h"] > h_ratio_max)
            h_reasons.append(f"razao > {h_ratio_max}")
        if has_corr:
            h_conds.append(df["corr_km_h"] < corr_min)
            h_reasons.append(f"corr < {corr_min}")

        mask_h_noise = pd.Series(False, index=df.index)
        if h_conds:
            mask_h_noise = (df["metrica_predominante"] == "H") & np.logical_or.reduce(
                h_conds
            )
            df.loc[mask_h_noise, "is_noisy"] = True
            df.loc[mask_h_noise, "noise_reason"] = "H inconsistente"

        df_clean = df[~df["is_noisy"]].copy()
        df_removed = df[df["is_noisy"]].copy()

        # Estatísticas
        print("Critérios aplicados:")
        if km_conds:
            print(f"  KM: {' OR '.join(km_reasons)}")
        if h_conds:
            print(f"  H:  {' OR '.join(h_reasons)}")
        print()

        print("Resultados:")
        print(f"  Total inicial:         {len(df):>6,}")
        print(f"  KM inconsistente:      {mask_km_noise.sum():>6,}")
        print(f"  H inconsistente:       {mask_h_noise.sum():>6,}")
        print(
            f"  Total removido:        {len(df_removed):>6,} "
            f"({len(df_removed)/len(df)*100:5.1f}%)"
        )
        print(
            f"  Total mantido (clean): {len(df_clean):>6,} "
            f"({len(df_clean)/len(df)*100:5.1f}%)"
        )
        print()

        if len(df_removed) > 0:
            print("Removidos por tipo:")
            for tipo, count in df_removed["metrica_predominante"].value_counts().items():
                pct = count / len(df_removed) * 100
                print(f"  {tipo:12s}: {count:>6,} ({pct:5.1f}% dos removidos)")
            print()

        print("=" * 60)
        print()

        return df_clean, df_removed

    def run_stage1_classification(self, df: pd.DataFrame, test_size: float = 0.3):
        """Treina classificador de métrica predominante."""
        if len(df) < 10:
            print("⚠️  Dados insuficientes para classificação Etapa 1")
            return

        self.metric_classifier = TypeClassifier(
            str(self.classification_dir / "stage1")
        )

        X = df[self.type_extractor.feature_names].values
        y = df["cluster_metrica"].values
        self.metric_classifier.train(X, y, self.type_extractor.feature_names, test_size)

        predictions, probabilities = self.metric_classifier.predict(X)
        p_km = probabilities[:, self.metric_to_class_idx["KM"]]

        self.df_clean = df.copy()
        self.df_clean["predicted_class"] = predictions
        self.df_clean["p_km"] = p_km

    @staticmethod
    def compute_uncertainty_thresholds_by_percentile(
        p_km: np.ndarray, low_pct: float = 0.10, high_pct: float = 0.90
    ) -> dict:
        """Define thresholds de incerteza com base em percentis do score."""
        p_km = np.asarray(p_km)
        if not (0 < low_pct < high_pct < 1):
            raise ValueError("Percentis inválidos")
        p_min = np.quantile(p_km, low_pct)
        p_max = np.quantile(p_km, high_pct)
        return {
            "p_min": p_min,
            "p_max": p_max,
            "low_pct": low_pct,
            "high_pct": high_pct,
            "coverage_H": np.mean(p_km <= p_min),
            "coverage_KM": np.mean(p_km >= p_max),
            "coverage_uncertain": np.mean((p_km > p_min) & (p_km < p_max)),
            "mean_p_km": float(np.mean(p_km)),
            "std_p_km": float(np.std(p_km)),
        }

    # ------------------------------------------------------------------
    # 5. Etapa 2 — Clustering de segmentação
    # ------------------------------------------------------------------

    def run_stage2_clustering(
        self, k: Optional[int] = None, k_range: Tuple[int, int] = (2, 12)
    ):
        """Clustering de segmentação por métrica (KM e H separados)."""
        print("\n" + "=" * 80)
        print("ETAPA 2: CLUSTERING DE SEGMENTAÇÃO")
        print("=" * 80 + "\n")

        df = self.df_clean.copy()
        df["cluster_segmento"] = -1
        df["segmento"] = None

        for metric, extractor in [
            ("KM", self.km_extractor),
            ("H", self.h_extractor),
        ]:
            mask = df["metrica_predominante"] == metric
            if mask.sum() == 0:
                continue

            print(f"{metric}: {mask.sum()} veículos")
            clusterer = SegmentationClusterer(
                metric.lower(),
                str(self.output_base_dir / "clustering" / "stage2"),
            )

            X = df.loc[mask, extractor.feature_names].values
            labels = clusterer.fit(X, extractor.feature_names, k, k_range)

            df.loc[mask, "cluster_segmento"] = labels.astype(int)
            df.loc[mask, "segmento"] = [f"{metric}_C{l}" for l in labels]
            clusterer.plot_projections()

            if metric == "KM":
                self.km_clusterer = clusterer
            else:
                self.h_clusterer = clusterer

        clustering_dir = self.output_base_dir / "clustering" / "stage2"
        clustering_dir.mkdir(parents=True, exist_ok=True)
        df[["veiculo_id", "metrica_predominante", "cluster_segmento", "segmento"]].to_csv(
            clustering_dir / "clusters_segmento.csv", index=False
        )
        print(f"✓ Clusters salvos: {clustering_dir / 'clusters_segmento.csv'}\n")

        self.df_final = df
        return df

    # ------------------------------------------------------------------
    # 6. Etapa 2 — Classificação de segmentos
    # ------------------------------------------------------------------

    def run_stage2_classification(self, test_size: float = 0.3):
        """Treina classificadores de segmento para KM e H."""
        df = self.df_final.copy()

        for metric, extractor, attr_name in [
            ("KM", self.km_extractor, "km_classifier"),
            ("H", self.h_extractor, "h_classifier"),
        ]:
            subset = df[
                (df["metrica_predominante"] == metric) & (df["cluster_segmento"] >= 0)
            ].copy()

            if len(subset) <= 10 or subset["cluster_segmento"].nunique() <= 1:
                print(f"⚠️  {metric}: dados insuficientes")
                print()
                continue

            print("\n" + "=" * 80)
            print(f"CLASSIFICAÇÃO: {metric}")
            print("=" * 80)
            print(f"Veículos: {len(subset)}")
            print(f"Clusters: {subset['cluster_segmento'].nunique()}")
            print()

            min_samples = 4
            valid_clusters = subset["cluster_segmento"].value_counts()
            valid_clusters = valid_clusters[valid_clusters >= min_samples]

            if len(valid_clusters) < 2:
                print("⚠️  Clusters insuficientes")
                print()
                continue

            subset = subset[subset["cluster_segmento"].isin(valid_clusters.index)].copy()

            classifier = SegmentationClassifier(
                metric.lower(),
                str(self.classification_dir / "stage2"),
            )
            X = subset[extractor.feature_names].values
            y = subset["cluster_segmento"].astype(int).values
            classifier.train(X, y, extractor.feature_names, test_size)

            setattr(self, attr_name, classifier)

    # ------------------------------------------------------------------
    # 7. Relatório final + exportação
    # ------------------------------------------------------------------

    def generate_final_report(self) -> pd.DataFrame:
        """Gera relatório final com estatísticas dos segmentos."""
        print("\n" + "=" * 80)
        print("RELATÓRIO FINAL")
        print("=" * 80 + "\n")

        report_dir = self.output_base_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        df = self.df_final
        summary = {
            "total_veiculos": int(len(df)),
            "outliers": int((self.df_features["quality"] == SeriesQuality.OUTLIER).sum()),
            "veiculos_validos": int(len(df)),
            "metricas": {
                str(k): int(v)
                for k, v in df["metrica_predominante"].value_counts().to_dict().items()
            },
            "segmentos": {
                str(k): int(v)
                for k, v in df["segmento"].value_counts().to_dict().items()
            },
        }
        with open(report_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary: {report_dir / 'summary.json'}")

        stats_list = []

        print("\n" + "=" * 80)
        print("ESTATÍSTICAS DOS SEGMENTOS")
        print("=" * 80)

        for seg in sorted(df["segmento"].unique()):
            seg_data = df[df["segmento"] == seg]
            metrica = seg_data["metrica_predominante"].iloc[0]
            n = len(seg_data)
            pct = n / len(df) * 100

            print(f"\n{seg} (n={n}, {pct:.1f}%)")
            print("=" * 70)

            stats = {"segmento": seg, "metrica_predominante": metrica, "n": n, "pct": pct}

            features = (
                self.km_extractor.feature_names
                if metrica == "KM"
                else self.h_extractor.feature_names
            )

            for feat in features:
                if feat not in seg_data.columns:
                    continue
                mean_val = seg_data[feat].mean()
                std_val = seg_data[feat].std()
                stats[f"{feat}_mean"] = mean_val
                stats[f"{feat}_std"] = std_val

                if "taxa" in feat:
                    print(f"{feat:30s}: {mean_val:8.2%} ± {std_val:6.2%}")
                elif "gap" in feat or feat == "tamanho_periodo":
                    print(f"{feat:30s}: {mean_val:8.1f} ± {std_val:6.1f} dias")
                else:
                    print(f"{feat:30s}: {mean_val:8.2f} ± {std_val:6.2f}")

            stats_list.append(stats)

        df_stats = pd.DataFrame(stats_list)
        df_stats.to_csv(report_dir / "segment_statistics.csv", index=False)
        print(f"\n✓ Estatísticas salvas: {report_dir / 'segment_statistics.csv'}")
        print()

        return df_stats

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------

    def run(self):
        """Executa pipeline completo com captura de console."""
        console_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = _TeeOutput(original_stdout, console_capture)

        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETO")
            print("=" * 80)
            print(f"Timestamp: {timestamp}")
            print(f"Output: {self.output_base_dir}")
            print("=" * 80 + "\n")

            # 1. Features
            self.feature_engineering()

            # 2. Anomalias
            self.detect_anomalies()

            # 3. Etapa 1: Clustering
            self.run_stage1_clustering(self.k_stage1, self.k_range_stage1)

            # 4. Etapa 1: Classificação
            if self.train_classifiers:
                df = self.df_no_anomalies[
                    self.df_no_anomalies["metrica_predominante"].isin(["KM", "H"])
                ].copy()
                thresholds_clean = self.suggest_filter_thresholds(
                    df, percentile_clean=self.percentile_clean
                )
                df_clean, _ = self.filter_noisy_type_samples(df, **thresholds_clean)

                self.run_stage1_classification(df_clean, self.test_size)

                _, probs = self.metric_classifier.predict(
                    df[self.type_extractor.feature_names].values
                )
                p_km_full = probs[:, self.metric_to_class_idx["KM"]]
                stats = self.compute_uncertainty_thresholds_by_percentile(
                    p_km=p_km_full, low_pct=self.low_pct, high_pct=self.high_pct
                )
                print(stats)

            # 5. Etapa 2: Clustering
            self.run_stage2_clustering(self.k_stage2, self.k_range_stage2)

            # 6. Etapa 2: Classificação
            if self.train_classifiers:
                self.run_stage2_classification(self.test_size)

            # 7. Relatório final
            self.generate_final_report()

            # 8. Exportação
            self._export_final_datasets()

            # 9. Manifest de versionamento
            self._write_manifest()

            print("\n" + "=" * 80)
            print("PIPELINE CONCLUÍDO")
            print("=" * 80 + "\n")

        finally:
            sys.stdout = original_stdout
            self.console_output = console_capture.getvalue()

        # 9. Gerar PDF
        if self.generate_pdf:
            from moviasai.report import SegmentationReportGenerator

            report_gen = SegmentationReportGenerator(self.output_base_dir / "reports")
            report_gen.generate(self, self.console_output)

        return {
            "df_features": self.df_features,
            "df_final": self.df_final,
            "metric_clusterer": self.metric_clusterer,
            "km_clusterer": self.km_clusterer,
            "h_clusterer": self.h_clusterer,
            "metric_classifier": self.metric_classifier,
            "km_classifier": self.km_classifier,
            "h_classifier": self.h_classifier,
            "console_output": self.console_output,
        }

    def _export_final_datasets(self):
        """Salva CSV final e datasets por métrica."""
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.output_base_dir / "veiculos_segmentados_final.csv"
        self.df_final.to_csv(final_path, index=False)
        print(f"✓ Dataset de features final: {final_path}")

        final_data_dir = self.dataset_dir
        final_data_dir.mkdir(parents=True, exist_ok=True)

        for metrica in ["KM", "H"]:
            vids = self.df_final.loc[
                self.df_final["metrica_predominante"] == metrica, "veiculo_id"
            ].to_list()
            if not vids:
                continue
            output_path = final_data_dir / f"{metrica.lower()}.csv"
            df_data_final = self.df_daily.filter(pl.col("veiculo_id").is_in(vids))
            df_data_final.write_csv(output_path)
            print(f"✓ Dataset {metrica}: {output_path}")
            print(f"  Veículos: {df_data_final['veiculo_id'].n_unique():,}")
            print(f"  Registros: {len(df_data_final):,}")

    def _write_manifest(self):
        """Escreve manifest de versionamento com hash dos artefactos."""
        key_files = [self.output_base_dir / "veiculos_segmentados_final.csv"]
        for m in ("km", "h"):
            p = self.dataset_dir / f"{m}.csv"
            if p.exists():
                key_files.append(p)
        for stage in ("stage1", "stage2"):
            model_dir = self.classification_dir / stage
            if model_dir.exists():
                key_files.extend(sorted(model_dir.glob("*.pkl")))
                key_files.extend(sorted(model_dir.glob("*.onnx")))

        manifest = PipelineManifest("segmentation", self.output_base_dir)
        manifest.write(
            output_hash=PipelineManifest.hash_files(*key_files),
            config_hash=PipelineManifest.hash_config(self.filter_thresholds),
        )
        print(f"✓ Manifest: {manifest.path}")


# ======================================================================
# Helpers
# ======================================================================


class _TeeOutput:
    """Duplica stdout para capturar e exibir simultaneamente."""

    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, text):
        for output in self.outputs:
            output.write(text)

    def flush(self):
        for output in self.outputs:
            output.flush()



