import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                            davies_bouldin_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class BaseClusterer:
    """Classe base para clusterização com PCA e t-SNE"""
    
    def __init__(self, name: str, output_dir: str):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.pca = None
        self.tsne = None
        self.labels = None
        self.pca_coords = None
        self.tsne_coords = None
        self.metrics = {}
        
    def find_optimal_k(self, X_scaled: np.ndarray, k_range: Tuple[int, int] = (2, 12)) -> Tuple[int, pd.DataFrame]:
        """Busca k ótimo"""
        print(f"\n{'='*80}")
        print(f"BUSCA DE K ÓTIMO - {self.name}")
        print(f"{'='*80}\n")
        
        k_min, k_max = k_range
        k_values = list(range(k_min, k_max + 1))
        
        results = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            labels = kmeans.fit_predict(X_scaled)
            
            sil = silhouette_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            
            results.append({
                'k': k,
                'silhouette': sil,
                'calinski_harabasz': ch,
                'davies_bouldin': db,
                'inertia': kmeans.inertia_
            })
            
            print(f"  k={k:2d}: sil={sil:.4f}, CH={ch:8.2f}, DB={db:.4f}")
        
        df_results = pd.DataFrame(results)
        k_optimal = int(df_results.loc[df_results['silhouette'].idxmax(), 'k'])
        
        print(f"\n✓ K ótimo: {k_optimal}\n")
        
        df_results.to_csv(self.output_dir / f'{self.name}_k_optimization.csv', index=False)
        
        return k_optimal, df_results
    
    def fit(self, X: np.ndarray, feature_names: List[str], k: Optional[int] = None, 
            k_range: Tuple[int, int] = (2, 12)) -> np.ndarray:
        """Treina modelo com PCA e t-SNE"""
        print(f"\n{'='*80}")
        print(f"CLUSTERIZAÇÃO - {self.name}")
        print(f"{'='*80}")
        print(f"Amostras: {len(X)}")
        print(f"Features: {len(feature_names)}")
        print()
        
        # Normalizar
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Determinar k
        if k is None:
            k, df_k_results = self.find_optimal_k(X_scaled, k_range)
            self.plot_k_analysis(df_k_results, k)
        else:
            print(f"Usando k={k} (fixo)\n")
        
        # Treinar
        self.model = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        self.labels = self.model.fit_predict(X_scaled)
        
        # Métricas
        self.metrics = {
            'n_clusters': k,
            'n_samples': len(X),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'silhouette': float(silhouette_score(X_scaled, self.labels)),
            'calinski_harabasz': float(calinski_harabasz_score(X_scaled, self.labels)),
            'davies_bouldin': float(davies_bouldin_score(X_scaled, self.labels)),
            'inertia': float(self.model.inertia_)
        }
        
        print(f"Métricas:")
        print(f"  Silhouette: {self.metrics['silhouette']:.4f}")
        print(f"  Calinski-Harabasz: {self.metrics['calinski_harabasz']:.2f}")
        print(f"  Davies-Bouldin: {self.metrics['davies_bouldin']:.4f}")
        print()
        
        # Distribuição
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Distribuição:")
        for cluster_id, count in zip(unique, counts):
            pct = count / len(self.labels) * 100
            print(f"  Cluster {cluster_id}: {count:5d} ({pct:5.1f}%)")
        print()
        
        # PCA
        print("Calculando PCA...")
        self.pca = PCA(n_components=2, random_state=42)
        self.pca_coords = self.pca.fit_transform(X_scaled)
        var_pca = self.pca.explained_variance_ratio_
        print(f"  Variância: {var_pca.sum():.2%}")
        
        # t-SNE
        print("Calculando t-SNE...")
        perplexity = min(30, len(X) - 1)
        self.tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                        max_iter=1000, verbose=0)
        self.tsne_coords = self.tsne.fit_transform(X_scaled)
        print("  ✓ Concluído")
        print()
        
        # Salvar
        self.save_artifacts(X, feature_names)
        
        return self.labels
    
    def save_artifacts(self, X: np.ndarray, feature_names: List[str]):
        """Salva artefatos"""
        # Modelo
        model_path = self.output_dir / f'{self.name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'pca': self.pca,
                'tsne': self.tsne,
                'pca_coords': self.pca_coords,
                'tsne_coords': self.tsne_coords,
                'labels': self.labels,
                'feature_names': feature_names
            }, f)
        print(f"✓ Modelo: {model_path}")
        
        # Métricas
        metrics_path = self.output_dir / f'{self.name}_metrics.json'
        metrics_to_save = {k: v for k, v in self.metrics.items() if not isinstance(v, list)}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"✓ Métricas: {metrics_path}")
        
        # Estatísticas
        stats = self.compute_cluster_statistics(X, feature_names)
        stats_path = self.output_dir / f'{self.name}_cluster_stats.csv'
        stats.to_csv(stats_path, index=False)
        print(f"✓ Estatísticas: {stats_path}")
        print()
        
    def compute_cluster_statistics(self, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Calcula estatísticas"""
        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = self.labels
        
        stats = []
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            
            stat = {'cluster': int(cluster_id), 'n': len(cluster_data)}
            
            for feat in feature_names:
                stat[f'{feat}_mean'] = cluster_data[feat].mean()
                stat[f'{feat}_std'] = cluster_data[feat].std()
                stat[f'{feat}_median'] = cluster_data[feat].median()
            
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def plot_k_analysis(self, df_results: pd.DataFrame, k_optimal: int):
        """Plota análise de k"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        ax.plot(df_results['k'], df_results['inertia'], 'bo-', linewidth=2, markersize=8)
        ax.axvline(x=k_optimal, color='red', linestyle='--', linewidth=2, label=f'k={k_optimal}')
        ax.set_xlabel('k')
        ax.set_ylabel('Inércia')
        ax.set_title('Método do Cotovelo')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(df_results['k'], df_results['silhouette'], 'ro-', linewidth=2, markersize=8)
        ax.axvline(x=k_optimal, color='red', linestyle='--', linewidth=2, label=f'k={k_optimal}')
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Bom (>0.5)')
        ax.set_xlabel('k')
        ax.set_ylabel('Silhouette')
        ax.set_title('Silhouette Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(df_results['k'], df_results['calinski_harabasz'], 'go-', linewidth=2, markersize=8)
        ax.axvline(x=k_optimal, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('k')
        ax.set_ylabel('CH')
        ax.set_title('Calinski-Harabasz')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(df_results['k'], df_results['davies_bouldin'], 'mo-', linewidth=2, markersize=8)
        ax.axvline(x=k_optimal, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('k')
        ax.set_ylabel('DB')
        ax.set_title('Davies-Bouldin')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.name} - Análise de K Ótimo', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.name}_k_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Gráfico: {self.name}_k_analysis.png\n")
    
    def plot_projections(self, figsize=(16, 8)):
        """Plota PCA e t-SNE lado a lado"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        unique_labels = np.unique(self.labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # PCA
        ax = axes[0]
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            n = mask.sum()
            ax.scatter(
                self.pca_coords[mask, 0],
                self.pca_coords[mask, 1],
                c=[colors[i]],
                label=f'C{label} (n={n})',
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
            
            centroid = self.pca_coords[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c='black', marker='X', s=250,
                      edgecolors='white', linewidth=2.5, zorder=10)
            ax.text(centroid[0], centroid[1], f'{label}', fontsize=11,
                   fontweight='bold', ha='center', va='center', color='white')
        
        var = self.pca.explained_variance_ratio_
        ax.set_xlabel(f'Componente Principal 1 ({var[0]:.1%})', fontsize=11)
        ax.set_ylabel(f'Componente Principal 2 ({var[1]:.1%})', fontsize=11)
        ax.set_title(f'PCA - Variância Explicada: {var.sum():.1%}', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # t-SNE
        ax = axes[1]
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            n = mask.sum()
            ax.scatter(
                self.tsne_coords[mask, 0],
                self.tsne_coords[mask, 1],
                c=[colors[i]],
                label=f'C{label} (n={n})',
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
            
            centroid = self.tsne_coords[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c='black', marker='X', s=250,
                      edgecolors='white', linewidth=2.5, zorder=10)
            ax.text(centroid[0], centroid[1], f'{label}', fontsize=11,
                   fontweight='bold', ha='center', va='center', color='white')
        
        ax.set_xlabel('t-SNE Dimensão 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimensão 2', fontsize=11)
        ax.set_title('t-SNE - Projeção Não-Linear', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.name} - Projeções 2D', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.name}_projections.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Projeções (PCA + t-SNE): {self.name}_projections.png\n")


class TypeClusterer(BaseClusterer):
    def __init__(self, output_dir: str = './output/clustering/etapa1'):
        super().__init__('Etapa1_Tipo', output_dir)


class SegmentationClusterer(BaseClusterer):
    def __init__(self, tipo: str, output_dir: str = './output/clustering/etapa2'):
        super().__init__(f'Etapa2_{tipo}', output_dir)
        self.tipo = tipo
