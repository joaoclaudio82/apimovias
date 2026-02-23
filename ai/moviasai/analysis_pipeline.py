import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

import sys
import io
from datetime import datetime
from sklearn.metrics import confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional, Tuple

from feature_extraction import SegmentationFeatureExtractor, TypeFeatureExtractor
from classification import SegmentationClassifier, TypeClassifier
from clustering import SegmentationClusterer, TypeClusterer


import warnings
warnings.filterwarnings('ignore')


class ConsoleCapture:
    """Captura saída do console"""
    
    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stdout = sys.stdout
        
    def __enter__(self):
        sys.stdout = self.buffer
        return self.buffer
    
    def __exit__(self, *args):
        sys.stdout = self.original_stdout
    
    def get_output(self):
        return self.buffer.getvalue()



class PDFReportGenerator:
    """Gerador de relatório PDF completo"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def generate(self, pipeline, console_output: str):
        """
        Gera relatório PDF completo
        
        Parameters:
        -----------
        pipeline : VehicleAnalysisPipeline
            Pipeline executado
        console_output : str
            Saída capturada do console
        """
        pdf_path = self.output_dir / 'relatorio_completo.pdf'
        
        print(f"\n{'='*80}")
        print("GERANDO RELATÓRIO PDF")
        print(f"{'='*80}\n")
        
        with PdfPages(pdf_path) as pdf:
            # Página 1: Capa
            self._create_cover_page(pdf)
            
            # Página 2-N: Console output (quebrado em páginas)
            self._create_console_pages(pdf, console_output)
            
            # Análise de Outliers
            if (pipeline.df_features['is_outlier']).sum() > 0:
                self._create_outliers_page(pdf, pipeline)
            
            # Clustering Etapa 1
            if pipeline.type_clusterer:
                self._create_clustering_page(pdf, pipeline.type_clusterer, 'Etapa 1: Identificação de Tipo')
            
            # Classificação Etapa 1
            if pipeline.type_classifier and pipeline.type_classifier.results:
                self._create_classification_page(pdf, pipeline.type_classifier)
            
            # Clustering Etapa 2 - Deslocamento
            if pipeline.displacement_clusterer:
                self._create_clustering_page(pdf, pipeline.displacement_clusterer, 'Etapa 2: Segmentação - Deslocamento')
            
            # Classificação Etapa 2 - Deslocamento
            if pipeline.displacement_classifier and pipeline.displacement_classifier.results:
                self._create_classification_page(pdf, pipeline.displacement_classifier)
            
            # Clustering Etapa 2 - Máquina
            if pipeline.machine_clusterer:
                self._create_clustering_page(pdf, pipeline.machine_clusterer, 'Etapa 2: Segmentação - Máquina')
            
            # Classificação Etapa 2 - Máquina
            if pipeline.machine_classifier and pipeline.machine_classifier.results:
                self._create_classification_page(pdf, pipeline.machine_classifier)
            
            # KDE
            self._create_kde_page(pdf, pipeline)
            
            # Estatísticas dos Segmentos
            self._create_segments_stats_pages(pdf, pipeline)
        
        print(f"✓ PDF gerado: {pdf_path}")
        print(f"  Tamanho: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
        print()
        
        return pdf_path
    
    def _create_cover_page(self, pdf):
        """Cria página de capa"""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title_text = [
            "",
            "",
            "",
            "RELATÓRIO DE ANÁLISE",
            "DE VEÍCULOS",
            "",
            "Clusterização e Classificação",
            "em Duas Etapas",
            "",
            "",
            f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            "",
            "",
            "="*50,
            "",
            "Gerado automaticamente pelo",
            "Vehicle Analysis Pipeline",
        ]
        
        ax.text(0.5, 0.5, '\n'.join(title_text), transform=ax.transAxes,
               fontsize=16, ha='center', va='center', fontfamily='sans-serif',
               fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_console_pages(self, pdf, console_output: str):
        """Cria páginas com saída do console"""
        lines = console_output.split('\n')
        lines_per_page = 60
        
        for page_num, i in enumerate(range(0, len(lines), lines_per_page), start=1):
            batch = lines[i:i+lines_per_page]
            
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            text = '\n'.join(batch)
            
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top', fontfamily='monospace',
                   wrap=True)
            
            # Número da página
            ax.text(0.95, 0.02, f'Página {page_num}', transform=ax.transAxes,
                   fontsize=8, ha='right', va='bottom', style='italic')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    def _create_outliers_page(self, pdf, pipeline):
        """Cria página de outliers"""
        df = pipeline.df_features
        outliers = df[df['is_outlier']]
        normal = df[~df['is_outlier']]
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Análise de Outliers', fontsize=14, fontweight='bold')
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Scatter
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(normal['km_por_dia'], normal['h_por_dia'], alpha=0.5, s=20, label='Normal', color='blue')
        if len(outliers) > 0:
            ax.scatter(outliers['km_por_dia'], outliers['h_por_dia'], alpha=0.8, s=40,
                    label='Outlier', color='red', marker='x')
        ax.set_xlabel('KM/dia')
        ax.set_ylabel('Horas/dia')
        ax.set_title('Detecção de Outliers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Barras
        ax = fig.add_subplot(gs[0, 1])
        if len(outliers) > 0:
            outliers['outlier_reason'].value_counts().plot(kind='bar', ax=ax, color='red', alpha=0.7)
            ax.set_ylabel('Quantidade')
            ax.set_title('Outliers por Tipo')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Histogramas
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(normal['km_por_dia'], bins=40, alpha=0.6, label='Normal', color='blue')
        if len(outliers) > 0:
            ax.hist(outliers['km_por_dia'], bins=40, alpha=0.6, label='Outlier', color='red')
        ax.set_xlabel('KM/dia')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição KM/dia')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(normal['h_por_dia'], bins=40, alpha=0.6, label='Normal', color='blue')
        if len(outliers) > 0:
            ax.hist(outliers['h_por_dia'], bins=40, alpha=0.6, label='Outlier', color='red')
        ax.set_xlabel('Horas/dia')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição Horas/dia')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_clustering_page(self, pdf, clusterer, title):
        """Cria página de clustering com PCA e t-SNE"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.4)
        
        unique_labels = np.unique(clusterer.labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # PCA
        ax = fig.add_subplot(gs[0, 0])
        for i, label in enumerate(unique_labels):
            mask = clusterer.labels == label
            n = mask.sum()
            ax.scatter(clusterer.pca_coords[mask, 0], clusterer.pca_coords[mask, 1],
                      c=[colors[i]], label=f'C{label} (n={n})', alpha=0.6, s=40)
            
            centroid = clusterer.pca_coords[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c='black', marker='X', s=150,
                      edgecolors='white', linewidth=2)
            ax.text(centroid[0], centroid[1], f'{label}', fontsize=9,
                   fontweight='bold', ha='center', va='center', color='white')
        
        var = clusterer.pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({var[1]:.1%})')
        ax.set_title(f'PCA (Var: {var.sum():.1%})')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        
        # t-SNE
        ax = fig.add_subplot(gs[0, 1])
        for i, label in enumerate(unique_labels):
            mask = clusterer.labels == label
            n = mask.sum()
            ax.scatter(clusterer.tsne_coords[mask, 0], clusterer.tsne_coords[mask, 1],
                      c=[colors[i]], label=f'C{label} (n={n})', alpha=0.6, s=40)
            
            centroid = clusterer.tsne_coords[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c='black', marker='X', s=150,
                      edgecolors='white', linewidth=2)
            ax.text(centroid[0], centroid[1], f'{label}', fontsize=9,
                   fontweight='bold', ha='center', va='center', color='white')
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Distribuição
        ax = fig.add_subplot(gs[1, 0])
        unique, counts = np.unique(clusterer.labels, return_counts=True)
        bars = ax.bar(unique, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Quantidade de Veículos')
        ax.set_title('Distribuição dos Clusters')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Métricas
        ax = fig.add_subplot(gs[1, 1])
        ax.axis('off')
        metrics_text = [
            "Métricas de Qualidade:",
            "",
            f"Silhouette:        {clusterer.metrics['silhouette']:8.4f}",
            f"Calinski-Harabasz: {clusterer.metrics['calinski_harabasz']:8.2f}",
            f"Davies-Bouldin:    {clusterer.metrics['davies_bouldin']:8.4f}",
            f"Inércia:           {clusterer.metrics['inertia']:8.2f}",
            "",
            "Informações:",
            f"Clusters:          {clusterer.metrics['n_clusters']:8d}",
            f"Amostras:          {clusterer.metrics['n_samples']:8d}",
            f"Features:          {clusterer.metrics['n_features']:8d}",
        ]
        ax.text(0.1, 0.9, '\n'.join(metrics_text), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    

    def _get_feature_importance(self, model, n_features: int) -> Tuple[Optional[np.ndarray], str]:
        """
        Extrai importância de features de forma genérica
        
        Parameters:
        -----------
        model : sklearn model
            Modelo treinado
        n_features : int
            Número de features (mantido para compatibilidade)
        
        Returns:
        --------
        Tuple[Optional[np.ndarray], str]
            (importances, importance_type) ou (None, '')
        """
        # Tree-based models (RandomForest, GradientBoosting, XGBoost, etc.)
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_, 'Importância (Gini/Entropy)'
        
        # Linear models (LogisticRegression, LinearSVC, Ridge, Lasso, etc.)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            
            if len(coef.shape) > 1:
                # Multiclasse: média absoluta dos coeficientes
                importances = np.abs(coef).mean(axis=0)
            else:
                # Binário: valor absoluto direto
                importances = np.abs(coef[0])
            
            return importances, 'Coeficiente Absoluto'
        
        # Modelo não suportado
        return None, ''

    def _create_classification_page(self, pdf, classifier):
        """Cria página de classificação"""
        best = classifier.results[classifier.best_model_name]
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(f'Classificação - {classifier.name}', fontsize=14, fontweight='bold')
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        # ====================================================================
        # 1. CONFUSION MATRIX
        # ====================================================================
        ax = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(best['y_test'], best['y_pred_test'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Contagem'})
        ax.set_xlabel('Predito', fontsize=10)
        ax.set_ylabel('Real', fontsize=10)
        ax.set_title(f'Matriz de Confusão - {classifier.best_model_name}', fontsize=11, fontweight='bold')
        # ====================================================================
        # 2. FEATURE IMPORTANCE (com fallback genérico)
        # ====================================================================
        ax = fig.add_subplot(gs[0, 1])
        
        # Obter importâncias usando método genérico
        importances, importance_type = self._get_feature_importance(
            best['model'], 
            len(classifier.feature_names)
        )
        
        if importances is not None:
            # Selecionar top 12 features
            n_top = min(12, len(importances))
            indices = np.argsort(importances)[::-1][:n_top]
            
            # Plotar barras horizontais
            ax.barh(range(len(indices)), importances[indices], 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Configurar eixos
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([classifier.feature_names[i] for i in indices], fontsize=8)
            ax.set_xlabel(importance_type, fontsize=10)
            ax.set_title(f'Top {n_top} Features', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Adicionar valores nas barras (opcional, mas recomendado)
            for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
                ax.text(imp, i, f' {imp:.3f}', va='center', fontsize=7, color='black')
        else:
            # Modelo não suporta importância de features
                ax.text(0.5, 0.5, 
                'Feature Importance\nnão disponível\npara este modelo', 
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
                ax.axis('off')
        # ====================================================================
        # 3. COMPARAÇÃO DE MODELOS
        # ====================================================================
        ax = fig.add_subplot(gs[1, 0])
        models = list(classifier.results.keys())
        f1_scores = [classifier.results[m]['f1_test'] for m in models]
        
        # Destacar melhor modelo
        colors = ['gold' if m == classifier.best_model_name else 'steelblue' for m in models]
        
        bars = ax.bar(range(len(models)), f1_scores, color=colors, 
                    alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
        ax.set_ylabel('F1 Score (Teste)', fontsize=10)
        ax.set_title('Comparação de Modelos', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        # Valores nas barras
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        # ====================================================================
        # 4. MÉTRICAS DETALHADAS
        # ====================================================================
        ax = fig.add_subplot(gs[1, 1])
        ax.axis('off')
        
        metrics_text = [
            f"Melhor Modelo:",
            f"{classifier.best_model_name}",
            "",
            "Métricas no Teste:",
            f"  Acurácia:  {best['acc_test']:.4f}",
            f"  F1 Score:  {best['f1_test']:.4f}",
            "",
            "Métricas no Treino:",
            f"  Acurácia:  {best['acc_train']:.4f}",
            f"  F1 Score:  {best['f1_train']:.4f}",
            "",
            "Cross-Validation (5-fold):",
            f"  F1 Médio:  {best['f1_cv_mean']:.4f}",
            f"  F1 Desvio: {best['f1_cv_std']:.4f}",
        ]
        
        ax.text(0.1, 0.9, '\n'.join(metrics_text), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        # Salvar no PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_kde_page(self, pdf, pipeline):
        """Cria página com KDE e histograma ao fundo"""
        df = pipeline.df_final
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Distribuições KDE', fontsize=14, fontweight='bold')
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # KDE por tipo - KM
        ax = fig.add_subplot(gs[0, 0])
        
        # Histograma ao fundo (todos os dados)
        ax.hist(df['km_por_dia'], bins=50, density=True, alpha=0.15, 
                color='gray', edgecolor='none', label='_nolegend_')
        
        # KDE por tipo
        for tipo in sorted(df['tipo_veiculo'].unique()):
            tipo_data = df[df['tipo_veiculo'] == tipo]
            if len(tipo_data) > 5:
                tipo_data['km_por_dia'].plot(kind='kde', ax=ax, linewidth=2.5, 
                                            label=tipo, alpha=0.85)
        
        ax.set_xlabel('KM/dia')
        ax.set_ylabel('Densidade')
        ax.set_title('KM/dia por Tipo')
        ax.set_xlim(left=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # KDE por tipo - Horas
        ax = fig.add_subplot(gs[0, 1])
        
        # Histograma ao fundo
        ax.hist(df['h_por_dia'], bins=50, density=True, alpha=0.15, 
                color='gray', edgecolor='none', label='_nolegend_')
        
        # KDE por tipo
        for tipo in sorted(df['tipo_veiculo'].unique()):
            tipo_data = df[df['tipo_veiculo'] == tipo]
            if len(tipo_data) > 5:
                tipo_data['h_por_dia'].plot(kind='kde', ax=ax, linewidth=2.5, 
                                                label=tipo, alpha=0.85)
        
        ax.set_xlabel('Horas/dia')
        ax.set_ylabel('Densidade')
        ax.set_title('Horas/dia por Tipo')
        ax.set_xlim(left=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # KDE por segmento - Deslocamento
        ax = fig.add_subplot(gs[1, 0])
        desl = df[df['tipo_veiculo'] == 'Deslocamento']
        
        if len(desl) > 0:
            # Histograma ao fundo (todos deslocamentos)
            ax.hist(desl['km_por_dia'], bins=40, density=True, alpha=0.15, 
                    color='gray', edgecolor='none', label='_nolegend_')
            
            # KDE por segmento
            for seg in sorted(desl['segmento'].unique())[:6]:
                seg_data = desl[desl['segmento'] == seg]
                if len(seg_data) > 5:
                    seg_data['km_por_dia'].plot(kind='kde', ax=ax, linewidth=2,
                                                label=seg.replace('Desl_C', 'C'), alpha=0.75)
            
            ax.set_xlabel('KM/dia')
            ax.set_ylabel('Densidade')
            ax.set_title('Deslocamento - Top 6 Segmentos')
            ax.set_xlim(left=0)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        # KDE por segmento - Máquina
        ax = fig.add_subplot(gs[1, 1])
        maq = df[df['tipo_veiculo'] == 'Máquina']
        
        if len(maq) > 0:
            # Histograma ao fundo (todas máquinas)
            ax.hist(maq['h_por_dia'], bins=40, density=True, alpha=0.15, 
                    color='gray', edgecolor='none', label='_nolegend_')
            
            # KDE por segmento
            for seg in sorted(maq['segmento'].unique())[:6]:
                seg_data = maq[maq['segmento'] == seg]
                if len(seg_data) > 5:
                    seg_data['h_por_dia'].plot(kind='kde', ax=ax, linewidth=2,
                                                label=seg.replace('Maq_C', 'C'), alpha=0.75)
            
            ax.set_xlabel('Horas/dia')
            ax.set_ylabel('Densidade')
            ax.set_title('Máquina - Top 6 Segmentos')
            ax.set_xlim(left=0)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_segments_stats_pages(self, pdf, pipeline):
        """Cria páginas com estatísticas dos segmentos"""
        df = pipeline.df_final
        segmentos = sorted(df['segmento'].unique())
        
        for i in range(0, len(segmentos), 5):
            batch = segmentos[i:i+5]
            
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(f'Estatísticas dos Segmentos', fontsize=14, fontweight='bold')
            
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            text = []
            
            for seg in batch:
                seg_data = df[df['segmento'] == seg]
                tipo = seg_data['tipo_veiculo'].iloc[0]
                n = len(seg_data)
                pct = n / len(df) * 100
                
                text.append("")
                text.append("="*70)
                text.append(f"{seg} (n={n}, {pct:.1f}%)")
                text.append("="*70)
                
                if tipo == 'Deslocamento':
                    text.append(f"KM/dia:              {seg_data['km_por_dia'].mean():8.2f} ± {seg_data['km_por_dia'].std():6.2f}")
                    text.append(f"Mediana KM:          {seg_data['mediana_km'].mean():8.2f}")
                    text.append(f"Max KM:              {seg_data['max_km'].mean():8.2f}")
                    text.append(f"Score continuidade:  {seg_data['score_continuidade_km'].mean():8.3f}")
                    text.append(f"Taxa semanas ativas: {seg_data['taxa_semanas_ativas_km'].mean():8.2%}")
                    text.append(f"Gap médio:           {seg_data['gap_medio_km'].mean():8.1f} dias")
                    text.append(f"CV:                  {seg_data['cv_km'].mean():8.2f}")
                else:
                    text.append(f"Horas/dia:           {seg_data['h_por_dia'].mean():8.2f} ± {seg_data['h_por_dia'].std():6.2f}")
                    text.append(f"Mediana H:           {seg_data['mediana_h'].mean():8.2f}")
                    text.append(f"Max H:               {seg_data['max_h'].mean():8.2f}")
                    text.append(f"Score continuidade:  {seg_data['score_continuidade_h'].mean():8.3f}")
                    text.append(f"Taxa semanas ativas: {seg_data['taxa_semanas_ativas_h'].mean():8.2%}")
                    text.append(f"Gap médio:           {seg_data['gap_medio_h'].mean():8.1f} dias")
                    text.append(f"CV:                  {seg_data['cv_h'].mean():8.2f}")
            
            ax.text(0.05, 0.95, '\n'.join(text), transform=ax.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

class VehicleAnalysisPipeline:
    """Pipeline completo de análise de veículos usando feature extractors"""
    
    def __init__(self, df_merged: pl.DataFrame, output_base_dir: str = './output'):
        self.df_merged = df_merged
        self.output_base_dir = Path(output_base_dir)
        self.console_output = ""
        
        if self.df_merged['data'].dtype not in [pl.Date, pl.Datetime]:
            self.df_merged = self.df_merged.with_columns(
                pl.col('data').str.strptime(pl.Date, '%Y-%m-%d')
            )
        
        self.df_features = None
        self.df_final = None
        self.df_no_outliers = None
        
        # Feature Extractors
        self.type_extractor = TypeFeatureExtractor()
        self.displacement_extractor = SegmentationFeatureExtractor('km')
        self.machine_extractor = SegmentationFeatureExtractor('h')
        
        # Clusterers
        self.type_clusterer = None
        self.displacement_clusterer = None
        self.machine_clusterer = None
        
        # Classifiers
        self.type_classifier = None
        self.displacement_classifier = None
        self.machine_classifier = None
        
        print("=" * 80)
        print("PIPELINE INICIALIZADO")
        print("=" * 80)
        print(f"Registros: {len(self.df_merged):,}")
        print(f"Veículos: {self.df_merged['veiculo_id'].n_unique()}")
        print(f"Output: {self.output_base_dir}")
        print()
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Gera TODAS as features necessárias para análise completa
        
        Combina features de todos os extractors para ter dataset completo
        """
        print("=" * 80)
        print("GERANDO FEATURES")
        print("=" * 80)
        
        # Extrair features de tipo (6 features)
        print("Extraindo features de tipo (km e h)...")
        features_type = self.type_extractor.extract(self.df_merged)
        
        # Extrair features de segmentação km (16 features)
        print("Extraindo features de segmentação (km)...")
        features_km = self.displacement_extractor.extract(self.df_merged)
        
        # Extrair features de segmentação h (16 features)
        print("Extraindo features de segmentação (h)...")
        features_h = self.machine_extractor.extract(self.df_merged)
        
        # Combinar TODAS as features (6 + 16 + 16 = 38 features)
        features_pd = features_type.merge(
            features_km,
            on='veiculo_id',
            how='left',
            suffixes=('', '_dup')
        )
        
        features_pd = features_pd.merge(
            features_h,
            on='veiculo_id',
            how='left',
            suffixes=('', '_dup')
        )
        
        # Remover colunas duplicadas (km_por_dia e h_por_dia aparecem em type e segmentation)
        dup_cols = [c for c in features_pd.columns if c.endswith('_dup')]
        features_pd = features_pd.drop(columns=dup_cols)
        
        print(f"Features totais: {len(features_pd.columns) - 1}")
        print(f"  • Tipo: {len(self.type_extractor.feature_names)}")
        print(f"  • Segmentação KM: {len(self.displacement_extractor.feature_names)}")
        print(f"  • Segmentação H: {len(self.machine_extractor.feature_names)}")
        print(f"Veículos: {len(features_pd)}")
        print()
        
        self.df_features = features_pd
        return features_pd
    
    def detect_outliers(self) -> pd.DataFrame:
        """Detecta e marca outliers"""
        print("=" * 80)
        print("DETECTANDO OUTLIERS")
        print("=" * 80)
        
        df = self.df_features.copy()
        df['is_outlier'] = False
        df['outlier_reason'] = None
        
        km_muito_baixo = df['km_por_dia'].quantile(0.10)
        km_alto = df['km_por_dia'].quantile(0.75)
        h_muito_baixa = df['h_por_dia'].quantile(0.05)
        
        print(f"Limites:")
        print(f"  • KM/dia muito baixo: < {km_muito_baixo:.2f}")
        print(f"  • KM/dia alto: > {km_alto:.2f}")
        print(f"  • Horas/dia muito baixa: < {h_muito_baixa:.2f}")
        print()
        
        # Outlier 1: KM baixo + H zerada
        mask1 = (df['km_por_dia'] < km_muito_baixo) & (df['km_por_dia'] > 0) & (df['h_por_dia'] < h_muito_baixa)
        df.loc[mask1, 'is_outlier'] = True
        df.loc[mask1, 'outlier_reason'] = 'KM baixo + H zerada'
        print(f"Tipo 1: {mask1.sum()}")
        
        # Outlier 2: KM alto + H zerada
        mask2 = (df['km_por_dia'] > km_alto) & (df['h_por_dia'] < h_muito_baixa) & (~mask1)
        df.loc[mask2, 'is_outlier'] = True
        df.loc[mask2, 'outlier_reason'] = 'KM alto + H zerada'
        print(f"Tipo 2: {mask2.sum()}")
        
        # Outlier 3: Razão extrema
        razao_extrema = df['razao_km_h'].quantile(0.99)
        mask3 = (df['razao_km_h'] > razao_extrema) & (df['h_por_dia'] > 0) & (~mask1) & (~mask2)
        df.loc[mask3, 'is_outlier'] = True
        df.loc[mask3, 'outlier_reason'] = 'Razão extrema'
        print(f"Tipo 3: {mask3.sum()}")
        
        total = df['is_outlier'].sum()
        print(f"\nTotal: {total} ({total/len(df)*100:.2f}%)")
        print()
        
        # Salvar
        outliers_dir = self.output_base_dir / 'outliers'
        outliers_dir.mkdir(parents=True, exist_ok=True)
        df[df['is_outlier']].to_csv(outliers_dir / 'outliers.csv', index=False)
        print(f"✓ Outliers salvos: {outliers_dir / 'outliers.csv'}")
        print()
        
        self.df_no_outliers = df[~df['is_outlier']].copy()
        self.df_features = df
        
        return df
    
    def run_stage1_clustering(self, k: Optional[int] = 2, k_range: Tuple[int, int] = (2, 10)):
        """ETAPA 1: Clustering de tipo"""
        print("\n" + "="*80)
        print("ETAPA 1: CLUSTERING DE TIPO")
        print("="*80 + "\n")
        
        df = self.df_no_outliers.copy()
        
        # Aplicar regras primeiro
        km_threshold = df['km_por_dia'].quantile(0.75)
        
        df['cluster_tipo'] = -1
        df['metodo'] = None
        
        # Regra 1: Alto KM e alta razão
        mask1 = (df['km_por_dia'] > km_threshold) & (df['razao_km_h'] > 20)
        df.loc[mask1, 'cluster_tipo'] = 0
        df.loc[mask1, 'metodo'] = 'Regra'
        
        # Regra 2: Baixo KM e alta H
        mask2 = (df['km_por_dia'] < df['km_por_dia'].quantile(0.25)) & (df['h_por_dia'] > df['h_por_dia'].quantile(0.75))
        mask2 = mask2 & (df['cluster_tipo'] == -1)
        df.loc[mask2, 'cluster_tipo'] = 1
        df.loc[mask2, 'metodo'] = 'Regra'
        
        # Clustering para demais
        mask_cluster = df['cluster_tipo'] == -1
        
        if mask_cluster.sum() > 0:
            print(f"Clustering para {mask_cluster.sum()} veículos...")
            
            self.type_clusterer = TypeClusterer(str(self.output_base_dir / 'clustering' / 'stage1'))
            
            # Usar features do TypeFeatureExtractor
            X_type = df.loc[mask_cluster, self.type_extractor.feature_names].values
            labels = self.type_clusterer.fit(X_type, self.type_extractor.feature_names, k, k_range)
            
            df.loc[mask_cluster, 'cluster_tipo'] = labels.astype(int)
            df.loc[mask_cluster, 'metodo'] = 'Clustering'
            
            self.type_clusterer.plot_projections()
        
        # Mapear para nomes
        cluster_stats = df.groupby('cluster_tipo')[['proporcao_km']].mean()
        if cluster_stats.loc[0, 'proporcao_km'] > cluster_stats.loc[1, 'proporcao_km']:
            tipo_map = {0: 'Deslocamento', 1: 'Máquina'}
        else:
            tipo_map = {1: 'Deslocamento', 0: 'Máquina'}
        
        df['tipo_veiculo'] = df['cluster_tipo'].map(tipo_map)
        
        # Salvar
        clustering_dir = self.output_base_dir / 'clustering' / 'stage1'
        clustering_dir.mkdir(parents=True, exist_ok=True)
        df[['veiculo_id', 'cluster_tipo', 'tipo_veiculo', 'metodo']].to_csv(
            clustering_dir / 'clusters_tipo.csv', index=False
        )
        print(f"✓ Clusters salvos: {clustering_dir / 'clusters_tipo.csv'}\n")
        
        self.df_no_outliers = df
        return df
    
    def run_stage1_classification(self, test_size: float = 0.3):
        """ETAPA 1: Classificação de tipo"""
        df = self.df_no_outliers[self.df_no_outliers['tipo_veiculo'].isin(['Deslocamento', 'Máquina'])].copy()
        
        if len(df) < 10:
            print("⚠️  Dados insuficientes para classificação Etapa 1")
            return
        
        self.type_classifier = TypeClassifier(str(self.output_base_dir / 'classification' / 'stage1'))
        
        # Usar features do TypeFeatureExtractor
        X = df[self.type_extractor.feature_names].values
        y = (df['tipo_veiculo'] == 'Deslocamento').astype(int).values
        
        self.type_classifier.train(X, y, self.type_extractor.feature_names, test_size)
    
    def run_stage2_clustering(self, k: Optional[int] = None, k_range: Tuple[int, int] = (2, 12)):
        """ETAPA 2: Clustering de segmentação"""
        print("\n" + "="*80)
        print("ETAPA 2: CLUSTERING DE SEGMENTAÇÃO")
        print("="*80 + "\n")
        
        df = self.df_no_outliers.copy()
        df['cluster_segmento'] = -1
        df['segmento'] = None
        
        # Deslocamento
        desl_mask = df['tipo_veiculo'] == 'Deslocamento'
        if desl_mask.sum() > 0:
            print(f"DESLOCAMENTO: {desl_mask.sum()} veículos")
            
            self.displacement_clusterer = SegmentationClusterer(
                'displacement',
                str(self.output_base_dir / 'clustering' / 'stage2')
            )
            
            # Usar features do DisplacementFeatureExtractor
            X_desl = df.loc[desl_mask, self.displacement_extractor.feature_names].values
            labels_desl = self.displacement_clusterer.fit(
                X_desl, 
                self.displacement_extractor.feature_names, 
                k, 
                k_range
            )
            
            df.loc[desl_mask, 'cluster_segmento'] = labels_desl.astype(int)
            df.loc[desl_mask, 'segmento'] = [f'Desl_C{l}' for l in labels_desl]
            
            self.displacement_clusterer.plot_projections()
        
        # Máquina
        maq_mask = df['tipo_veiculo'] == 'Máquina'
        if maq_mask.sum() > 0:
            print(f"MÁQUINA: {maq_mask.sum()} veículos")
            
            self.machine_clusterer = SegmentationClusterer(
                'machine',
                str(self.output_base_dir / 'clustering' / 'stage2')
            )
            
            # Usar features do MachineFeatureExtractor
            X_maq = df.loc[maq_mask, self.machine_extractor.feature_names].values
            labels_maq = self.machine_clusterer.fit(
                X_maq, 
                self.machine_extractor.feature_names, 
                k, 
                k_range
            )
            
            df.loc[maq_mask, 'cluster_segmento'] = labels_maq.astype(int)
            df.loc[maq_mask, 'segmento'] = [f'Maq_C{l}' for l in labels_maq]
            
            self.machine_clusterer.plot_projections()
        
        # Salvar
        clustering_dir = self.output_base_dir / 'clustering' / 'stage2'
        clustering_dir.mkdir(parents=True, exist_ok=True)
        df[['veiculo_id', 'tipo_veiculo', 'cluster_segmento', 'segmento']].to_csv(
            clustering_dir / 'clusters_segmento.csv', index=False
        )
        print(f"✓ Clusters salvos: {clustering_dir / 'clusters_segmento.csv'}\n")
        
        self.df_final = df
        return df
    
    def run_stage2_classification(self, test_size: float = 0.3):
        """ETAPA 2: Classificação de segmentos"""
        df = self.df_final.copy()
        
        # Deslocamento
        desl = df[(df['tipo_veiculo'] == 'Deslocamento') & (df['cluster_segmento'] >= 0)].copy()
        
        if len(desl) > 10 and desl['cluster_segmento'].nunique() > 1:
            print("\n" + "="*80)
            print("CLASSIFICAÇÃO: DESLOCAMENTO")
            print("="*80)
            print(f"Veículos: {len(desl)}")
            print(f"Clusters: {desl['cluster_segmento'].nunique()}")
            print()
            
            # Validar clusters
            min_samples = 4
            valid_clusters = desl['cluster_segmento'].value_counts()
            valid_clusters = valid_clusters[valid_clusters >= min_samples]
            
            if len(valid_clusters) < 2:
                print("⚠️  Clusters insuficientes")
                print()
            else:
                desl = desl[desl['cluster_segmento'].isin(valid_clusters.index)].copy()
                
                self.displacement_classifier = SegmentationClassifier(
                    'displacement',
                    str(self.output_base_dir / 'classification' / 'stage2')
                )
                
                # Usar features do DisplacementFeatureExtractor
                X_desl = desl[self.displacement_extractor.feature_names].values
                y_desl = desl['cluster_segmento'].astype(int).values
                
                self.displacement_classifier.train(
                    X_desl, 
                    y_desl, 
                    self.displacement_extractor.feature_names, 
                    test_size
                )
        else:
            print("⚠️  Deslocamento: dados insuficientes")
            print()
        
        # Máquina
        maq = df[(df['tipo_veiculo'] == 'Máquina') & (df['cluster_segmento'] >= 0)].copy()
        
        if len(maq) > 10 and maq['cluster_segmento'].nunique() > 1:
            print("\n" + "="*80)
            print("CLASSIFICAÇÃO: MÁQUINA")
            print("="*80)
            print(f"Veículos: {len(maq)}")
            print(f"Clusters: {maq['cluster_segmento'].nunique()}")
            print()
            
            valid_clusters = maq['cluster_segmento'].value_counts()
            valid_clusters = valid_clusters[valid_clusters >= 4]
            
            if len(valid_clusters) < 2:
                print("⚠️  Clusters insuficientes")
                print()
            else:
                maq = maq[maq['cluster_segmento'].isin(valid_clusters.index)].copy()
                
                self.machine_classifier = SegmentationClassifier(
                    'machine',
                    str(self.output_base_dir / 'classification' / 'stage2')
                )
                
                # Usar features do MachineFeatureExtractor
                X_maq = maq[self.machine_extractor.feature_names].values
                y_maq = maq['cluster_segmento'].astype(int).values
                
                self.machine_classifier.train(
                    X_maq, 
                    y_maq, 
                    self.machine_extractor.feature_names, 
                    test_size
                )
        else:
            print("⚠️  Máquina: dados insuficientes")
            print()
    
    def generate_final_report(self):
        """Gera relatório final com estatísticas claras"""
        print("\n" + "="*80)
        print("RELATÓRIO FINAL")
        print("="*80 + "\n")
        
        report_dir = self.output_base_dir / 'reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.df_final
        
        # Summary JSON
        summary = {
            'total_veiculos': int(len(df)),
            'outliers': int((self.df_features['is_outlier']).sum()),
            'veiculos_validos': int(len(df)),
            'tipos': {str(k): int(v) for k, v in df['tipo_veiculo'].value_counts().to_dict().items()},
            'segmentos': {str(k): int(v) for k, v in df['segmento'].value_counts().to_dict().items()}
        }
        
        with open(report_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary: {report_dir / 'summary.json'}")
        
        # Estatísticas detalhadas por segmento
        stats_list = []
        
        print("\n" + "="*80)
        print("ESTATÍSTICAS DOS SEGMENTOS")
        print("="*80)
        
        for seg in sorted(df['segmento'].unique()):
            seg_data = df[df['segmento'] == seg]
            tipo = seg_data['tipo_veiculo'].iloc[0]
            n = len(seg_data)
            pct = n / len(df) * 100
            
            print(f"\n{seg} (n={n}, {pct:.1f}%)")
            print("="*70)
            
            stats = {
                'segmento': seg,
                'tipo_veiculo': tipo,
                'n': n,
                'pct': pct
            }
            
            if tipo == 'Deslocamento':
                # Volume normalizado
                stats['km_por_dia_mean'] = seg_data['km_por_dia'].mean()
                stats['km_por_dia_std'] = seg_data['km_por_dia'].std()
                stats['km_por_dia_median'] = seg_data['km_por_dia'].median()
                
                # Estatísticas dos dias ativos
                stats['media_km_mean'] = seg_data['media_km'].mean()
                stats['mediana_km_mean'] = seg_data['mediana_km'].mean()
                stats['max_km_mean'] = seg_data['max_km'].mean()
                
                # Continuidade
                stats['score_continuidade_mean'] = seg_data['score_continuidade_km'].mean()
                stats['taxa_semanas_ativas_mean'] = seg_data['taxa_semanas_ativas_km'].mean()
                stats['taxa_dias_ativos_mean'] = seg_data['taxa_dias_ativos_km'].mean()
                
                # Gaps
                stats['gap_medio_mean'] = seg_data['gap_medio_km'].mean()
                stats['gap_max_mean'] = seg_data['gap_max_km'].mean()
                
                # Variabilidade
                stats['cv_mean'] = seg_data['cv_km'].mean()
                
                print(f"Volume (normalizado por dia):")
                print(f"  KM/dia (cluster):      {stats['km_por_dia_mean']:8.2f} ± {stats['km_por_dia_std']:6.2f}")
                print(f"  KM/dia (mediana):      {stats['km_por_dia_median']:8.2f}")
                print()
                print(f"Estatísticas dos Dias Ativos (média entre veículos):")
                print(f"  Média KM/dia ativo:    {stats['media_km_mean']:8.2f}")
                print(f"  Mediana KM/dia ativo:  {stats['mediana_km_mean']:8.2f}")
                print(f"  Max KM/dia:            {stats['max_km_mean']:8.2f}")
                print()
                print(f"Continuidade:")
                print(f"  Score continuidade:    {stats['score_continuidade_mean']:8.3f}")
                print(f"  Taxa semanas ativas:   {stats['taxa_semanas_ativas_mean']:8.2%}")
                print(f"  Taxa dias ativos:      {stats['taxa_dias_ativos_mean']:8.2%}")
                print(f"  Gap médio:             {stats['gap_medio_mean']:8.1f} dias")
                print(f"  Gap máximo:            {stats['gap_max_mean']:8.1f} dias")
                print()
                print(f"Variabilidade:")
                print(f"  Coef. Variação (CV):   {stats['cv_mean']:8.2f}")
                
            else:  # Máquina
                stats['h_por_dia_mean'] = seg_data['h_por_dia'].mean()
                stats['h_por_dia_std'] = seg_data['h_por_dia'].std()
                stats['h_por_dia_median'] = seg_data['h_por_dia'].median()
                
                stats['media_h_mean'] = seg_data['media_h'].mean()
                stats['mediana_h_mean'] = seg_data['mediana_h'].mean()
                stats['max_h_mean'] = seg_data['max_h'].mean()
                
                stats['score_continuidade_mean'] = seg_data['score_continuidade_h'].mean()
                stats['taxa_semanas_ativas_mean'] = seg_data['taxa_semanas_ativas_h'].mean()
                stats['taxa_dias_ativos_mean'] = seg_data['taxa_dias_ativos_h'].mean()
                
                stats['gap_medio_mean'] = seg_data['gap_medio_h'].mean()
                stats['gap_max_mean'] = seg_data['gap_max_h'].mean()
                
                stats['cv_mean'] = seg_data['cv_h'].mean()
                
                print(f"Volume (normalizado por dia):")
                print(f"  Horas/dia (cluster):   {stats['h_por_dia_mean']:8.2f} ± {stats['h_por_dia_std']:6.2f}")
                print(f"  Horas/dia (mediana):   {stats['h_por_dia_median']:8.2f}")
                print()
                print(f"Estatísticas dos Dias Ativos (média entre veículos):")
                print(f"  Média H/dia ativo:     {stats['media_h_mean']:8.2f}")
                print(f"  Mediana H/dia ativo:   {stats['mediana_h_mean']:8.2f}")
                print(f"  Max H/dia:             {stats['max_h_mean']:8.2f}")
                print()
                print(f"Continuidade:")
                print(f"  Score continuidade:    {stats['score_continuidade_mean']:8.3f}")
                print(f"  Taxa semanas ativas:   {stats['taxa_semanas_ativas_mean']:8.2%}")
                print(f"  Taxa dias ativos:      {stats['taxa_dias_ativos_mean']:8.2%}")
                print(f"  Gap médio:             {stats['gap_medio_mean']:8.1f} dias")
                print(f"  Gap máximo:            {stats['gap_max_mean']:8.1f} dias")
                print()
                print(f"Variabilidade:")
                print(f"  Coef. Variação (CV):   {stats['cv_mean']:8.2f}")
            
            stats_list.append(stats)
        
        df_stats = pd.DataFrame(stats_list)
        df_stats.to_csv(report_dir / 'segment_statistics.csv', index=False)
        print(f"\n✓ Estatísticas salvas: {report_dir / 'segment_statistics.csv'}")
        print()
        
        return df_stats
    
    def run_complete_pipeline(self,
                             k_stage1: Optional[int] = 2,
                             k_range_stage1: Tuple[int, int] = (2, 10),
                             k_stage2: Optional[int] = None,
                             k_range_stage2: Tuple[int, int] = (2, 12),
                             train_classifiers: bool = True,
                             test_size: float = 0.3,
                             generate_pdf: bool = True):
        """
        Executa pipeline completo com captura de console
        
        Parameters:
        -----------
        k_stage1 : int, optional
            Número de clusters Etapa 1 (None = busca automática)
        k_range_stage1 : tuple
            Range para busca Etapa 1
        k_stage2 : int, optional
            Número de clusters Etapa 2 (None = busca automática)
        k_range_stage2 : tuple
            Range para busca Etapa 2
        train_classifiers : bool
            Se True, treina classificadores
        test_size : float
            Proporção para teste
        generate_pdf : bool
            Se True, gera relatório PDF
        """
        # Capturar console
        console_capture = io.StringIO()
        
        class TeeOutput:
            def __init__(self, *outputs):
                self.outputs = outputs
            
            def write(self, text):
                for output in self.outputs:
                    output.write(text)
            
            def flush(self):
                for output in self.outputs:
                    output.flush()
        
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(original_stdout, console_capture)
        
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETO")
            print("="*80)
            print(f"Timestamp: {timestamp}")
            print(f"Output: {self.output_base_dir}")
            print("="*80 + "\n")
            
            # 1. Features (usando extractors)
            self.engineer_features()
            
            # 2. Outliers
            self.detect_outliers()
            
            # 3. ETAPA 1: Clustering
            self.run_stage1_clustering(k_stage1, k_range_stage1)
            
            # 4. ETAPA 1: Classificação
            if train_classifiers:
                self.run_stage1_classification(test_size)
            
            # 5. ETAPA 2: Clustering
            self.run_stage2_clustering(k_stage2, k_range_stage2)
            
            # 6. ETAPA 2: Classificação
            if train_classifiers:
                self.run_stage2_classification(test_size)
            
            # 7. Relatório Final
            self.generate_final_report()
            
            # 8. Salvar dataset final
            final_path = self.output_base_dir / 'veiculos_segmentados_final.csv'
            self.df_final.to_csv(final_path, index=False)
            print(f"✓ Dataset final: {final_path}")
            
            print("\n" + "="*80)
            print("PIPELINE CONCLUÍDO")
            print("="*80 + "\n")
            
        finally:
            sys.stdout = original_stdout
            self.console_output = console_capture.getvalue()
        
        # 9. Gerar PDF
        if generate_pdf:
            pdf_gen = PDFReportGenerator(self.output_base_dir / 'reports')
            pdf_gen.generate(self, self.console_output)
        
        return {
            'df_features': self.df_features,
            'df_final': self.df_final,
            'type_clusterer': self.type_clusterer,
            'displacement_clusterer': self.displacement_clusterer,
            'machine_clusterer': self.machine_clusterer,
            'type_classifier': self.type_classifier,
            'displacement_classifier': self.displacement_classifier,
            'machine_classifier': self.machine_classifier,
            'console_output': self.console_output
        }