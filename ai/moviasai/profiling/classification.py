import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Union

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

import warnings
warnings.filterwarnings('ignore')

from moviasai.profiling.feature_extraction import TypeFeatureExtractor, SegmentationFeatureExtractor


# ============================================================================
# CLASSIFIERS
# ============================================================================

class BaseClassifier:
    """Classe base para classificação"""
    
    def __init__(self, name: str, output_dir: str):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.feature_names = []
        self.results = {}
        self.metadata = {}
        self._use_onnx = False
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        test_size: float = 0.3, 
        cv_folds: int = 5
    ) -> Dict:
        """
        Treina múltiplos modelos e seleciona o melhor
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        feature_names : List[str]
            Nomes das features
        test_size : float
            Proporção de teste
        cv_folds : int
            Folds para cross-validation
        
        Returns
        -------
        Dict
            Resultados dos modelos treinados
        """
        print(f"\n{'='*80}")
        print(f"TREINAMENTO - {self.name}")
        print(f"{'='*80}")
        print(f"Amostras: {len(X)}, Features: {len(feature_names)}, Classes: {len(np.unique(y))}")
        print()
        
        self.feature_names = feature_names
        
        self.metadata = {
            'name': self.name,
            'n_samples': int(len(X)),
            'n_features': int(len(feature_names)),
            'n_classes': int(len(np.unique(y))),
            'feature_names': feature_names,
            'classes': [int(c) for c in np.unique(y)],
            'timestamp': datetime.now().isoformat()
        }
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
        print()
        
        # Normalização
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modelos
        self.models = {
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000, 
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10, 
                class_weight='balanced'
            ),
        }
        
        # Treinar e avaliar
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            
            # Previsão no teste
            y_pred_test = model.predict(X_test_scaled)
            acc_test = accuracy_score(y_test, y_pred_test)
            f1_test = f1_score(y_test, y_pred_test, average='weighted')
            
            # Previsão no treino
            y_pred_train = model.predict(X_train_scaled)
            acc_train = accuracy_score(y_train, y_pred_train)
            f1_train = f1_score(y_train, y_pred_train, average='weighted')
            
            # Cross-validation
            cv = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=cv_folds, scoring='f1_weighted'
            )
            
            print(
                f"{name}: "
                f"Acc={acc_test:.4f}, F1={f1_test:.4f}, "
                f"Acc_train={acc_train:.4f}, F1_train={f1_train:.4f}, "
                f"CV={cv.mean():.4f}±{cv.std():.4f}"
            )
            
            self.results[name] = {
                'model': model,
                'acc_test': float(acc_test),
                'f1_test': float(f1_test),
                'acc_train': float(acc_train),
                'f1_train': float(f1_train),
                'f1_cv_mean': float(cv.mean()),
                'f1_cv_std': float(cv.std()),
                'y_pred_test': y_pred_test,
                'y_pred_train': y_pred_train,
                'y_test': y_test,
                'y_train': y_train,
                'X_test': X_test,
                'X_train': X_train
            }
        
        # Selecionar melhor
        self.best_model_name = max(
            self.results.keys(), 
            key=lambda k: self.results[k]['f1_test']
        )
        self.best_model = self.results[self.best_model_name]['model']
        
        print(
            f"\n✓ Melhor: {self.best_model_name} "
            f"(F1={self.results[self.best_model_name]['f1_test']:.4f})\n"
        )
        
        self.save_models()
        return self.results
    
    def save_models(self):
        """Salva modelos em PKL e ONNX"""
        # Salvar cada modelo em PKL
        for name, result in self.results.items():
            pkl_path = self.output_dir / f'{self.name}_{name}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'model': result['model'],
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'metadata': self.metadata
                }, f)
        
        # Salvar melhor modelo em PKL
        best_pkl = self.output_dir / f'{self.name}_BEST.pkl'
        with open(best_pkl, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metadata': self.metadata,
                'model_name': self.best_model_name
            }, f)
        
        print(f"✓ Modelos PKL salvos: {self.output_dir}")
        
        # Salvar em ONNX
        try:
            onnx_path = self.output_dir / f'{self.name}_BEST.onnx'
            scaler_json_path = self.output_dir / f'{self.name}_BEST_scaler.json'
            metadata_path = self.output_dir / f'{self.name}_BEST_metadata.json'
            
            # Converter modelo para ONNX
            initial_type = [('float_input', FloatTensorType([None, len(self.feature_names)]))]
            onnx_model = convert_sklearn(self.best_model, initial_types=initial_type)
            
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            # Salvar scaler em JSON
            scaler_dict = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'var': self.scaler.var_.tolist(),
                'n_features_in': int(self.scaler.n_features_in_),
                'n_samples_seen': int(self.scaler.n_samples_seen_)
            }
            
            with open(scaler_json_path, 'w') as f:
                json.dump(scaler_dict, f, indent=2)
            
            # Salvar metadata
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"✓ ONNX salvo: {onnx_path}")
            print(f"  • Scaler (JSON): {scaler_json_path}")
            print(f"  • Metadata: {metadata_path}")
        
        except Exception as e:
            print(f"⚠️  Erro ao salvar ONNX: {e}")
        print()
    
    @classmethod
    def load_model(cls, model_path: str, format: str = 'auto'):
        """
        Carrega modelo salvo (PKL ou ONNX)
        
        Parameters
        ----------
        model_path : str
            Caminho do arquivo do modelo
        format : str
            'pkl', 'onnx' ou 'auto' (detecta pela extensão)
        
        Returns
        -------
        BaseClassifier
            Instância do classificador carregado
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Detectar formato
        if format == 'auto':
            format = 'onnx' if model_path.suffix.lower() == '.onnx' else 'pkl'
        
        if format == 'pkl':
            return cls._load_pkl(model_path)
        elif format == 'onnx':
            return cls._load_onnx(model_path)
        else:
            raise ValueError(f"Formato inválido: {format}")
    
    @classmethod
    def _load_pkl(cls, model_path: Path):
        """Carrega modelo PKL"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Criar instância
        instance = cls.__new__(cls)
        instance.name = data['metadata'].get('name', model_path.stem)
        instance.output_dir = model_path.parent
        instance.best_model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.metadata = data['metadata']
        instance.best_model_name = data.get('model_name', 'unknown')
        instance.models = {}
        instance.results = {}
        instance._use_onnx = False
        
        print(f"✓ Modelo PKL carregado: {model_path}")
        print(f"  Nome: {instance.name}")
        print(f"  Features: {len(instance.feature_names)}")
        print(f"  Classes: {instance.metadata.get('n_classes', 'N/A')}")
        print()
        
        return instance
    
    @classmethod
    def _load_onnx(cls, model_path: Path):
        """Carrega modelo ONNX"""
        # Validar arquivos auxiliares
        scaler_json_path = model_path.parent / f'{model_path.stem}_scaler.json'
        metadata_path = model_path.parent / f'{model_path.stem}_metadata.json'
        
        if not scaler_json_path.exists():
            raise FileNotFoundError(f"Scaler não encontrado: {scaler_json_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata não encontrada: {metadata_path}")
        
        # Carregar ONNX
        onnx_session = rt.InferenceSession(str(model_path))
        
        # Carregar scaler do JSON
        with open(scaler_json_path, 'r') as f:
            scaler_dict = json.load(f)
        
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_dict['mean'])
        scaler.scale_ = np.array(scaler_dict['scale'])
        scaler.var_ = np.array(scaler_dict['var'])
        scaler.n_features_in_ = scaler_dict['n_features_in']
        scaler.n_samples_seen_ = scaler_dict['n_samples_seen']
        
        # Carregar metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Criar instância
        instance = cls.__new__(cls)
        instance.name = metadata.get('name', model_path.stem)
        instance.output_dir = model_path.parent
        instance.best_model = onnx_session
        instance.scaler = scaler
        instance.feature_names = metadata['feature_names']
        instance.metadata = metadata
        instance.best_model_name = 'ONNX'
        instance.models = {}
        instance.results = {}
        instance._use_onnx = True
        
        print(f"✓ Modelo ONNX carregado: {model_path}")
        print(f"  Scaler (JSON): {scaler_json_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Nome: {instance.name}")
        print(f"  Features: {len(instance.feature_names)}")
        print()
        
        return instance
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prediz classes e probabilidades
        
        Parameters
        ----------
        X : np.ndarray
            Features (não normalizadas)
        
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (predictions, probabilities)
        """
        X_scaled = self.scaler.transform(X)
        
        if self._use_onnx:
            return self._predict_onnx(X_scaled)
        else:
            return self._predict_pkl(X_scaled)
    
    def _predict_pkl(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predição com modelo PKL"""
        predictions = self.best_model.predict(X_scaled)
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_scaled)
        else:
            probabilities = None
        
        return predictions, probabilities
    
    def _predict_onnx(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predição com modelo ONNX"""
        input_name = self.best_model.get_inputs()[0].name
        label_name = self.best_model.get_outputs()[0].name
        
        predictions = self.best_model.run(
            [label_name],
            {input_name: X_scaled.astype(np.float32)}
        )[0]
        
        # Tentar obter probabilidades
        try:
            prob_name = self.best_model.get_outputs()[1].name
            probabilities = self.best_model.run(
                [prob_name],
                {input_name: X_scaled.astype(np.float32)}
            )[0]
            probabilities = pd.DataFrame(probabilities).values
        except:
            probabilities = None
        
        return predictions, probabilities


class TypeClassifier(BaseClassifier):
    """
    Classificador de Tipo de Veículo (Etapa 1)
    Classifica: H (0) ou KM (1)
    """
    
    def __init__(self, output_dir: str = './output/classification/stage1'):
        super().__init__('stage1_type', output_dir)


class SegmentationClassifier(BaseClassifier):
    """
    Classificador de Segmentação (Etapa 2)
    Classifica em clusters: 0, 1, 2, ...
    """
    
    def __init__(self, vehicle_type: str, output_dir: str = './output/classification/stage2'):
        """
        Parameters
        ----------
        vehicle_type : str
            'km' ou 'h'
        """
        super().__init__(f'stage2_{vehicle_type}', output_dir)
        self.vehicle_type = vehicle_type


# ============================================================================
# PREDICTOR
# ============================================================================

class VehiclePredictor:
    """
    Predictor genérico para uso em produção
    
    Suporta:
    - Predição em lote (DataFrame)
    - Predição individual
    - Predição TimeSeries (Darts)
    - PKL e ONNX (detectado automaticamente)
    """
    
    def __init__(self, model_path: str, classifier_class=None):
        """
        Parameters
        ----------
        model_path : str
            Caminho do arquivo do modelo (.pkl ou .onnx)
        classifier_class : class, optional
            Classe do classificador (TypeClassifier, SegmentationClassifier)
            Se None, usa BaseClassifier
        """
        self.model_path = Path(model_path)
        
        # Carregar usando o classificador apropriado
        if classifier_class is None:
            self.classifier = BaseClassifier.load_model(str(self.model_path))
        else:
            self.classifier = classifier_class.load_model(str(self.model_path))
        
        # Atalhos
        self.feature_names = self.classifier.feature_names
        self.metadata = self.classifier.metadata
        
        # Carregar feature extractor
        self._load_feature_extractor()
    
    def _load_feature_extractor(self):
        """Carrega feature extractor apropriado baseado no modelo"""
        model_name = self.metadata.get('name', '')
        
        if 'stage1_type' in model_name or 'type' in model_name:
            self.feature_extractor = TypeFeatureExtractor()
            print(f"  Extrator: TypeFeatureExtractor")
            print(f"  Requer: km_dia_clean E h_dia_clean")
        elif 'stage2_km' in model_name or 'km' in model_name:
            self.feature_extractor = SegmentationFeatureExtractor('km')
            print(f"  Extrator: SegmentationFeatureExtractor(km)")
            print(f"  Requer: km_dia_clean APENAS")
        elif 'stage2_h' in model_name or 'h' in model_name:
            self.feature_extractor = SegmentationFeatureExtractor('h')
            print(f"  Extrator: SegmentationFeatureExtractor(h)")
            print(f"  Requer: h_dia_clean APENAS")
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_name}")
        
        print()
    
    def predict_batch(self, df: Union[pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """
        Prediz em lote
        
        Parameters
        ----------
        df : pl.DataFrame ou pd.DataFrame
            DataFrame com dados necessários
        
        Returns
        -------
        pd.DataFrame
            veiculo_id, prediction, proba_class_*
        """
        # Extrair features
        features = self.feature_extractor.extract(df)
        
        # Preparar X
        X = features[self.feature_names].values
        
        # Predizer
        predictions, probabilities = self.classifier.predict(X)
        
        # Montar resultado
        result = pd.DataFrame({
            'veiculo_id': features['veiculo_id'],
            'prediction': predictions
        })
        
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                result[f'proba_class_{i}'] = probabilities[:, i]
        
        return result