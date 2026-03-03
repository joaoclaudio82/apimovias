import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from darts import TimeSeries
import warnings
warnings.filterwarnings('ignore')


from moviasai.feature_extraction import TypeFeatureExtractor, SegmentationFeatureExtractor
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
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
             test_size: float = 0.3, cv_folds: int = 5) -> Dict:
        """Treina múltiplos modelos"""
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
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
        print()
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        }
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
            cv = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='f1_weighted')
            print(f"{name}: Acc={acc_test:.4f}, F1={f1_test:.4f}, Acc_train={acc_train:.4f}, F1_train={f1_train:.4f}, CV={cv.mean():.4f}±{cv.std():.4f}")
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
                'y_train': y_train                  
            }      
                    
        self.best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_test'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n✓ Melhor: {self.best_model_name} (F1={self.results[self.best_model_name]['f1_test']:.4f})\n")
        
        self.save_models()
        return self.results
    
    def save_models(self):
        """Salva modelos em PKL e ONNX (com scaler em JSON para ONNX)"""
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
            
            # Salvar scaler em JSON para ONNX
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


class TypeClassifier(BaseClassifier):
    """
    Classificador de Tipo de Veículo (Etapa 1)
    
    Classifica: Máquina (0) ou Deslocamento (1)
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
        Parameters:
        -----------
        vehicle_type : str
            'displacement' ou 'machine'
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
    - PKL e ONNX (detectado automaticamente pela extensão)
    """
    def __init__(self, model_path: str):
        """
        Parameters:
        -----------
        model_path : str
            Caminho do arquivo do modelo (.pkl ou .onnx)
            - Se .onnx: requer _scaler.json e _metadata.json no mesmo diretório
        """
        self.model_path = Path(model_path)
        self._use_onnx = self.model_path.suffix.lower() == '.onnx'
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metadata = {}
        self.feature_extractor = None
        self.onnx_session = None
        self._validate_files()
        self._load()
        
    def _validate_files(self):
        """Valida existência de arquivos necessários"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        if self._use_onnx:
            # Verificar arquivos auxiliares do ONNX
            scaler_path = self.model_path.parent / f'{self.model_path.stem}_scaler.json'
            metadata_path = self.model_path.parent / f'{self.model_path.stem}_metadata.json'
            missing = []
            if not scaler_path.exists():
                missing.append(str(scaler_path))
            if not metadata_path.exists():
                missing.append(str(metadata_path))
            if missing:
                raise FileNotFoundError(
                    f"Arquivos obrigatórios para ONNX não encontrados:\n" +
                    "\n".join(f"  - {f}" for f in missing)
                )
    def _load(self):
        """Carrega modelo e componentes"""
        if self._use_onnx:
            self._load_onnx()
        else:
            self._load_pkl()
        self._load_feature_extractor()
        
    def _load_pkl(self):
        """Carrega modelo PKL"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.metadata = data.get('metadata', {})
        print(f"✓ Modelo PKL carregado: {self.model_path}")
        print(f"  Nome: {self.metadata.get('name', 'N/A')}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Classes: {self.metadata.get('n_classes', 'N/A')}")
        
    def _load_onnx(self):
        """Carrega modelo ONNX com scaler em JSON"""
        scaler_json_path = self.model_path.parent / f'{self.model_path.stem}_scaler.json'
        metadata_path = self.model_path.parent / f'{self.model_path.stem}_metadata.json'
        # Carregar ONNX
        self.onnx_session = rt.InferenceSession(str(self.model_path))
        # Carregar scaler do JSON
        with open(scaler_json_path, 'r') as f:
            scaler_dict = json.load(f)
        # Reconstruir StandardScaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(scaler_dict['mean'])
        self.scaler.scale_ = np.array(scaler_dict['scale'])
        self.scaler.var_ = np.array(scaler_dict['var'])
        self.scaler.n_features_in_ = scaler_dict['n_features_in']
        self.scaler.n_samples_seen_ = scaler_dict['n_samples_seen']
        # Carregar metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata['feature_names']
        print(f"✓ Modelo ONNX carregado: {self.model_path}")
        print(f"  Scaler (JSON): {scaler_json_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Nome: {self.metadata.get('name', 'N/A')}")
        print(f"  Features: {len(self.feature_names)}")
        
    def _load_feature_extractor(self):
        """Carrega feature extractor apropriado baseado no modelo"""
        model_name = self.metadata.get('name', '')
        if 'stage1_type' in model_name:
            self.feature_extractor = TypeFeatureExtractor()
            print(f"  Extrator: TypeFeatureExtractor")
            print(f"  Requer: km_dia_clean E h_dia_clean")
        elif 'stage2_displacement' in model_name:
            self.feature_extractor = SegmentationFeatureExtractor('km')
            print(f"  Extrator: SegmentationFeatureExtractor(km)")
            print(f"  Requer: km_dia_clean APENAS")
        elif 'stage2_machine' in model_name:
            self.feature_extractor = SegmentationFeatureExtractor('h')
            print(f"  Extrator: SegmentationFeatureExtractor(h)")
            print(f"  Requer: h_dia_clean APENAS")
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_name}")
        print()
        
    def _predict(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Executa predição (PKL ou ONNX)"""
        if self._use_onnx:
            # ONNX
            input_name = self.onnx_session.get_inputs()[0].name
            label_name = self.onnx_session.get_outputs()[0].name
            predictions = self.onnx_session.run([label_name], {input_name: X_scaled.astype(np.float32)})[0]
            # Tentar obter probabilidades
            try:
                prob_name = self.onnx_session.get_outputs()[1].name
                probabilities = self.onnx_session.run([prob_name], {input_name: X_scaled.astype(np.float32)})[0]
                probabilities = pd.DataFrame(probabilities).values
            except:
                probabilities = None
            return predictions, probabilities
        else:
            # PKL
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled) if hasattr(self.model, 'predict_proba') else None
            return predictions, probabilities
          
    def predict_batch(self, df: Union[pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """
        Prediz em lote
        Parameters:
        -----------
        df : pl.DataFrame ou pd.DataFrame
            Formato depende do modelo:
            - stage1_type: veiculo_id, data, km_dia_clean, h_dia_clean
            - stage2_displacement: veiculo_id, data, km_dia_clean
            - stage2_machine: veiculo_id, data, h_dia_clean
        Returns:
        --------
        pd.DataFrame
            veiculo_id, prediction, proba_class_*
        """
        # Extrair features
        features = self.feature_extractor.extract(df)
        # Preparar X
        X = features[self.feature_names].values
        # Normalizar usando scaler (funciona para PKL e ONNX)
        X_scaled = self.scaler.transform(X)
        # Predizer
        predictions, probabilities = self._predict(X_scaled)
        # Montar resultado
        result = pd.DataFrame({
            'veiculo_id': features['veiculo_id'],
            'prediction': predictions
        })
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                result[f'proba_class_{i}'] = probabilities[:, i]
        return result
        
    def predict_single(self, df: Union[pl.DataFrame, pd.DataFrame], veiculo_id: int) -> Dict:
        """
        Prediz para um único veículo
        Parameters:
        -----------
        df : pl.DataFrame ou pd.DataFrame
            DataFrame com dados do veículo
        veiculo_id : int
            ID do veículo a predizer
        Returns:
        --------
        dict
            {'veiculo_id': int, 'prediction': int, 'probabilities': dict}
        """
        if isinstance(df, pl.DataFrame):
            df_vehicle = df.filter(pl.col('veiculo_id') == veiculo_id)
        else:
            df_vehicle = df[df['veiculo_id'] == veiculo_id]
        if len(df_vehicle) == 0:
            raise ValueError(f"Veículo {veiculo_id} não encontrado")
        result_df = self.predict_batch(df_vehicle)
        row = result_df.iloc[0]
        result = {
            'veiculo_id': int(row['veiculo_id']),
            'prediction': int(row['prediction'])
        }
        prob_cols = [c for c in result_df.columns if c.startswith('proba_')]
        if prob_cols:
            result['probabilities'] = {col: float(row[col]) for col in prob_cols}
        return result
        
    def predict_timeseries(self, ts: TimeSeries, veiculo_id: int) -> Dict:
        """
        Prediz para uma TimeSeries do Darts
        Parameters:
        -----------
        ts : darts.TimeSeries
            Série temporal com os targets necessários:
            - stage1_type: 2 componentes (km_dia_clean, h_dia_clean)
            - stage2_displacement: 1 componente (km_dia_clean)
            - stage2_machine: 1 componente (h_dia_clean)
        veiculo_id : int
            ID do veículo
        Returns:
        --------
        dict
            {'veiculo_id': int, 'prediction': int, 'probabilities': dict}
        """
        # Extrair features da TimeSeries
        features = self.feature_extractor.extract_fromm_timeseries(ts, veiculo_id)
        # Preparar X
        X = features[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        # Predizer
        predictions, probabilities = self._predict(X_scaled)
        result = {
            'veiculo_id': veiculo_id,
            'prediction': int(predictions[0])
        }
        if probabilities is not None:
            result['probabilities'] = {
                f'proba_class_{i}': float(probabilities[0, i])
                for i in range(probabilities.shape[1])
            }
        return result