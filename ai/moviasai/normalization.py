import json
import numpy as np


class WinsorizedMinMaxScaler:
    """
    Scaler por veículo com winsorização leve (p1-p99) + MinMaxScaler.
    
    Este scaler foi projetado especificamente para séries de deslocamento:
    - Remove apenas outliers extremos causados por falhas de GPS/odômetro.
    - Preserva a forma original da série.
    - Mantém a faixa [0,1], que é ideal para TCN/NBEATS/TFT/etc.
    """
    
    def __init__(self, p_lower=0, p_upper=99):
        self.p_lower = p_lower
        self.p_upper = p_upper
        self.min = None
        self.max = None
        self.lower = None
        self.upper = None
        # Armazenar histórico para cálculo incremental de percentis
        self._values_history = []
        self._n_samples = 0
    
    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self._values_history = values.tolist()
        self._n_samples = len(values)
        
        # 1. Winsorização leve
        self.lower = np.percentile(values, self.p_lower)
        self.upper = np.percentile(values, self.p_upper)
        clipped = np.clip(values, self.lower, self.upper)
        
        # 2. MinMax com valores winsorizados
        self.min = float(clipped.min())
        self.max = float(clipped.max())
        if abs(self.max - self.min) < 1e-9:
            self.max = self.min + 1e-6  # impedir divisão por zero
    
    def update(self, values, method='recompute'):
        """
        Atualiza o escalonador com novos dados.
        
        Parameters:
        -----------
        values : array-like
            Novos valores para atualizar o scaler
        method : str, default='recompute'
            Método de atualização:
            - 'recompute': Recalcula tudo com histórico completo (mais preciso)
            - 'incremental': Atualização aproximada sem histórico completo (mais rápido)
        """
        values = np.asarray(values, dtype=float)
        
        if method == 'recompute':
            # Método mais preciso: mantém histórico e recalcula
            self._values_history.extend(values.tolist())
            self._n_samples += len(values)
            
            # Limitar histórico para evitar crescimento infinito (opcional)
            max_history = 364  
            if len(self._values_history) > max_history:
                # Manter apenas os valores mais recentes
                self._values_history = self._values_history[-max_history:]
            
            # Recalcular com todo o histórico
            all_values = np.array(self._values_history)
            self.lower = np.percentile(all_values, self.p_lower)
            self.upper = np.percentile(all_values, self.p_upper)
            clipped = np.clip(all_values, self.lower, self.upper)
            self.min = float(clipped.min())
            self.max = float(clipped.max())
            
        elif method == 'incremental':
            # Método aproximado: atualiza sem histórico completo
            self._n_samples += len(values)
            
            # Atualizar percentis (aproximação)
            new_lower = np.percentile(values, self.p_lower)
            new_upper = np.percentile(values, self.p_upper)
            
            # Média ponderada entre antigo e novo
            weight_old = 0.7  # peso para valores antigos
            weight_new = 0.3  # peso para valores novos
            self.lower = weight_old * self.lower + weight_new * new_lower
            self.upper = weight_old * self.upper + weight_new * new_upper
            
            # Atualizar min/max
            clipped = np.clip(values, self.lower, self.upper)
            new_min = float(clipped.min())
            new_max = float(clipped.max())
            
            self.min = min(self.min, new_min)
            self.max = max(self.max, new_max)
        
        else:
            raise ValueError(f"Método '{method}' não reconhecido. Use 'recompute' ou 'incremental'.")
        
        # Garantir que min != max
        if abs(self.max - self.min) < 1e-9:
            self.max = self.min + 1e-6
    
    def transform(self, values):
        values = np.asarray(values, dtype=float)
        # 1. winsor
        clipped = np.clip(values, self.lower, self.upper)
        # 2. minmax
        return (clipped - self.min) / (self.max - self.min)
    
    def inverse_transform(self, scaled):
        scaled = np.asarray(scaled, dtype=float)
        return scaled * (self.max - self.min) + self.min
    
    def to_json(self, filepath=None, include_history=False):
        """
        Serializa o scaler para JSON.
        
        Parameters:
        -----------
        filepath : str, optional
            Caminho do arquivo para salvar. Se None, retorna string JSON.
        include_history : bool, default=False
            Se True, inclui o histórico de valores (pode ser grande).
            Se False, salva apenas os parâmetros essenciais.
        
        Returns:
        --------
        str or None
            String JSON se filepath=None, caso contrário None.
        """
        data = {
            'p_lower': self.p_lower,
            'p_upper': self.p_upper,
            'min': self.min,
            'max': self.max,
            'lower': self.lower,
            'upper': self.upper,
            'n_samples': self._n_samples
        }
        
        if include_history:
            data['values_history'] = self._values_history
        
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        else:
            return json_str
    
    @classmethod
    def from_json(cls, json_input, is_filepath=True):
        """
        Carrega o scaler de um JSON.
        
        Parameters:
        -----------
        json_input : str
            Caminho do arquivo JSON ou string JSON
        is_filepath : bool, default=True
            Se True, json_input é um caminho de arquivo.
            Se False, json_input é uma string JSON.
        
        Returns:
        --------
        WinsorizedMinMaxScaler
            Instância do scaler carregada.
        """
        if is_filepath:
            with open(json_input, 'r') as f:
                data = json.load(f)
        else:
            data = json.loads(json_input)
        
        scaler = cls(p_lower=data['p_lower'], p_upper=data['p_upper'])
        scaler.min = data['min']
        scaler.max = data['max']
        scaler.lower = data['lower']
        scaler.upper = data['upper']
        scaler._n_samples = data.get('n_samples', 0)
        scaler._values_history = data.get('values_history', [])
        
        return scaler
