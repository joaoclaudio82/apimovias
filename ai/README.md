# ChemBERT: Guia de Uso do Projeto

Este documento apresenta um guia prático para utilização do projeto ChemBERT, que realiza predição de propriedades químicas a partir de representações SMILES utilizando modelos baseados em BERT com PyTorch Lightning.

---

## 1. Preparação dos Dados

- **Carregamento:** Utilize o `DipprDatasetLoader` para carregar os dados químicos, obtendo conjuntos para os modos **pure** (entrada única) e **mix** (entrada dupla).

- **Divisão:** Separe os dados em conjuntos de treino, validação (dev) e teste, garantindo que os dados estejam organizados conforme o modo escolhido.

- **Normalização:** Aplique normalização (ex: `StandardScaler`) especialmente nas variáveis contínuas como temperatura para melhorar a performance do modelo.

---

## 2. Configuração do Tokenizador e DataModule

- **Tokenizador:** Carregue o tokenizador compatível com o modelo base BERT que será utilizado (ex: através do Hugging Face Transformers).

- **DataModule:** Instancie o `ChemBERTDataModule` passando o tokenizador, os dados preparados e parâmetros como batch size e comprimento máximo de sequência (`max_length`).

- O DataModule gerenciará os datasets e dataloaders para os estágios de treino, validação e teste.

---

## 3. Inicialização do Modelo

- Escolha o modo de operação:  
  - `pure` para modelos que recebem um único SMILES e temperatura;  
  - `mix` para modelos que recebem dois SMILES, fração molar e temperatura.

- Configure se deseja usar LoRA (Low-Rank Adaptation) para uma adaptação eficiente do modelo base, ou treinar apenas a cabeça MLP com o modelo base congelado.

- Instancie o `ChemBERTModel` com o modelo base, modo, hiperparâmetros de aprendizado (learning rates, dropout), e dimensão oculta da cabeça MLP.

---

## 4. Treinamento

- Defina callbacks importantes:  
  - `ModelCheckpoint` para salvar o melhor modelo segundo métrica de validação (ex: $R^2$);  
  - `EarlyStopping` para cessar treinamento ao não observar melhora por um número definido de épocas;  
  - `BestModelExporter` Callback customizado para exportação do melhor modelo e tokenizador.

- Configure o trainer do PyTorch Lightning para utilizar GPU se disponível, número máximo de épocas, callbacks, etc.

- Execute o treinamento com o método `fit` passando o modelo e o datamodule.

---

## 5. Avaliação e Teste

- Após treinamento, utilize os métodos de validação e teste do trainer para avaliar desempenho.

- Métrica principal utilizada é o coeficiente de determinação ($R^2$), acompanhado da perda MSE.

---

## 6. Inferência / Predição

- Utilize a classe `ChemBERTPredictor` para carregar o modelo treinado e realizar predições.

- Forneça os inputs adequados conforme o modo:  
  - Para `pure`: lista de SMILES e temperaturas.  
  - Para `mix`: listas de dois SMILES, frações molares, e temperaturas.

- A saída será um array numpy com os valores previstos pelo modelo.

--- 


## 8. Considerações Adicionais

- Ajuste os hiperparâmetros (learning rate, dropout, dimensão oculta) conforme a complexidade e tamanho do seu dataset.

- Para cenários com poucos dados, recomenda-se o uso do LoRA para melhorar eficiência de treinamento.

- Certifique-se de que os dados de entrada estejam devidamente pré-processados e normalizados de forma consistente entre treino e inferência.

- Utilize a classe `ChemFeaturizer` para extrair features moleculares com base na biblioteca rdkit, total integração com o dataset loadet 

- Notebooks com exemplos de treinamento e otimização de hiperparâmetros estão disponíveis na pasta notebook.

- A pasta `api` contém uma REST-API para executar experimentos, tanto em tempo real quanto em batch. A API implementa autenticação e gestão de cadastro de usuários (executar pelo comando `task run_api`, dentro da pasta raiz do projeto). 


- A pasta `streamlit_app` contém uma aplicação de exemplo que consome a API implementada (executar pelo comando `task run_app`, dentro da pasta `streamlit_app`). Para a aplicação funcionar, a api deverá ser inicializada primeiro. Usuário padrão: (user: admin, password: admin123)

- Veja o arquivo pyproject.toml, na seção [tool.taskipy.tasks], para ver como utilizar a ferramenta task para diferentes tarefas. 

- Notebook `análise exploratória.ipynb` mostra uma análise exploratória para as features extraídas.
---


## Contato e Suporte

Em caso de dúvidas ou para colaborar no desenvolvimento, entre em contato com <livio.freire@aluno.uece.br>

