# apimovias

API em **FastAPI** para gerar e baixar um **CSV** (em `data/movias.csv`) com métricas/features diárias por veículo a partir de dados no **PostgreSQL**.

## Pré-requisitos

- Python 3.10+ e `pip`
- Docker + Docker Compose
- (Opcional) `make` disponível no terminal

## Estrutura completa do projeto

- **`api/`**: código da aplicação.
  - **`main.py`**: cria o `FastAPI` e registra as rotas.
  - **`routes/`**: endpoints HTTP (hoje: `relatorio_route.py`).
  - **`services/`**: regras de negócio + processamento com Pandas/Numpy (gera/append no CSV).
  - **`repositories/`**: acesso ao banco (query SQL via SQLAlchemy).
  - **`schemas/`**: modelos Pydantic dos payloads.
  - **`database.py`**: cria o `engine` do SQLAlchemy (Postgres).
  - **`settings.py`**: lê config do `.env` e define paths (ex.: `data/movias.csv`).
  - **`utils/`**: helpers simples (ex.: verificação de arquivo vazio).
- **`data/sql/mock.sql`**: schema/tabelas e dados mock usados pelo Postgres do compose.
- **`docs/`**: materiais auxiliares (notebook e script de extração).
- **`docker-compose.dev.yml`**: sobe um Postgres local e aplica o `mock.sql` no init.
- **`Makefile`**: atalho para rodar a API (`make api`).
- **`requirements.txt`**: dependências Python.

## Portas expostas

- **API FastAPI**: `8000` (padrão do `uvicorn`)
- **PostgreSQL (Docker)**: `5432`

## Fluxo (request → banco → CSV)

1. Você chama um endpoint em `/relatorios`.
2. A rota chama `RelatorioService`.
3. O service busca dados no banco via `RelatorioRepository` (query em `trips.alltrips` + `public.veiculo`).
4. O service:
   - normaliza datas e métricas (km/h do dia),
   - aplica tratamento de outliers,
   - cria features por janela (7/14/21/28 dias) e flags (fim de semana, parado etc.),
   - escreve/append no CSV `data/movias.csv` evitando duplicar por chave `(veiculo_id, data)`.
5. Você pode baixar o CSV pelo endpoint de download.

## Endpoints

- **GET** `/relatorios/download`
  - Retorna o arquivo `data/movias.csv` (404 se não existir/estiver vazio).
- **POST** `/relatorios/`
  - Body: `{ "veiculo_id": 1, "data_ini": "2025-01-01", "data_fim": "2025-01-31" }`
  - Gera/append no CSV para **um** veículo (na implementação atual, `id_start=id_end=veiculo_id`).
- **POST** `/relatorios/batch`
  - Body: `{ "id_start": 1, "id_end": 10, "data_ini": "2025-01-01", "data_fim": "2025-01-31" }`
  - Gera/append no CSV para um range de veículos.

## Como rodar do zero

### 1) Subir o Postgres (com dados mock)

```bash
docker compose -f docker-compose.dev.yml up -d
```

Isso cria um Postgres local em `localhost:5432` e executa `data/sql/mock.sql` no init.

### 2) Criar `.env`

Crie um arquivo `.env` na raiz (mesmo nível de `requirements.txt`) com:

```env
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_DATABASE=movias
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres
DATABASE_SSL_MODE=disable
```

### 3) Instalar dependências e rodar a API

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
make api
```

A API sobe em modo reload via `uvicorn`. Você pode acessar:

- `http://localhost:8000/docs` (Swagger)
- `http://localhost:8000/redoc` (ReDoc)

Se não tiver `make`, rode diretamente:

```bash
uvicorn api.main:app --reload
```

