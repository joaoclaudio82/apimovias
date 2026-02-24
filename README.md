# Movias Stack (Root)

Stack unificado com:

- `extractor-api` (FastAPI): gera/atualiza `extractor/data/movias.csv`.
- `ai-api` (FastAPI): atualiza perfis e executa predição.
- Banco do `extractor`: externo (já existente), configurado via `DATABASE_*` no `.env` da raiz.
- Volume compartilhado do CSV: `./extractor/data` montado em `/shared` nos dois serviços no Docker.

## Subir tudo

1. Copie o arquivo de ambiente:

```bash
cp .env.example .env
```

2. Suba o stack:

```bash
docker compose up -d --build
```

Antes de subir, ajuste no `.env` as variáveis do banco externo do extractor:

- `DATABASE_HOST`
- `DATABASE_PORT`
- `DATABASE_DATABASE`
- `DATABASE_USER`
- `DATABASE_PASSWORD`
- `DATABASE_SSL_MODE`

## Portas padrão

- Extractor API: `http://localhost:8000`
- AI API: `http://localhost:8010`

## Fluxo integrado

1. `POST /relatorios/` ou `POST /relatorios/batch` no `extractor-api`.
2. O extractor atualiza o CSV local.
3. O `ai-api` monitora o CSV compartilhado automaticamente.
4. Quando o arquivo muda, o `ai-api` sincroniza os perfis nos targets:
   - `km_dia_clean`
   - `h_dia_clean`
5. O `ai-api` resolve automaticamente o arquivo:
   - Docker: `/shared/movias.csv`
   - Local: `extractor/data/movias.csv`

## Comandos úteis

```bash
make ai
make extractor
make up
```
