# Movias Stack (Root)

Stack unificado com:

- `extractor` (FastAPI): gera/atualiza `extractor/data/movias.csv`.
- `ai` (FastAPI): atualiza perfis e executa predição.
- `gateway` (FastAPI): expõe rota única e Swagger unificado dos dois serviços.
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

- Gateway (entrada única): `http://localhost:8080`
- Swagger unificado: `http://localhost:8080/docs`
- Apenas o `gateway` é exposto no host.
- `ai` e `extractor` ficam acessíveis somente na rede interna do compose.

## Uso via gateway

- Rotas da AI API ficam em `http://localhost:8080/ai/*`
  Ex.: `POST http://localhost:8080/ai/predictions/date-to-reach`
- Rotas da Extractor API ficam em `http://localhost:8080/extractor/*`
  Ex.: `POST http://localhost:8080/extractor/relatorios/`

## Fluxo integrado

1. `POST /extractor/relatorios/` ou `POST /extractor/relatorios/batch` no gateway.
2. O extractor atualiza o CSV local.
3. O `ai` monitora o CSV compartilhado automaticamente.
4. Quando o arquivo muda, o `ai` sincroniza os perfis nos targets:
   - `km_dia_clean`
   - `h_dia_clean`
5. O `ai` resolve automaticamente o arquivo:
   - Docker: `/shared/movias.csv`
   - Local: `extractor/data/movias.csv`

## Comandos úteis

```bash
make ai
make extractor
make gateway
make up
```
