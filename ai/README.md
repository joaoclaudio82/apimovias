# Movias AI - API de Predição de Manutenção

API REST para gerenciamento de perfis de veículos e predições de manutenção baseadas em Machine Learning.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Endpoints da API](#endpoints-da-api)
- [Schemas de Dados](#schemas-de-dados)
- [Desenvolvimento](#desenvolvimento)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

A **Movias AI API** é uma solução completa para análise preditiva de manutenção de veículos, utilizando séries temporais e modelos de Machine Learning. O sistema processa dados históricos de quilometragem e horas de operação para:

- **Classificar veículos** em categorias (KM/H) e segmentos
- **Extrair features** comportamentais de cada veículo
- **Predizer datas** de manutenção baseadas em valores acumulados
- **Estimar valores acumulados** em datas futuras

### Características principais:

✅ **API RESTful** com FastAPI  
✅ **Processamento assíncrono** com SQLAlchemy async  
✅ **Modelos ML** em PyTorch e Scikit-learn  
✅ **Inferência ONNX** para produção  
✅ **Pipeline completo** de ingestão, treinamento e predição  
✅ **Validação automática** de dados com Pydantic  

---

## 🚀 Funcionalidades

### 1. **Perfis de Veículos** (`/vehicle-profiles`)

- Importação de dados históricos
- Classificação automática (KM vs H)
- Segmentação por comportamento
- Extração de 85+ features por veículo
- Atualização incremental de perfis

### 2. **Predições** (`/predictions`)

- **Date-to-reach**: Quando um veículo atingirá determinado valor acumulado
- **Accumulated-at-step**: Quanto será acumulado em uma data futura
- Predições em lote com paralelização
- Cache de modelos para performance

### 3. **Ingestão de Dados** (`/data-ingestion`)

- Pipeline automatizado de processamento
- Formatação e validação de datasets
- Geração de janelas temporais
- Limpeza de arquivos intermediários

### 4. **Treinamento** (`/training`)

- Treinamento de modelos PyTorch (DLinear, NLinear)
- Treinamento de modelos globais (LightGBM, XGBoost)
- Execução em background
- Validação e métricas automáticas

---

## 🏗️ Arquitetura
