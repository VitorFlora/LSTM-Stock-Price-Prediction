# Tech Challenge Fase 4 - API de Previsão de Ações com Deep Learning (LSTM)

Este repositório contém a entrega do Tech Challenge da Fase 4 da pós-graduação em Engenharia de Machine Learning. O projeto consiste em um pipeline completo de MLOps (Coleta, Processamento, Treinamento e Deploy) que utiliza uma rede neural recorrente (Stacked LSTM) para prever o preço de fechamento de ativos da bolsa de valores.

---

## 🚀 Arquitetura e Diferenciais do Projeto

- **Inteligência Artificial:** Arquitetura Stacked LSTM com camadas de `Dropout` para evitar Overfitting e `Early Stopping` para otimização do número de épocas de treinamento.

- **Varredura Retroativa Automática:** O algoritmo de pré-processamento avalia choques estruturais no mercado (como a instabilidade de 2020) e ajusta a janela histórica automaticamente baseado em picos de volatilidade.

- **Pipeline Orquestrado:** Criação de um MLOps automatizado (`pipeline.py`) que executa todas as fases do projeto sequencialmente.

- **Deploy Escalável:** API construída com FastAPI e totalmente conteinerizada com Docker utilizando Python 3.13.

```
LSTM-Stock-Price-Prediction/ 
├── data/
│   ├── processed/          # Guarda os arquivos X_dados.npy e Y_dados.npy
│   └── raw/                # Guarda os arquivos KO_raw.csv, TAEE11_raw.csv, etc.
├── models/
│   ├── modelo_lstm.keras   # Pesos da Rede Neural
│   └── scaler.pkl          # Objeto de normalização matemática
├── notebooks/
├── src/
│   ├── api.py              # Script FastAPI
│   ├── data_collection.py  # Coleta de dados via yfinance
│   ├── pipeline.py         # Orquestrador central do MLOps
│   ├── pre_processing.py   # Filtro de volatilidade e limpeza
│   ├── train_model.py      # Estrutura e treino da Stacked LSTM
│   └── validacao_local.py  # Script de teste de inferência local
├── .dockerignore
├── .gitignore
├── Dockerfile
├── gerar_teste.py      # Utilitário de payload para testar a API
└── requirements.txt
```

---

## 🛠️ Como Executar o Projeto via Docker

A aplicação foi projetada para rodar em qualquer ambiente utilizando contêineres, garantindo paridade total entre desenvolvimento e produção e evitando conflitos de versão do TensorFlow.

1. **Construa a Imagem Docker:**
Na raiz do projeto, execute o comando abaixo para construir a infraestrutura:

    `docker build --no-cache -t api_tech_challenge .`

2. **Inicie o Contêiner:**

    `docker run -p 8000:8000 --name tech_challenge_api api_tech_challenge`

###### Caso queira somente executar o container já criado, utilize `docker start tech_challenge_api`

A API estará online e a documentação interativa ficará disponível em: `http://127.0.0.1:8000/docs`

---

## 🧠 Como Testar a API (Passo a Passo)

O modelo Stacked LSTM exige um formato de dados muito específico para realizar a inferência: uma matriz contendo exatamente os últimos 60 dias de pregão reais do ativo. Como digitar 60 números financeiros manualmente é inviável, foi desenvolvido um utilitário automatizado de testes.

1. **Gerar o Payload de Teste:**

    Abra um novo terminal (mantendo a API rodando no Docker em segundo plano) e execute o script utilitário na raiz do projeto:

    `python gerar_teste.py`

    O que este script faz? Ele varre a pasta de dados brutos (`data/raw/`), identifica o último ativo que foi processado pelo pipeline, extrai o preço exato de fechamento dos últimos 60 dias e devolve a lista formatada e pronta para a API.

1. **Acessar o Painel da API:**

    Abra o seu navegador e acesse a interface do Swagger UI: `http://127.0.0.1:8000/docs`

2. **Realizar a Previsão:**

    - Expanda o endpoint `POST /prever`.
    - Clique no botão **"Try it out"** (Testar).
    - No campo `Request body`, apague o JSON padrão e cole a lista que foi gerada no terminal pelo script do Passo 1.
    - Clique no botão azul **"Execute"**.
    - Em instantes, a API passará seus dados pelo Scaler, fará a inferência na Rede Neural e devolverá, na seção "Server response", o valor exato da previsão para o próximo pregão.