import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import glob

print("=== Validação Local do Modelo LSTM ===")

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
raiz_projeto = os.path.dirname(diretorio_atual)

pasta_raw = os.path.join(raiz_projeto, 'data', 'raw')
arquivos_csv = glob.glob(os.path.join(pasta_raw, '*.csv'))

if not arquivos_csv:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado na pasta data/raw.")

caminho_csv = max(arquivos_csv, key=os.path.getmtime)
nome_arquivo = os.path.basename(caminho_csv)

print(f"Arquivo mais recente detectado automaticamente: {nome_arquivo}")

caminho_modelo = os.path.join(raiz_projeto, 'models', 'modelo_lstm.keras')
caminho_scaler = os.path.join(raiz_projeto, 'models', 'scaler.pkl')

modelo = load_model(caminho_modelo, compile=False) 
scaler = joblib.load(caminho_scaler)

df = pd.read_csv(caminho_csv)
df['Close'] = df['Close'].astype(str).str.replace(',', '.')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

df['Close'].tail(60).round(2)
ultimos_60_dias = df['Close'].tail(60).values

entrada_numpy = np.array(ultimos_60_dias).reshape(-1, 1)
entrada_escalonada = scaler.transform(entrada_numpy)
entrada_lstm = np.reshape(entrada_escalonada, (1, 60, 1))

previsao_escalonada = modelo.predict(entrada_lstm)
previsao_real = scaler.inverse_transform(previsao_escalonada)
preco_futuro = float(previsao_real[0][0])

print(f"\n Previsão para o próximo dia de fechamento: $ {preco_futuro:.2f}")