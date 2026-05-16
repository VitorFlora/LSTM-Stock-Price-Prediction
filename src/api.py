from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

diretorio_atual = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_modelo = os.path.join(diretorio_atual, 'models', 'modelo_lstm.keras')
caminho_scaler = os.path.join(diretorio_atual, 'models', 'scaler.pkl')

modelo = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo, scaler
    try:
        if not os.path.exists(caminho_modelo):
            mensagem_erro = f"Arquivo NÃO encontrado no Docker em: {caminho_modelo}"
        elif not os.path.exists(caminho_scaler):
            mensagem_erro = f"Arquivo NÃO encontrado no Docker em: {caminho_scaler}"
        else:
            modelo = load_model(caminho_modelo, compile=False)
            scaler = joblib.load(caminho_scaler)
    except Exception as e:
        mensagem_erro = f"Erro interno do TensorFlow: {str(e)}"
        
    print("-------------------------------------\n")
    
    yield

    modelo = None
    scaler = None
    print("API desligada. Memória e recursos liberados.")

app = FastAPI(
    title="API de Previsão de Ações (LSTM)",
    description="API RESTful para prever preços de ações usando um modelo Deep Learning LSTM.",
    version="1.0.0",
    lifespan=lifespan
)

class DadosEntrada(BaseModel):
    precos_historicos: list[float]

@app.post("/prever")
def prever_preco(dados: DadosEntrada):
    """Recebe um histórico de preços e retorna a previsão do próximo preço."""
    if scaler is None or modelo is None:
        raise HTTPException(status_code=503, detail="Serviço indisponível: O Modelo ou Scaler não carregou.")

    if len(dados.precos_historicos) != 60:
        raise HTTPException(status_code=400, detail="Você deve fornecer exatamente 60 preços históricos.")

    try:
        entrada_numpy = np.array(dados.precos_historicos).reshape(-1, 1)
        entrada_escalonada = scaler.transform(entrada_numpy)
        entrada_lstm = np.reshape(entrada_escalonada, (1, 60, 1))
        
        previsao_escalonada = modelo.predict(entrada_lstm)
        previsao_real = scaler.inverse_transform(previsao_escalonada)
        preco_futuro = float(previsao_real[0][0])
        
        return {"preco_previsto": round(preco_futuro, 2)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/health")
def health_check():
    if modelo is None or scaler is None:
        return {"status": "Erro ao carregar", "motivo": ""}
    return {"status": "A API está rodando perfeitamente com o modelo carregado!"}