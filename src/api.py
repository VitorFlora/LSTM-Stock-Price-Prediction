from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import joblib
import pandas as pd
import os
import logging
from datetime import datetime
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlops_logger")

diretorio_atual = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_modelo = os.path.join(diretorio_atual, 'models', 'modelo_lstm.keras')
caminho_scaler = os.path.join(diretorio_atual, 'models', 'scaler.pkl')
caminho_telemetria = os.path.join(diretorio_atual, 'data', 'telemetry.csv')

modelo = None
scaler = None
mensagem_erro = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo, scaler, mensagem_erro
    try:
        if not os.path.exists(caminho_modelo):
            mensagem_erro = f"Arquivo NÃO encontrado no Docker em: {caminho_modelo}"
            logger.error(f"{mensagem_erro}")
        elif not os.path.exists(caminho_scaler):
            mensagem_erro = f"Arquivo NÃO encontrado no Docker em: {caminho_scaler}"
            logger.error(f"{mensagem_erro}")
        else:
            modelo = load_model(caminho_modelo, compile=False)
            scaler = joblib.load(caminho_scaler)
            logger.info("Modelo LSTM e Scaler carregados com sucesso via Lifespan.")
    except Exception as e:
        mensagem_erro = f"Erro interno do TensorFlow: {str(e)}"
        logger.critical(f"{mensagem_erro}")
        
    print("-------------------------------------\n")
    
    yield

    modelo = None
    scaler = None
    print("API desligada. Memória e recursos liberados.")


app = FastAPI(
    title="API de Previsão de Ações (LSTM)",
    description="API RESTful para prever preços de ações usando um modelo Deep Learning LSTM com monitoramento de MLOps integrado.",
    version="1.0.0",
    lifespan=lifespan
)

class DadosEntrada(BaseModel):
    precos_historicos: list[float]

class DadosFeedback(BaseModel):
    preco_real_fechamento: float

@app.post("/prever")
def prever_preco(dados: DadosEntrada):
    """Recebe um histórico de preços, avalia Data Drift, salva telemetria e retorna a previsão."""
    if scaler is None or modelo is None:
        raise HTTPException(status_code=503, detail=f"Serviço indisponível: O Modelo ou Scaler não carregou. Motivo: {mensagem_erro}")

    if len(dados.precos_historicos) != 60:
        raise HTTPException(status_code=400, detail="Você deve fornecer exatamente 60 preços históricos.")

    try:
        # DETECÇÃO DE DATA DRIFT
        # Se o preço atual desviar mais de 4 desvios padrão da média móvel dos 60 dias informados,
        # significa que a volatilidade mudou drasticamente perto do fechamento.
        media_historica = np.mean(dados.precos_historicos)
        desvio_padrao = np.std(dados.precos_historicos)
        preco_atual = dados.precos_historicos[-1]
        
        if desvio_padrao > 0 and abs(preco_atual - media_historica) / desvio_padrao > 4.0:
            logger.warning(f"[DATA DRIFT DETECTED] O preço atual (R${preco_atual:.2f}) divergiu agressivamente da média recente do ativo. Risco de perda de acurácia.")

        entrada_numpy = np.array(dados.precos_historicos).reshape(-1, 1)
        entrada_escalonada = scaler.transform(entrada_numpy)
        entrada_lstm = np.reshape(entrada_escalonada, (1, 60, 1))
        
        previsao_escalonada = modelo.predict(entrada_lstm)
        previsao_real = scaler.inverse_transform(previsao_escalonada)
        preco_futuro = round(float(previsao_real[0][0]), 2)
        
        try:
            os.makedirs(os.path.dirname(caminho_telemetria), exist_ok=True)
            if not os.path.exists(caminho_telemetria):
                df_init = pd.DataFrame(columns=["timestamp", "ultimo_preco", "previsao"])
                df_init.to_csv(caminho_telemetria, index=False)
                
            novo_registro = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ultimo_preco": preco_atual,
                "previsao": preco_futuro
            }])
            novo_registro.to_csv(caminho_telemetria, mode='a', header=False, index=False)
            logger.info("[TELEMETRIA] Entrada e previsão registradas com sucesso no histórico de produção.")
        except Exception as tel_err:
            logger.error(f"Erro ao registrar dados de telemetria: {str(tel_err)}")
        
        return {"preco_previsto": preco_futuro}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento: {str(e)}")


@app.post("/feedback")
def feedback_loop(dados_fb: DadosFeedback):
    """
    Simula a chegada do preço real do dia seguinte para calcular o erro real 
    do modelo em produção e disparar gatilhos de degradação.
    """
    if not os.path.exists(caminho_telemetria):
        raise HTTPException(status_code=404, detail="Nenhuma previsão registrada no banco de telemetria para comparação.")
        
    try:
        df = pd.read_csv(caminho_telemetria)
        if df.empty:
            raise HTTPException(status_code=400, detail="O histórico de telemetria está vazio.")
            
        ultima_previsao = df["previsao"].iloc[-1]
        preco_real = dados_fb.preco_real_fechamento
        
        # Cálculo do erro percentual médio absoluto da última inferência
        mape = (abs(preco_real - ultima_previsao) / preco_real) * 100
        logger.info(f"[FEEDBACK LOOP] Avaliação: Previsão R${ultima_previsao:.2f} vs Valor Real R${preco_real:.2f} - MAPE: {mape:.2f}%")
        
        # GATILHO DE DEGRADAÇÃO DO MODELO
        if mape > 15.0:
            logger.error(f"[GATILHO DE DEGRADAÇÃO ACIONADO] O erro do modelo atingiu {mape:.2f}%, estourando o limite aceitável de 15%. Retreinamento imediato recomendado!")
            return {
                "status": "Alerta: Performance Degradada",
                "mape_detectado": round(mape, 2),
                "recomendacao": "Acionar a esteira de Continuous Training (pipeline.py)"
            }
            
        return {
            "status": "Modelo Operando Dentro dos Parâmetros",
            "mape_detectado": round(mape, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar loop de feedback: {str(e)}")


@app.get("/health")
def health_check():
    if modelo is None or scaler is None:
        return {"status": "Erro ao carregar", "motivo": mensagem_erro}
    return {"status": "A API está rodando perfeitamente com o modelo carregado!"}