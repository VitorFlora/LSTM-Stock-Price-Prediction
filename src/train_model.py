
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta avisos de CPU/GPU do C++
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Oculta o aviso de otimização oneDNN
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import math

def construir_e_treinar_modelo(X_train, Y_train, X_test, Y_test, caminho_salvar_modelo):
    print("\nConstruindo a arquitetura do modelo Stacked LSTM...")
    
    model = Sequential()

    model.add(Input(shape=(X_train.shape[1], 1)))
    
    # 1ª Camada
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # 2ª Camada
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Saída
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # CONFIGURANDO O EARLY STOPPING
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Iniciando o treinamento com Early Stopping (Até 100 épocas)...")
    
    model.fit(X_train, Y_train, 
              batch_size=32, 
              epochs=100, 
              validation_data=(X_test, Y_test),
              callbacks=[early_stop]) 
    
    print("\n Treinamento concluído! Avaliando o modelo...")
    previsoes = model.predict(X_test)
    
    mae = mean_absolute_error(Y_test, previsoes)
    rmse = math.sqrt(mean_squared_error(Y_test, previsoes))
    mape = mean_absolute_percentage_error(Y_test, previsoes)
    
    print("\n=== Resultados da Avaliação (Stacked LSTM) ===")
    print(f"MAE (Erro Absoluto Médio): {mae:.4f}")
    print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
    print(f"MAPE (Erro Percentual Absoluto Médio): {mape:.4f}")
    
    model.save(caminho_salvar_modelo)
    print(f"\nSucesso! Modelo salvo em: {caminho_salvar_modelo}")
    
    return model

if __name__ == "__main__":
    print("=== Treinamento da Rede Neural LSTM ===")
    
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    raiz_projeto = os.path.dirname(diretorio_atual)
    
    caminho_X = os.path.join(raiz_projeto, 'data', 'processed', 'X_dados.npy')
    caminho_Y = os.path.join(raiz_projeto, 'data', 'processed', 'Y_dados.npy')
    caminho_modelo = os.path.join(raiz_projeto, 'models', 'modelo_lstm.keras') 
    
    if not os.path.exists(caminho_X) or not os.path.exists(caminho_Y):
        print("Erro: Arquivos de dados processados não encontrados. Rode o pre_processing.py primeiro.")
    else:
        X = np.load(caminho_X)
        Y = np.load(caminho_Y)
        
        tamanho_treino = int(len(X) * 0.8)
        
        X_train, X_test = X[:tamanho_treino], X[tamanho_treino:]
        Y_train, Y_test = Y[:tamanho_treino], Y[tamanho_treino:]
        
        print(f"Dados divididos: {len(X_train)} amostras para treino e {len(X_test)} para teste.")
        
        construir_e_treinar_modelo(X_train, Y_train, X_test, Y_test, caminho_modelo)