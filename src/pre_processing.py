import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

def preparar_dados(caminho_csv, dias_historico=60):
    print(f"\nIniciando pré-processamento do arquivo: {caminho_csv}")

    df = pd.read_csv(caminho_csv)
    df['Close'] = df['Close'].astype(str).str.replace(',', '.')    
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    
    dias_1_ano = 252
    dias_minimos = 504  # 2 anos
    limite_volatilidade = 0.50
    
    print("\n Iniciando filtro de volatilidade...")
    anos_estaveis = 0
    
    for i in range(1, 11):
        dias_analise = i * dias_1_ano
        
        if len(df) < dias_analise:
            anos_estaveis = i - 1
            break

        inicio_fatia = len(df) - (i * dias_1_ano)
        fim_fatia = len(df) - ((i - 1) * dias_1_ano)
        
        # Foi usado um caso especial quando fim_fatia == len(df) para garantir que pegamos até o final do DataFrame,
        # pois o slicing iloc[inicio:fim] é exclusivo do índice final. Assim, iloc[inicio_fatia:] pega até o último registro,
        # enquanto iloc[inicio_fatia:fim_fatia] pega até fim_fatia-1.
        if fim_fatia == len(df):
            fatia_ano = df['Close'].iloc[inicio_fatia:]
        else:
            fatia_ano = df['Close'].iloc[inicio_fatia:fim_fatia]
            
        preco_max = fatia_ano.max()
        preco_min = fatia_ano.min()
        if preco_min == 0:
            print(f"  Atenção: preço mínimo zero encontrado no ano {i}. Flutuação definida como zero para evitar divisão por zero.")
            flutuacao = 0
        else:
            flutuacao = (preco_max - preco_min) / preco_min
        
        if flutuacao > limite_volatilidade:
            print(f"Alta volatilidade encontrada há {i} ano(s) (Volatilidade: {flutuacao*100:.2f}%).")
            print(f"Varredura interrompida. Descartando histórico anterior a este evento.")
            break
        else:
            print(f"Ano {i} retrospectivo estável (Volatilidade: {flutuacao*100:.2f}% de 50%).")
            anos_estaveis = i
            
    dias_para_manter = max(anos_estaveis * dias_1_ano, dias_minimos)
    
    if dias_para_manter < len(df):
        print(f"\nCORTANDO A BASE: O modelo treinará apenas com os últimos {dias_para_manter} dias úteis.")
        df = df.tail(dias_para_manter).reset_index(drop=True)
    else:
        print(f"\nHISTÓRICO LIMPO: Utilizando toda a base de dados disponível.")
    
    dados_fechamento = df.filter(['Close']).values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_escalonados = scaler.fit_transform(dados_fechamento)
    
    X, Y = [], []
    
    if len(dados_escalonados) <= dias_historico:
        raise ValueError("A base de dados ficou menor que a janela histórica. Não é possível treinar.")
    
    for i in range(dias_historico, len(dados_escalonados)):
        X.append(dados_escalonados[i-dias_historico:i, 0])
        Y.append(dados_escalonados[i, 0])
        
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    print(f"\nFormato final dos dados de entrada (X): {X.shape}")
    print(f"Formato final dos dados alvo (Y): {Y.shape}")
    
    return X, Y, scaler

if __name__ == "__main__":
    print("=== Pré-processamento de Dados para LSTM ===")
    
    nome_arquivo = input("Digite o nome do arquivo bruto (ex: BTLG11_raw.csv) ou aperte Enter para o padrão 'KO_raw.csv': ")
    if not nome_arquivo:
        nome_arquivo = 'KO_raw.csv'
        
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    raiz_projeto = os.path.dirname(diretorio_atual)
    
    caminho_entrada = os.path.join(raiz_projeto, 'data', 'raw', nome_arquivo)
    
    if not os.path.exists(caminho_entrada):
        print(f"Erro: O arquivo {caminho_entrada} não foi encontrado.")
    else:
        X, Y, scaler = preparar_dados(caminho_entrada, dias_historico=60)
        
        caminho_X = os.path.join(raiz_projeto, 'data', 'processed', 'X_dados.npy')
        caminho_Y = os.path.join(raiz_projeto, 'data', 'processed', 'Y_dados.npy')
        caminho_scaler = os.path.join(raiz_projeto, 'models', 'scaler.pkl')
        
        os.makedirs(os.path.dirname(caminho_X), exist_ok=True)
        os.makedirs(os.path.dirname(caminho_scaler), exist_ok=True)
        
        np.save(caminho_X, X)
        np.save(caminho_Y, Y)
        joblib.dump(scaler, caminho_scaler)
        
        print("\nArquivos processados e salvos com sucesso! \n")