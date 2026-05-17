import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def baixar_dados(ticker, data_inicio, data_fim, caminho_salvar):
    """
    Baixa os dados históricos de uma ação usando yfinance e salva em um arquivo CSV.
    """
    print(f"\nBaixando dados para {ticker} do período {data_inicio} até {data_fim}...")
    
    try:
        df = yf.download(ticker, start=data_inicio, end=data_fim, auto_adjust=True)
        
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"Aviso: Nenhum dado válido encontrado para o código '{ticker}'. Verifique se o símbolo está correto.")
            return False

        os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)

        df.to_csv(caminho_salvar)
        print(f"Sucesso! Dados salvos em: {caminho_salvar}")
        return True

    except Exception as e:
        print(f"Ocorreu um erro durante o download: {e}")
        return False

if __name__ == "__main__":
    print("=== Coletor de Dados Financeiros ===")
    
    acao_escolhida = input("Digite o código da empresa/ativo (ex: DIS, AAPL, BTLG11.SA) ou aperte Enter para o padrão (ex: KO): ").upper()
    if not acao_escolhida:
        acao_escolhida = 'KO'
        
    data_hoje = datetime.today().strftime('%Y-%m-%d')
    data_antiga = (datetime.today().replace(year=datetime.today().year - 10)).strftime('%Y-%m-%d')

    data_inicio = input(f"Digite a data de início (AAAA-MM-DD) ou aperte Enter para o padrão {data_antiga}: ")
    if not data_inicio:
        data_inicio = data_antiga

    data_fim = input(f"Digite a data de fim (AAAA-MM-DD) ou aperte Enter para hoje ({data_hoje}): ")
    if not data_fim:
        data_fim = data_hoje

    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    raiz_projeto = os.path.dirname(diretorio_atual)    
    nome_arquivo = f"{acao_escolhida.replace('.SA', '')}_raw.csv"
    caminho_arquivo = os.path.join(raiz_projeto, 'data', 'raw', nome_arquivo)

    baixar_dados(acao_escolhida, data_inicio, data_fim, caminho_arquivo)