import os
import glob
import pandas as pd

def obter_ultimos_60_dias_reais():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    pasta_raw = os.path.join(diretorio_atual, 'data', 'raw')
    
    arquivos_csv = glob.glob(os.path.join(pasta_raw, '*.csv'))
    
    if not arquivos_csv:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado na pasta data/raw.")
        
    caminho_csv = max(arquivos_csv, key=os.path.getmtime)
    nome_arquivo = os.path.basename(caminho_csv)
    
    print(f"Extraindo dados reais do arquivo mais recente: {nome_arquivo}")
    
    df = pd.read_csv(caminho_csv)
    df['Close'] = df['Close'].astype(str).str.replace(',', '.')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    ultimos_60_dias = df['Close'].tail(60).round(2).tolist()
    
    if len(ultimos_60_dias) < 60:
        raise ValueError(f"O arquivo {nome_arquivo} tem menos de 60 dias válidos.")
        
    return ultimos_60_dias

if __name__ == "__main__":
    print("=== GERADOR DE PAYLOAD PARA A API ===")
    try:
        lista_teste = obter_ultimos_60_dias_reais()
        
        print("\nPronto! Copie exatamente a lista abaixo (incluindo os colchetes [ ])")
        print("e cole no painel do FastAPI (http://127.0.0.1:8000/docs) para testar:\n")
        
        # Imprime a lista limpa no formato JSON Array
        print(lista_teste)
        
        print("\n=====================================")
    except Exception as e:
        print(f"\nErro ao gerar os dados: {e}")