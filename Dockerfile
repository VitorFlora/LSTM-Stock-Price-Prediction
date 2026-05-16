# 1. Usa uma imagem oficial do Python, versão mais leve (slim)
FROM python:3.13-slim

# 2. Define a pasta de trabalho dentro do contêiner
WORKDIR /app

# 3. Copia apenas o arquivo de requisitos primeiro (ajuda na velocidade do Docker)
COPY requirements.txt .

# 4. Atualiza o pip para garantir que as dependências sejam instaladas corretamente
RUN pip install --upgrade pip

# 5. Instala as dependências (a tag --no-cache-dir deixa a imagem menor)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copia as pastas essenciais do seu projeto para dentro do contêiner
COPY src/ /app/src/
COPY models/ /app/models/

# 7. Informa que o contêiner vai usar a porta 8000
EXPOSE 8000

# 8. O comando que o Docker vai executar quando o contêiner ligar
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]