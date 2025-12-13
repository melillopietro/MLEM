# Usa un'immagine Python leggera
FROM python:3.10-slim

# Imposta la cartella di lavoro
WORKDIR /app

# Copia i file necessari
COPY requirements.txt .
COPY . .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Esponi la porta di Streamlit
EXPOSE 8501

# Comando di avvio
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]