FROM python:3.11-slim

WORKDIR /app

# System dependencies for scientific Python
RUN apt-get update && apt-get install -y \
    gcc g++ libffi-dev libssl-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models results

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
