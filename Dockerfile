FROM mcr.microsoft.com/devcontainers/python:3.11

# ライブラリのインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# ポート8501を開放（Streamlit用）
EXPOSE 8501