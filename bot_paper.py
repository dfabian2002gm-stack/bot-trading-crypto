import os
import time
import datetime
import io
import requests
import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask

# --- CONFIGURACIÓN DE ENTORNO ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
BINANCE_API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
BINANCE_SECRET = os.environ.get("BINANCE_TESTNET_SECRET")

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot de Trading - Prueba de Telegram", 200

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("¡Faltan las credenciales de Telegram en las variables de entorno!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Respuesta de Telegram: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error crítico en Telegram msg: {e}")

def background_loop():
    import threading
    def worker():
        time.sleep(5) # Esperar que levante el servidor
        print("Enviando mensaje de prueba a Telegram...")
        send_telegram_message("🚀 *¡Hola Danny! El bot de trading se ha conectado correctamente a Render.*")
            
    t = threading.Thread(target=worker, daemon=True)
    t.start()

background_loop()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
