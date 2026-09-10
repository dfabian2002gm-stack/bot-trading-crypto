import os
import time
import threading
import ccxt
import xgboost as xgb
import numpy as np
import pandas as pd
import requests
from flask import Flask

# Configuración inicial de Flask para mantener vivo el servicio web en Render (Gunicorn busca 'app')
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot de Trading XGBoost en funcionamiento (Paper Trading con PnL)", 200

def run_flask():
    app.run(host='0.0.0.0', port=10000)

# Credenciales de Telegram desde las variables de entorno de Render
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram no configurado. Mensaje omitido.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")

# Conexión a Kraken (Exchange público para datos y simulación)
exchange = ccxt.kraken()

# Variables de simulación (Paper Trading y PnL)
paper_balance_usdt = 1000.0  # Capital inicial virtual
paper_btc_held = 0.0
position_status = "FLAT"       # FLAT (sin posición) o LONG (comprado)
entry_price = 0.0

def fetch_data():
    try:
        # Descarga datos de velas de 1 hora para BTC/USDT desde Kraken
        bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Indicadores técnicos básicos para XGBoost
        df['returns'] = df['close'].pct_change()
        df['sma'] = df['close'].rolling(window=10).mean()
        df['volatility'] = df['returns'].rolling(window=10).std()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error obteniendo datos de Kraken: {e}")
        return None

def train_and_predict(df):
    try:
        features = ['returns', 'sma', 'volatility', 'volume']
        # Definir objetivo: 1 si el precio sube en la siguiente vela, 0 si baja
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        X = df[features].iloc[:-1]
        y = df['target'].iloc[:-1]
        
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(X, y)
        
        # Predecir sobre la última vela disponible
        latest_features = df[features].iloc[[-1]]
        prediction = model.predict(latest_features)[0]
        return prediction
    except Exception as e:
        print(f"Error entrenando modelo XGBoost: {e}")
        return None

def bot_loop():
    global paper_balance_usdt, paper_btc_held, position_status, entry_price
    
    print("Iniciando bucle de trading autónomo (Cada 30 minutos)...")
    send_telegram_message("🤖 *Bot de Trading Iniciado*\nModo Paper Trading activado con reportes cada 30 minutos.")
    
    while True:
        try:
            print("Ejecutando ciclo de análisis...")
            df = fetch_data()
            if df is not None and not df.empty:
                current_price = df['close'].iloc[-1]
                prediction = train_and_predict(df)
                
                print(f"Precio actual BTC: ${current_price:.2f} | Predicción XGBoost: {prediction}")
                
                # Reporte obligatorio en cada ciclo para confirmar que está vivo
                status_msg = (f"📊 *Reporte Periódico (30 min)*\n"
                              f"• Precio BTC: `${current_price:.2f}`\n"
                              f"• Predicción: `{prediction}`\n"
                              f"• Estado: `{position_status}`\n"
                              f"• Saldo Virtual: `${paper_balance_usdt:.2f}`")
                send_telegram_message(status_msg)
                
                # Lógica de Paper Trading con PnL
                if prediction == 1 and position_status == "FLAT":
                    paper_btc_held = (paper_balance_usdt * 0.99) / current_price 
                    entry_price = current_price
                    paper_balance_usdt = 0.0
                    position_status = "LONG"
                    
                    msg = (f"🟢 *SIMULACIÓN DE COMPRA (LONG)*\n"
                           f"• Precio de Entrada: `${entry_price:.2f}`\n"
                           f"• BTC Adquirido: `{paper_btc_held:.5f}`")
                    send_telegram_message(msg)
                    
                elif prediction == 0 and position_status == "LONG":
                    sale_proceeds = paper_btc_held * current_price * 0.99 
                    pnl_usd = sale_proceeds - (paper_btc_held * entry_price)
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    
                    paper_balance_usdt = sale_proceeds
                    paper_btc_held = 0.0
                    position_status = "FLAT"
                    
                    msg = (f"🔴 *SIMULACIÓN DE VENTA / CIERRE*\n"
                           f"• Precio de Salida: `${current_price:.2f}`\n"
                           f"• PnL de la Operación: `${pnl_usd:.2f}` (`{pnl_pct:+.2f}%`)\n"
                           f"• Saldo Virtual Total: `${paper_balance_usdt:.2f}`")
                    send_telegram_message(msg)
            
        except Exception as e:
            print(f"Error en el ciclo del bot: {e}")
            send_telegram_message(f"⚠️ *Error en el Bot*: {e}")
            
        # Esperar 30 minutos para el próximo ciclo
        time.sleep(1800)

# Iniciar el bucle del bot en un hilo en segundo plano al cargar el módulo en Render
bot_thread = threading.Thread(target=bot_loop, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    run_flask()
