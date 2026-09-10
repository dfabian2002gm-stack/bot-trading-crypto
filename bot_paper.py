import os
import time
import threading
import ccxt
import xgboost as xgb
import numpy as np
import pandas as pd
import requests
from flask import Flask

# Configuración inicial de Flask para Render (Gunicorn busca 'app')
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot de Trading Profesional XGBoost (Futuros / Paper Trading) activo y despierto", 200

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

# Conexión a Kraken (Exchange público)
exchange = ccxt.kraken()

# Variables de simulación de Futuros (Paper Trading)
paper_balance_usdt = 1000.0  # Capital inicial virtual
leverage = 2.0                 # Apalancamiento controlado (2x)
position_status = "FLAT"       # FLAT, LONG o SHORT
entry_price = 0.0
position_size = 0.0
take_profit_price = 0.0
stop_loss_price = 0.0

def fetch_data():
    try:
        # Descarga datos de 1 hora
        bars_1h = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=150)
        df_1h = pd.DataFrame(bars_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Descarga datos de 4 horas para el Filtro de Tendencia Macro
        bars_4h = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=50)
        df_4h = pd.DataFrame(bars_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Indicadores técnicos en 1h
        df_1h['returns'] = df_1h['close'].pct_change()
        df_1h['sma'] = df_1h['close'].rolling(window=14).mean()
        df_1h['volatility'] = df_1h['returns'].rolling(window=14).std()
        
        # Cálculo de RSI (Índice de Fuerza Relativa) para mayor rentabilidad
        delta = df_1h['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_1h['rsi'] = 100 - (100 / (1 + rs))
        
        # Cálculo de ATR (Average True Range) para Stop Loss / Take Profit dinámicos
        high_low = df_1h['high'] - df_1h['low']
        high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
        low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_1h['atr'] = true_range.rolling(window=14).mean()
        
        # Filtro Macro 4h
        df_1h['macro_trend'] = df_4h['close'].rolling(window=10).mean().iloc[-1]
        
        df_1h.dropna(inplace=True)
        return df_1h
    except Exception as e:
        print(f"Error obteniendo datos de Kraken: {e}")
        return None

def train_and_predict(df):
    try:
        features = ['returns', 'sma', 'volatility', 'volume', 'rsi', 'atr']
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        if len(df) < 20:
            return 0, 50.0
            
        X = df[features].iloc[:-1]
        y = df['target'].iloc[:-1]
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42)
        model.fit(X, y)
        
        latest_features = df[features].iloc[[-1]]
        prediction = int(model.predict(latest_features)[0])
        probabilities = model.predict_proba(latest_features)[0]
        confidence = float(probabilities[prediction] * 100)
        
        return prediction, confidence
    except Exception as e:
        print(f"Error entrenando modelo XGBoost: {e}")
        return 0, 50.0

def self_ping():
    """Función para hacer ping a sí mismo cada 5 minutos y evitar que Render duerma el bot"""
    while True:
        time.sleep(300) # Cada 5 minutos
        try:
            # Reemplaza esto con tu URL exacta de Render si deseas, o usa localhost
            requests.get("http://localhost:10000/")
        except:
            pass

def bot_loop():
    global paper_balance_usdt, position_status, entry_price, position_size, take_profit_price, stop_loss_price
    
    print("Iniciando bot de futuros avanzado con RSI y Keep-Alive...")
    send_telegram_message("🚀 *Bot de Futuros XGBoost Reiniciado*\n• RSI y optimización de rentabilidad activos.\n• Sistema Keep-Alive para evitar suspensiones.")
    
    while True:
        try:
            df = fetch_data()
            if df is not None and not df.empty:
                current_price = df['close'].iloc[-1]
                current_atr = df['atr'].iloc[-1]
                macro_sma = df['sma'].iloc[-1]
                
                prediction, confidence = train_and_predict(df)
                macro_filter = "Alcista 🟢" if current_price >= macro_sma else "Bajista 🔴"
                
                # Gestión de posición abierta (SL / TP)
                if position_status == "LONG":
                    if current_price >= take_profit_price or current_price <= stop_loss_price:
                        pnl_usd = (current_price - entry_price) * position_size
                        paper_balance_usdt += (position_size * entry_price) + pnl_usd
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
                        
                        msg = (f"🎯 *CIERRE LONG (SL/TP)*\n• Precio Salida: `${current_price:.2f}`\n• PnL: `${pnl_usd:+.2f}` (`{pnl_pct:+.2f}%`)\n• Saldo: `${paper_balance_usdt:.2f}`")
                        send_telegram_message(msg)
                        position_status = "FLAT"
                        
                elif position_status == "SHORT":
                    if current_price <= take_profit_price or current_price >= stop_loss_price:
                        pnl_usd = (entry_price - current_price) * position_size
                        paper_balance_usdt += (position_size * entry_price) + pnl_usd
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
                        
                        msg = (f"🎯 *CIERRE SHORT (SL/TP)*\n• Precio Salida: `${current_price:.2f}`\n• PnL: `${pnl_usd:+.2f}` (`{pnl_pct:+.2f}%`)\n• Saldo: `${paper_balance_usdt:.2f}`")
                        send_telegram_message(msg)
                        position_status = "FLAT"

                # Apertura de posiciones con umbral optimizado (52%)
                if position_status == "FLAT":
                    if prediction == 1 and macro_filter == "Alcista 🟢" and confidence >= 52.0:
                        entry_price = current_price
                        position_size = (paper_balance_usdt * leverage) / entry_price
                        take_profit_price = entry_price + (2.5 * current_atr)
                        stop_loss_price = entry_price - (1.5 * current_atr)
                        position_status = "LONG"
                        
                        send_telegram_message(f"📈 *LONG (2x)*\n• Entrada: `${entry_price:.2f}`\n• Confianza: `{confidence:.1f}%`")
                        
                    elif prediction == 0 and macro_filter == "Bajista 🔴" and confidence >= 52.0:
                        entry_price = current_price
                        position_size = (paper_balance_usdt * leverage) / entry_price
                        take_profit_price = entry_price - (2.5 * current_atr)
                        stop_loss_price = entry_price + (1.5 * current_atr)
                        position_status = "SHORT"
                        
                        send_telegram_message(f"📉 *SHORT (2x)*\n• Entrada: `${entry_price:.2f}`\n• Confianza: `{confidence:.1f}%`")

                # Reporte periódico cada 30 minutos asegurado
                status_msg = (f"📊 *Reporte Periódico (30 min)*\n"
                              f"• Precio BTC: `${current_price:.2f}`\n"
                              f"• Predicción IA: `{prediction}` (Conf: `{confidence:.1f}%`)\n"
                              f"• Estado: `{position_status}`\n"
                              f"• Saldo Virtual: `${paper_balance_usdt:.2f}`")
                send_telegram_message(status_msg)
                
        except Exception as e:
            print(f"Error en el ciclo del bot: {e}")
            
        time.sleep(1800)

# Iniciar hilos en segundo plano (Trading + Keep-Alive)
threading.Thread(target=bot_loop, daemon=True).start()
threading.Thread(target=self_ping, daemon=True).start()

if __name__ == "__main__":
    run_flask()
