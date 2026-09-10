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
    return "Bot de Trading Profesional XGBoost (Futuros / Paper Trading) activo", 200

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
        # Descarga datos de 1 hora para operativa y cálculo de ATR
        bars_1h = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=150)
        df_1h = pd.DataFrame(bars_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Descarga datos de 4 horas para el Filtro de Tendencia Macro
        bars_4h = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=50)
        df_4h = pd.DataFrame(bars_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Indicadores técnicos en 1h
        df_1h['returns'] = df_1h['close'].pct_change()
        df_1h['sma'] = df_1h['close'].rolling(window=14).mean()
        df_1h['volatility'] = df_1h['returns'].rolling(window=14).std()
        
        # Cálculo de ATR (Average True Range) para Stop Loss / Take Profit dinámicos
        high_low = df_1h['high'] - df_1h['low']
        high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
        low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_1h['atr'] = true_range.rolling(window=14).mean()
        
        # Filtro Macro 4h (Media Móvil simple de 4h)
        df_1h['macro_trend'] = df_4h['close'].rolling(window=10).mean().iloc[-1] # Simplificado para alineación
        
        df_1h.dropna(inplace=True)
        return df_1h
    except Exception as e:
        print(f"Error obteniendo datos de Kraken: {e}")
        return None

def train_and_predict(df):
    try:
        features = ['returns', 'sma', 'volatility', 'volume', 'atr']
        # Objetivo: 1 si sube en la siguiente vela, 0 si baja
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

def bot_loop():
    global paper_balance_usdt, position_status, entry_price, position_size, take_profit_price, stop_loss_price
    
    print("Iniciando bot de futuros avanzado...")
    send_telegram_message("🚀 *Bot de Futuros XGBoost Activado*\nModo profesional con apalancamiento 2x, Long/Short y gestión por ATR.")
    
    while True:
        try:
            df = fetch_data()
            if df is not None and not df.empty:
                current_price = df['close'].iloc[-1]
                current_atr = df['atr'].iloc[-1]
                macro_sma = df['sma'].iloc[-1] # Referencia de tendencia local/macro
                
                prediction, confidence = train_and_predict(df)
                
                # Definir filtro macro (Alcista si precio > SMA macro, Bajista si precio < SMA macro)
                macro_filter = "Alcista 🟢" if current_price >= macro_sma else "Bajista 🔴"
                
                # --- GESTIÓN DE POSICIÓN ABIERICA (Revisión de Stop Loss / Take Profit) ---
                if position_status == "LONG":
                    if current_price >= take_profit_price or current_price <= stop_loss_price:
                        # Cierre por SL o TP
                        pnl_usd = (current_price - entry_price) * position_size
                        paper_balance_usdt += (position_size * entry_price) + pnl_usd
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
                        
                        msg = (f"🎯 *CIERRE DE POSICIÓN LONG (SL/TP)*\n"
                               f"• Precio Salida: `${current_price:.2f}`\n"
                               f"• PnL: `${pnl_usd:+.2f}` (`{pnl_pct:+.2f}%`)\n"
                               f"• Saldo Virtual: `${paper_balance_usdt:.2f}`")
                        send_telegram_message(msg)
                        position_status = "FLAT"
                        
                elif position_status == "SHORT":
                    if current_price <= take_profit_price or current_price >= stop_loss_price:
                        # Cierre por SL o TP en Short
                        pnl_usd = (entry_price - current_price) * position_size
                        paper_balance_usdt += (position_size * entry_price) + pnl_usd
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
                        
                        msg = (f"🎯 *CIERRE DE POSICIÓN SHORT (SL/TP)*\n"
                               f"• Precio Salida: `${current_price:.2f}`\n"
                               f"• PnL: `${pnl_usd:+.2f}` (`{pnl_pct:+.2f}%`)\n"
                               f"• Saldo Virtual: `${paper_balance_usdt:.2f}`")
                        send_telegram_message(msg)
                        position_status = "FLAT"

                # --- APERTURA DE NUEVAS POSICIONES SEGÚN SEÑAL Y FILTRO ---
                if position_status == "FLAT":
                    # Señal LONG (IA predice subida y tendencia macro es alcista o confianza alta)
                    if prediction == 1 and macro_filter == "Alcista 🟢" and confidence >= 53.0:
                        entry_price = current_price
                        position_size = (paper_balance_usdt * leverage) / entry_price
                        take_profit_price = entry_price + (2.0 * current_atr)
                        stop_loss_price = entry_price - (1.5 * current_atr)
                        position_status = "LONG"
                        
                        msg = (f"📈 *SEÑAL DE COMPRA (LONG 2x)*\n"
                               f"• Precio Entrada: `${entry_price:.2f}`\n"
                               f"• Confianza IA: `{confidence:.1f}%`\n"
                               f"• Take Profit: `${take_profit_price:.2f}`\n"
                               f"• Stop Loss: `${stop_loss_price:.2f}`\n"
                               f"• Filtro Macro: {macro_filter}")
                        send_telegram_message(msg)
                        
                    # Señal SHORT (IA predice bajada y tendencia macro es bajista)
                    elif prediction == 0 and macro_filter == "Bajista 🔴" and confidence >= 53.0:
                        entry_price = current_price
                        position_size = (paper_balance_usdt * leverage) / entry_price
                        take_profit_price = entry_price - (2.0 * current_atr)
                        stop_loss_price = entry_price + (1.5 * current_atr)
                        position_status = "SHORT"
                        
                        msg = (f"📉 *SEÑAL DE VENTA (SHORT 2x)*\n"
                               f"• Precio Entrada: `${entry_price:.2f}`\n"
                               f"• Confianza IA: `{confidence:.1f}%`\n"
                               f"• Take Profit: `${take_profit_price:.2f}`\n"
                               f"• Stop Loss: `${stop_loss_price:.2f}`\n"
                               f"• Filtro Macro: {macro_filter}")
                        send_telegram_message(msg)

                # Reporte periódico de estado cada 30 minutos
                status_msg = (f"📊 *Reporte Periódico de Futuros (30 min)*\n"
                              f"• Precio BTC: `${current_price:.2f}`\n"
                              f"• Predicción IA: `{prediction}` (Conf: `{confidence:.1f}%`)\n"
                              f"• Estado Posición: `{position_status}`\n"
                              f"• Filtro Macro: {macro_filter}\n"
                              f"• Saldo Virtual: `${paper_balance_usdt:.2f}`")
                send_telegram_message(status_msg)
                
        except Exception as e:
            print(f"Error en el ciclo del bot: {e}")
            send_telegram_message(f"⚠️ *Error crítico en el Bot*: {e}")
            
        # Esperar 30 minutos para el próximo ciclo
        time.sleep(1800)

# Iniciar hilo en segundo plano
bot_thread = threading.Thread(target=bot_loop, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    run_flask()
