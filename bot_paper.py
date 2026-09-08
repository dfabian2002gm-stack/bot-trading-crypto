import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import time
import os
import threading
from datetime import datetime
from flask import Flask

# ==========================================
# SERVIDOR WEB PARA RENDER (HEALTH CHECK)
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot de Trading activo y monitoreando 24/7."

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ==========================================
# CONFIGURACIÓN GENERAL Y TELEGRAM
# ==========================================
SYMBOL = "BTC/USDT"
TIMEFRAME_OPERATIVO = "1h"
TIMEFRAME_MACRO = "4h"
EMA_MACRO_PERIOD = 200
UMBRAL_PREDICCION = 0.50

TELEGRAM_TOKEN = "8685818386:AAFw7zOFdc1XeJxqtOixb15bE9UU5t7ixdw"
TELEGRAM_CHAT_ID = "6814040736"

in_position = False
entry_price = 0.0
take_profit = 0.0
trailing_stop = 0.0
highest_price = 0.0

exchange = ccxt.kraken()

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Error enviando mensaje a Telegram: {e}")

def get_macro_trend():
    try:
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME_MACRO, limit=250)
        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['ema_macro'] = df['close'].ewm(span=EMA_MACRO_PERIOD, adjust=False).mean()
        
        last_close = df['close'].iloc[-1]
        last_ema = df['ema_macro'].iloc[-1]
        return (last_close > last_ema), last_close, last_ema
    except Exception as e:
        print(f"⚠️ Error obteniendo tendencia macro: {e}")
        return True, 0, 0

def calculate_features(df):
    df['returns'] = df['close'].pct_change()
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['volatility'] = df['returns'].rolling(14).std()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    return df

MODEL_PATH = "xgb_btc_model.json"

def get_or_train_model():
    if not os.path.exists(MODEL_PATH):
        print("⏳ Entrenando modelo XGBoost...")
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME_OPERATIVO, limit=1000)
        df_train = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df_train = calculate_features(df_train)
        df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
        df_train = df_train.dropna()
        
        X = df_train[['returns', 'ema_fast', 'ema_slow', 'volatility', 'rsi']]
        y = df_train['target']
        
        dtrain = xgb.DMatrix(X, label=y)
        params = {'max_depth': 3, 'eta': 0.05, 'objective': 'binary:logistic'}
        mdl = xgb.train(params, dtrain, num_boost_round=50)
        mdl.save_model(MODEL_PATH)
        print("✓ Modelo entrenado y guardado.")
        return mdl
    else:
        mdl = xgb.Booster()
        mdl.load_model(MODEL_PATH)
        print("✓ Modelo cargado desde archivo.")
        return mdl

model = get_or_train_model()

def check_signals():
    global in_position, entry_price, take_profit, trailing_stop, highest_price
    
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME_OPERATIVO, limit=100)
    df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = calculate_features(df)
    
    macro_bullish, _, _ = get_macro_trend()
    last_row = df.iloc[-1]
    current_price = last_row['close']
    atr = last_row['atr']
    
    features = df[['returns', 'ema_fast', 'ema_slow', 'volatility', 'rsi']].iloc[[-1]]
    dmatrix = xgb.DMatrix(features)
    pred_val = float(model.predict(dmatrix)[0])
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] BTC: ${current_price:.2f} | Pred: {pred_val:.4f} | En Posición: {in_position}")
    
    if in_position:
        if current_price > highest_price:
            highest_price = current_price
            trailing_stop = highest_price - (atr * 1.5)

        if current_price >= take_profit:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            msg = f"🎯 *TAKE PROFIT ALCANZADO*\n• Salida: ${current_price:.2f}\n• Entrada: ${entry_price:.2f}\n• PnL: +{pnl_pct:.2f}% ✅"
            send_telegram(msg)
            in_position = False

        elif current_price <= trailing_stop:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            status = "✅" if pnl_pct > 0 else "❌"
            msg = f"🛑 *TRAILING STOP ACTIVADO*\n• Salida: ${current_price:.2f}\n• Entrada: ${entry_price:.2f}\n• PnL: {pnl_pct:+.2f}% {status}"
            send_telegram(msg)
            in_position = False

    else:
        if pred_val > UMBRAL_PREDICCION:
            if macro_bullish:
                in_position = True
                entry_price = current_price
                highest_price = current_price
                take_profit = current_price + (atr * 2.0)
                trailing_stop = current_price - (atr * 1.5)
                
                msg = (f"🚀 *COMPRA SIMULADA (PAPER TRADING)*\n\n"
                       f"• Par: {SYMBOL}\n"
                       f"• Entrada: ${entry_price:.2f}\n"
                       f"• XGBoost: {pred_val*100:.1f}%\n"
                       f"• Take Profit: ${take_profit:.2f}\n"
                       f"• Trailing Stop: ${trailing_stop:.2f}\n"
                       f"• Filtro Macro (4h): Alcista 🟢")
                send_telegram(msg)

def bot_loop():
    send_telegram("🤖 *Bot Desplegado en Render Web Service (24/7)*")
    while True:
        try:
            check_signals()
            time.sleep(60)
        except Exception as e:
            print(f"Error en bucle: {e}")
            time.sleep(10)

if __name__ == "__main__":
    # Inicia Flask en un hilo separado
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()
    
    # Inicia el bucle principal del bot
    bot_loop()
