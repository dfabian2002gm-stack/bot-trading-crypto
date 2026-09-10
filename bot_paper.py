import os
import time
import threading
import ccxt
import xgboost as xgb
import numpy as np
import pandas as pd
import requests
from flask import Flask

# Configuración inicial de Flask para mantener vivo el servicio web en Render
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
    send_telegram_message("🤖 *Bot de Trading Iniciado*\nModo Paper Trading activado con PnL y revisiones cada 30 minutos.")
    
    while True:
        try:
            print("Ejecutando ciclo de análisis...")
            df = fetch_data()
            if df is not None and not df.empty:
                current_price = df['close'].iloc[-1]
                prediction = train_and_predict(df)
                
                print(f"Precio actual BTC: ${current_price:.2f} | Predicción XGBoost: {prediction}")
                
                # Lógica de Paper Trading con PnL
                if prediction == 1 and position_status == "FLAT":
                    # Simular Compra (LONG)
                    paper_btc_held = (paper_balance_usdt * 0.99) / current_price # Simulando comisión del 1%
                    entry_price = current_price
                    paper_balance_usdt = 0.0
                    position_status = "LONG"
                    
                    msg = (f"🟢 *SIMULACIÓN DE COMPRA (LONG)*\n"
                           f"• Precio de Entrada: `${entry_price:.2f}`\n"
                           f"• BTC Adquirido: `{paper_btc_held:.5f}`")
                    send_telegram_message(msg)
                    
                elif prediction == 0 and position_status == "LONG":
                    # Simular Venta / Cierre de posición
                    sale_proceeds = paper_btc_held * current_price * 0.99 # Comisión de venta
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
            
        # Esperar exactamente 30 minutos (1800 segundos) para el próximo ciclo
        time.sleep(1800)

# Iniciar el bucle del bot en un hilo en segundo plano al cargar el módulo en Render
bot_thread = threading.Thread(target=bot_loop, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    run_flask()in_position = False
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

# Iniciar el bucle del bot en un hilo en segundo plano al cargar el módulo en Render
bot_thread = threading.Thread(target=bot_loop, daemon=True)
bot_thread.start()
