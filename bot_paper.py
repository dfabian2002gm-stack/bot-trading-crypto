import os
import time
import io
import requests
import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Modo sin interfaz gráfica para que funcione perfectamente en la nube
import matplotlib.pyplot as plt
from flask import Flask
from xgboost import XGBClassifier

# --- CONFIGURACIÓN DESDE VARIABLES DE ENTORNO ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
BINANCE_API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
BINANCE_SECRET = os.environ.get("BINANCE_TESTNET_SECRET")

# Configuración estricta para cuenta simulada de $50 USD
INITIAL_CAPITAL = 50.0 
LEVERAGE = 2  # Apalancamiento conservador 2x

# Servidor Flask simple para mantener vivo el servicio en Render
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot de Trading con gráficos visuales para Telegram + IA", 200

# --- CONFIGURACIÓN DEL EXCHANGE EN MODO TESTNET ---
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})
exchange.set_sandbox_mode(True)

def send_telegram_message(message):
    """Envía notificaciones de texto al chat de Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error al enviar mensaje a Telegram: {e}")

def send_telegram_photo(photo_bytes, caption):
    """Envía una imagen generada con gráficos directamente a Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('chart.png', photo_bytes, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=data, files=files, timeout=15)
    except Exception as e:
        print(f"Error al enviar foto a Telegram: {e}")

def get_fear_and_greed_index():
    """Consulta el índice de Miedo y Codicia del mercado cripto."""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            value = int(data["data"][0]["value"])
            classification = data["data"][0]["value_classification"]
            return value, classification
    except Exception as e:
        print(f"No se pudo obtener el Fear & Greed Index: {e}")
    return 50, "Neutral"

def fetch_data():
    """Descarga velas de 4h de BTC/USDT desde la Testnet de Binance."""
    bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=500)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_indicators(df):
    """Añade indicadores técnicos avanzados al DataFrame."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()

    df.dropna(inplace=True)
    return df

def generate_chart(df):
    """Genera un gráfico limpio de precios y EMAs para enviar por Telegram."""
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    
    # Tomamos las últimas 60 velas para que el gráfico se vea detallado
    subset = df.tail(60)
    plt.plot(subset['timestamp'], subset['close'], label='Precio BTC', color='#00ffcc', linewidth=1.5)
    plt.plot(subset['timestamp'], subset['ema_20'], label='EMA 20', color='#ff007f', linewidth=1)
    plt.plot(subset['timestamp'], subset['ema_50'], label='EMA 50', color='#ffcc00', linewidth=1)
    
    plt.title('Análisis Técnico - Bot Testnet ($50)', fontsize=12, color='white')
    plt.xlabel('Fecha / Hora', color='gray')
    plt.ylabel('Precio (USDT)', color='gray')
    plt.legend(loc='upper left')
    plt.grid(True, color='#333333', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=15, color='gray', fontsize=8)
    plt.yticks(color='gray', fontsize=8)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    return buf.read()

def run_trading_bot():
    try:
        print("Ejecutando ciclo del bot con gráficos y cuenta de $50...")
        df = fetch_data()
        df = calculate_indicators(df)
        
        current_price = df['close'].iloc[-1]
        fg_value, fg_text = get_fear_and_greed_index()
        
        try:
            exchange.set_leverage(LEVERAGE, 'BTC/USDT')
        except Exception:
            pass
        
        # Preparar Machine Learning (XGBoost)
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        features = ['rsi', 'ema_20', 'ema_50', 'macd', 'macd_signal', 'atr']
        
        X = df[features].iloc[:-1]
        y = df['target'].iloc[:-1]
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        model.fit(X, y)
        
        latest_features = df[features].iloc[[-1]]
        prediction = model.predict(latest_features)[0]
        
        # Consultar posiciones abiertas en Testnet
        positions = exchange.fetch_positions()
        btc_position = next((p for p in positions if p['symbol'] == 'BTC/USDT:USDT' and float(p['contracts']) > 0), None)
        
        report_msg = (
            f"🧪 *Reporte Testnet ($50 Base)*\n"
            f"• Apalancamiento: `{LEVERAGE}x`\n"
            f"• Precio BTC: `${current_price:,.2f}`\n"
            f"• Sentimiento: `{fg_value} / 100 ({fg_text})`\n"
        )
        
        if not btc_position:
            if fg_text in ["Extreme Greed", "Extreme Fear"]:
                report_msg += f"⚠️ *Filtro activado:* Mercado en {fg_text}. Operación pausada."
            elif prediction == 1:
                amount = 0.001 
                exchange.create_market_buy_order('BTC/USDT', amount)
                report_msg += f"🟢 *Orden LONG ejecutada* (`0.001 BTC` a `${current_price:,.2f}`)"
            else:
                report_msg += "⚪ *Sin posición:* Esperando señal de compra de la IA."
        else:
            entry_price = float(btc_position['entryPrice'])
            pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE
            
            if pnl_pct <= -0.015 or pnl_pct >= 0.03:
                exchange.create_market_sell_order('BTC/USDT', float(btc_position['contracts']))
                action_type = "Stop-Loss 🔴" if pnl_pct <= -0.015 else "Take-Profit 🔵"
                report_msg += f"🏁 *{action_type}*. Cerrada con PnL: `{pnl_pct*100:+.2f}%`"
            else:
                report_msg += f"📈 *Manteniendo posición*. PnL actual: `{pnl_pct*100:+.2f}%`"
                
        # Generar imagen del gráfico y enviarla a Telegram junto al texto
        chart_bytes = generate_chart(df)
        send_telegram_photo(chart_bytes, report_msg)
        
    except Exception as e:
        print(f"Error en el ciclo del bot: {e}")
        send_telegram_message(f"⚠️ *Error en Bot Testnet:* `{str(e)}`")

def background_loop():
    import threading
    def worker():
        while True:
            run_trading_bot()
            time.sleep(14400) # Cada 4 horas
            
    t = threading.Thread(target=worker, daemon=True)
    t.start()

if __name__ == "__main__":
    background_loop()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
