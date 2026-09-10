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
from xgboost import XGBClassifier

# --- CONFIGURACIÓN DE ENTORNO ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
BINANCE_API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
BINANCE_SECRET = os.environ.get("BINANCE_TESTNET_SECRET")

# Configuración de Capital y Riesgo Diario
INITIAL_CAPITAL = 50.0 
LEVERAGE = 2  
PROFIT_TARGET_PCT = 0.20  # +20% meta de ganancia diaria
MAX_LOSS_PCT = -0.10      # -10% límite máximo de pérdida diaria

# Variables de control de estado diario
current_day = None
starting_daily_balance = None
trading_halted_today = False

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot de Trading con Control de Riesgo Diario + IA", 200

# Inicializar exchange con control de errores
try:
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    exchange.set_sandbox_mode(True)
    print("Exchange de Binance Testnet configurado correctamente.")
except Exception as e:
    print(f"Error al configurar CCXT Binance: {e}")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Faltan credenciales de Telegram.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Telegram msg status: {response.status_code}")
    except Exception as e:
        print(f"Error Telegram msg: {e}")

def send_telegram_photo(photo_bytes, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('chart.png', photo_bytes, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=data, files=files, timeout=15)
        print(f"Telegram photo status: {response.status_code}")
    except Exception as e:
        print(f"Error Telegram photo: {e}")

def get_fear_and_greed_index():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            return int(data["data"][0]["value"]), data["data"][0]["value_classification"]
    except Exception as e:
        print(f"Error Fear & Greed API: {e}")
    return 50, "Neutral"

def fetch_data():
    bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_indicators(df):
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
    df['atr'] = np.max(ranges, axis=1).rolling(14).mean()

    df.dropna(inplace=True)
    return df

def generate_chart(df):
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    subset = df.tail(60)
    plt.plot(subset['timestamp'], subset['close'], label='Precio BTC', color='#00ffcc', linewidth=1.5)
    plt.plot(subset['timestamp'], subset['ema_20'], label='EMA 20', color='#ff007f', linewidth=1)
    plt.plot(subset['timestamp'], subset['ema_50'], label='EMA 50', color='#ffcc00', linewidth=1)
    plt.title('Control Diario de Riesgo - Bot IA ($50 Base)', fontsize=12, color='white')
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
    global current_day, starting_daily_balance, trading_halted_today
    
    print("Iniciando ciclo de trading...")
    try:
        now = datetime.datetime.utcnow().date()
        
        balance_info = exchange.fetch_balance()
        total_wallet_balance = float(balance_info['total']['USDT'])
        
        if current_day != now:
            current_day = now
            starting_daily_balance = total_wallet_balance
            trading_halted_today = False
            send_telegram_message(f"🌅 *Nuevo día de trading ({current_day})*\nBalance inicial del día registrado: `${total_wallet_balance:,.2f} USDT`")

        daily_pnl_pct = (total_wallet_balance - starting_daily_balance) / starting_daily_balance if starting_daily_balance > 0 else 0

        if daily_pnl_pct >= PROFIT_TARGET_PCT:
            if not trading_halted_today:
                trading_halted_today = True
                send_telegram_message(f"🎯 *¡Meta diaria del +20% alcanzada!* (${total_wallet_balance:,.2f} USDT). El bot pausará operaciones hasta mañana para asegurar ganancias.")
            return

        if daily_pnl_pct <= MAX_LOSS_PCT:
            if not trading_halted_today:
                trading_halted_today = True
                positions = exchange.fetch_positions()
                for p in positions:
                    if p['symbol'] == 'BTC/USDT:USDT' and float(p['contracts']) > 0:
                        exchange.create_market_sell_order('BTC/USDT', float(p['contracts']))
                send_telegram_message(f"🛑 *Límite de pérdida diaria alcanzado ({daily_pnl_pct*100:.1f}%)*. Posiciones cerradas. El bot descansa hasta mañana.")
            return

        if trading_halted_today:
            print("Trading pausado por hoy según límites de riesgo.")
            return

        df = fetch_data()
        df = calculate_indicators(df)
        current_price = df['close'].iloc[-1]
        fg_value, fg_text = get_fear_and_greed_index()
        
        try:
            exchange.set_leverage(LEVERAGE, 'BTC/USDT')
        except Exception:
            pass
        
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        features = ['rsi', 'ema_20', 'ema_50', 'macd', 'macd_signal', 'atr']
        X = df[features].iloc[:-1]
        y = df['target'].iloc[:-1]
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        model.fit(X, y)
        
        latest_features = df[features].iloc[[-1]]
        prediction = model.predict(latest_features)[0]
        
        positions = exchange.fetch_positions()
        btc_position = next((p for p in positions if p['symbol'] == 'BTC/USDT:USDT' and float(p['contracts']) > 0), None)
        
        report_msg = (
            f"📊 *Monitoreo Activo (Control Diario)*\n"
            f"• Balance actual: `${total_wallet_balance:,.2f} USDT`\n"
            f"• Rendimiento hoy: `{daily_pnl_pct*100:+.2f}%` (Meta: +20% | Límite: -10%)\n"
            f"• Precio BTC: `${current_price:,.2f}`\n"
        )
        
        if not btc_position:
            if fg_text in ["Extreme Greed", "Extreme Fear"]:
                report_msg += f"⚠️ *Filtro de Sentimiento:* Mercado en {fg_text}."
            elif prediction == 1:
                amount = 0.001 
                exchange.create_market_buy_order('BTC/USDT', amount)
                report_msg += f"🟢 *Orden LONG abierta* (`0.001 BTC`)"
            else:
                report_msg += "⚪ *Buscando entradas:* IA en espera."
        else:
            entry_price = float(btc_position['entryPrice'])
            pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE
            if pnl_pct <= -0.015 or pnl_pct >= 0.03:
                exchange.create_market_sell_order('BTC/USDT', float(btc_position['contracts']))
                report_msg += f"🏁 Posición cerrada por Take-Profit/Stop-Loss técnico. PnL: `{pnl_pct*100:+.2f}%`"
            else:
                report_msg += f"📈 Posición abierta. PnL actual: `{pnl_pct*100:+.2f}%`"
                
        chart_bytes = generate_chart(df)
        send_telegram_photo(chart_bytes, report_msg)
        print("Ciclo de trading ejecutado y enviado con éxito a Telegram.")
        
    except Exception as e:
        print(f"❌ Error crítico en ciclo de trading: {e}")

def background_loop():
    import threading
    def worker():
        time.sleep(5) # Esperar a que Flask/Gunicorn levanten
        # Mensaje de arranque inicial con éxito
        send_telegram_message("🚀 *¡El bot con IA y control de riesgo se ha iniciado correctamente!*")
        
        while True:
            try:
                run_trading_bot()
            except Exception as ex:
                print(f"Error general en hilo de trabajo: {ex}")
            time.sleep(300) # Esperar 5 minutos para el siguiente ciclo
            
    t = threading.Thread(target=worker, daemon=True)
    t.start()

background_loop()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
