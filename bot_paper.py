import os
import time
import threading
import requests
import ccxt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from flask import Flask

# Configuración inicial de Flask para mantener el servidor web activo
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot de Trading Avanzado Activo 24/7", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# Configuración del bot de trading (Paper Trading en Binance Futures)
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY', ''),
    'secret': os.getenv('BINANCE_SECRET_KEY', ''),
    'options': {'defaultType': 'future'}
})
exchange.set_sandbox_mode(True)  # Modo de prueba (Paper Trading)

SYMBOL = 'BTC/USDT'
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Variables de Estado del Paper Trading y Gestión de Riesgo
saldo_virtual = 1000.0  
posicion_actual = "FLAT"  # "FLAT", "LONG", o "SHORT"
precio_entrada = 0.0
margen_invertido = 0.0
apalancamiento = 2.0
cantidad_btc = 0.0
stop_loss_precio = 0.0
take_profit_precio = 0.0

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")

# --- INDICADORES TÉCNICOS AVANZADOS ---
def calcular_indicadores(df):
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMAs (Medias Móviles Exponenciales)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # ATR (Volatilidad)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Target de predicción para la IA
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df.dropna(inplace=True)
    return df

def trading_loop():
    global saldo_virtual, posicion_actual, precio_entrada, margen_invertido, cantidad_btc, stop_loss_precio, take_profit_precio
    print("Iniciando bot de futuros mejorado con histórico amplio y gestión de riesgo...")
    
    send_telegram_message(
        "🤖 *Bot de Futuros Mejorado Reiniciado*\n"
        "• Historial ampliado (500 velas 4h).\n"
        "• Indicadores: RSI, EMAs, MACD, ATR.\n"
        "• Protección activa: Stop-Loss y Take-Profit."
    )

    while True:
        try:
            # Ampliamos el historial a 500 velas para que la IA aprenda mejor
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='4h', limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df = calcular_indicadores(df)
            precio_actual = df['close'].iloc[-1]
            prediction_prob = 0.5
            
            if len(df) > 50:
                features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'ema_20', 'ema_50', 'macd', 'macd_signal', 'atr']
                X = df[features]
                y = df['target']
                
                # Entrenamiento del modelo XGBoost con más profundidad y robustez
                model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42)
                model.fit(X, y)
                
                latest_data = X.iloc[[-1]]
                prediction_prob = model.predict_proba(latest_data)[0][1]
                
            print(f"Precio BTC: {precio_actual} | Predicción IA: {prediction_prob:.2f} | Estado: {posicion_actual}")

            # --- GESTIÓN DE POSICIONES ABIERTAS (Revisión de Stop-Loss / Take-Profit) ---
            if posicion_actual == "LONG":
                # Verificar si tocó Stop-Loss o Take-Profit o señal contraria
                if precio_actual <= stop_loss_precio or precio_actual >= take_profit_precio or prediction_prob < 0.45:
                    pnl = (precio_actual - precio_entrada) * cantidad_btc
                    saldo_virtual += pnl
                    porcentaje_pnl = (pnl / margen_invertido) * 100
                    
                    motivo = "Take-Profit 🎯" if precio_actual >= take_profit_precio else ("Stop-Loss 🛑" if precio_actual <= stop_loss_precio else "Señal Contraria 🔄")
                    
                    mensaje = (
                        f"🔴 *CIERRE LONG ({motivo})*\n"
                        f"• Precio Salida: ${precio_actual:,.2f}\n"
                        f"• PnL: ${pnl:,.2f} ({porcentaje_pnl:+.2f}%)\n"
                        f"• Saldo Actualizado: ${saldo_virtual:,.2f}"
                    )
                    send_telegram_message(mensaje)
                    posicion_actual = "FLAT"

            elif posicion_actual == "SHORT":
                if precio_actual >= stop_loss_precio or precio_actual <= take_profit_precio or prediction_prob > 0.55:
                    pnl = (precio_entrada - precio_actual) * cantidad_btc
                    saldo_virtual += pnl
                    porcentaje_pnl = (pnl / margen_invertido) * 100
                    
                    motivo = "Take-Profit 🎯" if precio_actual <= take_profit_precio else ("Stop-Loss 🛑" if precio_actual >= stop_loss_precio else "Señal Contraria 🔄")
                    
                    mensaje = (
                        f"🟢 *CIERRE SHORT ({motivo})*\n"
                        f"• Precio Salida: ${precio_actual:,.2f}\n"
                        f"• PnL: ${pnl:,.2f} ({porcentaje_pnl:+.2f}%)\n"
                        f"• Saldo Actualizado: ${saldo_virtual:,.2f}"
                    )
                    send_telegram_message(mensaje)
                    posicion_actual = "FLAT"

            # --- APERTURA DE NUEVAS POSICIONES ---
            if posicion_actual == "FLAT":
                if prediction_prob > 0.55:  # Mayor filtro de seguridad para entrar
                    margen_invertido = saldo_virtual * 0.5
                    cantidad_btc = (margen_invertido * apalancamiento) / precio_actual
                    precio_entrada = precio_actual
                    
                    # Definir Stop-Loss (1.5%) y Take-Profit (3%) automáticos
                    stop_loss_precio = precio_entrada * 0.985
                    take_profit_precio = precio_entrada * 1.03
                    posicion_actual = "LONG"
                    
                    mensaje = (
                        f"🚀 *NUEVO LONG ({int(apalancamiento)}x)*\n"
                        f"• Entrada: ${precio_entrada:,.2f}\n"
                        f"• Stop-Loss: ${stop_loss_precio:,.2f}\n"
                        f"• Take-Profit: ${take_profit_precio:,.2f}\n"
                        f"• Confianza IA: {prediction_prob * 100:.1f}%"
                    )
                    send_telegram_message(mensaje)

                elif prediction_prob < 0.45:
                    margen_invertido = saldo_virtual * 0.5
                    cantidad_btc = (margen_invertido * apalancamiento) / precio_actual
                    precio_entrada = precio_actual
                    
                    # Definir Stop-Loss (1.5%) y Take-Profit (3%) para Short
                    stop_loss_precio = precio_entrada * 1.015
                    take_profit_precio = precio_entrada * 0.97
                    posicion_actual = "SHORT"
                    
                    confianza_short = (1 - prediction_prob) * 100
                    mensaje = (
                        f"📉 *NUEVO SHORT ({int(apalancamiento)}x)*\n"
                        f"• Entrada: ${precio_entrada:,.2f}\n"
                        f"• Stop-Loss: ${stop_loss_precio:,.2f}\n"
                        f"• Take-Profit: ${take_profit_precio:,.2f}\n"
                        f"• Confianza IA: {confianza_short:.1f}%"
                    )
                    send_telegram_message(mensaje)

            # --- REPORTE PERIÓDICO CADA 30 MINUTOS ---
            pred_binaria = 1 if prediction_prob > 0.5 else 0
            confianza_actual = prediction_prob if pred_binaria == 1 else (1 - prediction_prob)
            
            reporte = (
                f"📊 *Reporte Periódico (30 min)*\n"
                f"• Precio BTC: ${precio_actual:,.2f}\n"
                f"• Predicción IA: {pred_binaria} (Conf: {confianza_actual * 100:.1f}%)\n"
                f"• Estado: {posicion_actual}\n"
                f"• Saldo Virtual: ${saldo_virtual:,.2f}"
            )
            send_telegram_message(reporte)
            
        except Exception as e:
            print(f"Error en el ciclo de trading: {e}")
        
        time.sleep(1800)

# Iniciar el hilo del bot automáticamente al importar el módulo
t = threading.Thread(target=trading_loop)
t.daemon = True
t.start()

if __name__ == '__main__':
    run_flask()
