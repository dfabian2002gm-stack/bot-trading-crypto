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
    return "Bot de Trading Activo y en Funcionamiento 24/7", 200

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
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Variables de Estado del Paper Trading
saldo_virtual = 1000.0  # Capital inicial base
posicion_actual = "FLAT"  # "FLAT", "LONG", o "SHORT"
precio_entrada = 0.0
margen_invertido = 0.0
apalancamiento = 2.0
cantidad_btc = 0.0

def send_telegram_message(message):
    print(f"DEBUG -> Token configurado: {bool(TELEGRAM_TOKEN)} | Chat ID configurado: {bool(TELEGRAM_CHAT_ID)}")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: Faltan las credenciales de Telegram en las variables de entorno.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"DEBUG -> Respuesta Telegram Status: {response.status_code}, Res: {response.text}")
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def trading_loop():
    global saldo_virtual, posicion_actual, precio_entrada, margen_invertido, cantidad_btc
    print("Iniciando bot de futuros avanzado con desglose transparente...")
    
    # Mensaje inicial de reinicio
    send_telegram_message(
        "🤖 *Bot de Futuros XGBoost Reiniciado*\n"
        "• RSI y optimización de rentabilidad activos.\n"
        "• Sistema Keep-Alive para evitar suspensiones."
    )

    while True:
        try:
            # Obtener datos de velas de 4 horas
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='4h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Indicadores técnicos
            df['rsi'] = calcular_rsi(df['close'], 14)
            df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            df.dropna(inplace=True)
            
            precio_actual = df['close'].iloc[-1]
            prediction_prob = 0.5
            
            if len(df) > 30:
                features = ['open', 'high', 'low', 'close', 'volume', 'rsi']
                X = df[features]
                y = df['target']
                
                # Entrenamiento del modelo XGBoost
                model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
                model.fit(X, y)
                
                # Predicción de la última vela
                latest_data = X.iloc[[-1]]
                prediction_prob = model.predict_proba(latest_data)[0][1]
                
            print(f"Precio BTC: {precio_actual} | Predicción IA: {prediction_prob:.2f} | Estado: {posicion_actual}")

            # --- LÓGICA DE GESTIÓN DE POSICIONES ---
            if posicion_actual == "FLAT":
                # Buscar entrada LONG (Confianza > 52%)
                if prediction_prob > 0.52:
                    margen_invertido = saldo_virtual * 0.5  # Arriesga el 50% del saldo actual como margen
                    cantidad_btc = (margen_invertido * apalancamiento) / precio_actual
                    precio_entrada = precio_actual
                    posicion_actual = "LONG"
                    
                    mensaje = (
                        f"🚀 *LONG ({int(apalancamiento)}x)*\n"
                        f"• Entrada: ${precio_entrada:,.2f}\n"
                        f"• Margen Usado: ${margen_invertido:,.2f}\n"
                        f"• Tamaño Posición: {cantidad_btc:.4f} BTC\n"
                        f"• Confianza IA: {prediction_prob * 100:.1f}%"
                    )
                    send_telegram_message(mensaje)

                # Buscar entrada SHORT (Confianza < 48%)
                elif prediction_prob < 0.48:
                    margen_invertido = saldo_virtual * 0.5  # Arriesga el 50% del saldo actual como margen
                    cantidad_btc = (margen_invertido * apalancamiento) / precio_actual
                    precio_entrada = precio_actual
                    posicion_actual = "SHORT"
                    
                    confianza_short = (1 - prediction_prob) * 100
                    mensaje = (
                        f"📉 *SHORT ({int(apalancamiento)}x)*\n"
                        f"• Entrada: ${precio_entrada:,.2f}\n"
                        f"• Margen Usado: ${margen_invertido:,.2f}\n"
                        f"• Tamaño Posición: {cantidad_btc:.4f} BTC\n"
                        f"• Confianza IA: {confianza_short:.1f}%"
                    )
                    send_telegram_message(mensaje)

            elif posicion_actual == "LONG":
                # Condición de cierre para LONG (si la IA se vuelve bajista)
                if prediction_prob < 0.48:
                    pnl = (precio_actual - precio_entrada) * cantidad_btc
                    saldo_virtual += pnl
                    porcentaje_pnl = (pnl / margen_invertido) * 100
                    
                    mensaje = (
                        f"🔴 *CIERRE LONG (Señal contraria)*\n"
                        f"• Precio Salida: ${precio_actual:,.2f}\n"
                        f"• PnL: ${pnl:,.2f} ({porcentaje_pnl:+.2f}%)\n"
                        f"• Margen Inicial: ${margen_invertido:,.2f}\n"
                        f"• Saldo Actualizado: ${saldo_virtual:,.2f}"
                    )
                    send_telegram_message(mensaje)
                    posicion_actual = "FLAT"

            elif posicion_actual == "SHORT":
                # Condición de cierre para SHORT (si la IA se vuelve alcista)
                if prediction_prob > 0.52:
                    pnl = (precio_entrada - precio_actual) * cantidad_btc
                    saldo_virtual += pnl
                    porcentaje_pnl = (pnl / margen_invertido) * 100
                    
                    mensaje = (
                        f"🟢 *CIERRE SHORT (Señal contraria)*\n"
                        f"• Precio Salida: ${precio_actual:,.2f}\n"
                        f"• PnL: ${pnl:,.2f} ({porcentaje_pnl:+.2f}%)\n"
                        f"• Margen Inicial: ${margen_invertido:,.2f}\n"
                        f"• Saldo Actualizado: ${saldo_virtual:,.2f}"
                    )
                    send_telegram_message(mensaje)
                    posicion_actual = "FLAT"

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
        
        # Esperar 30 minutos antes de la siguiente ejecución
        time.sleep(1800)

# Iniciar el hilo del bot automáticamente al importar el módulo (para Gunicorn)
t = threading.Thread(target=trading_loop)
t.daemon = True
t.start()

if __name__ == '__main__':
    run_flask()
