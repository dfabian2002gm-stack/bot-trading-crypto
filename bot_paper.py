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
                
                # Enviar reporte obligatorio en cada ciclo para que veas que está vivo
                status_msg = (f"📊 *Reporte Periódico (30 min)*\n"
                              f"• Precio BTC: `${current_price:.2f}`\n"
                              f"• Predicción: `{prediction}`\n"
                              f"• Estado: `{position_status}`\n"
                              f"• Saldo Virtual: `${paper_balance_usdt:.2f}`")
                send_telegram_message(status_msg)
                
                # Lógica de Paper Trading con PnL
                if prediction == 1 and position_status == "FLAT":
                    # Simular Compra (LONG)
                    paper_btc_held = (paper_balance_usdt * 0.99) / current_price 
                    entry_price = current_price
                    paper_balance_usdt = 0.0
                    position_status = "LONG"
                    
                    msg = (f"🟢 *SIMULACIÓN DE COMPRA (LONG)*\n"
                           f"• Precio de Entrada: `${entry_price:.2f}`\n"
                           f"• BTC Adquirido: `{paper_btc_held:.5f}`")
                    send_telegram_message(msg)
                    
                elif prediction == 0 and position_status == "LONG":
                    # Simular Venta / Cierre de posición
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
