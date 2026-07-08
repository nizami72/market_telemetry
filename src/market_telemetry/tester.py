import sys
import joblib
import numpy as np
import pandas as pd
import configparser
import os

def run_lgbm_fixed_backtester(csv_file):
    config = configparser.ConfigParser()
    config.read("config.ini")

    model_file = config.get("MARKET_DATA", "model_file")

    print(f"📖 Загружаю датасет {csv_file}...")
    try:
        df_test = pd.read_csv(csv_file)
        print(f"🛡️ Режим полного тестирования. Для симуляции взят весь файл: {len(df_test)} строк.")
    except FileNotFoundError as e:
        print(f"❌ Ошибка загрузки файлов: {e}")
        return

    threshold = config.getfloat("BACKTESTER", "confidence_threshold")
    initial_balance = config.getfloat("BACKTESTER", "initial_balance")
    tp_sl_size = config.getfloat("BACKTESTER", "tp_sl_size")
    risk_per_trade = config.getfloat("BACKTESTER", "risk_per_trade")
    commission_rate = config.getfloat("BACKTESTER", "commission_rate")

    feature_cols = [
        "imbalance_5", "imbalance_20", "imbalance_50",
        "market_delta_10s", "trade_speed_10s", "speed_zscore",
        "delta_rolling_2m", "delta_rolling_5m", "imb_20_velocity",
        "delta_rolling_30m", "delta_rolling_1h", "price_velocity_15m",
        "speed_ratio_1m", "speed_ratio_5m", "speed_ratio_15m",
        "cum_delta_1m", "cum_delta_5m", "cum_delta_15m",
        "price_change_5m", "price_change_1h"
    ]

    for col in feature_cols:
        if col not in df_test.columns:
            df_test[col] = 0.0

    X_test = df_test[feature_cols].values

    print(f"🤖 Загружаю модель {model_file}...")
    model = joblib.load(model_file)

    print("🔮 Делаю предсказания...")
    preds_proba = model.predict(X_test)

    df_test["proba_down"] = preds_proba[:, 0]
    df_test["proba_flat"] = preds_proba[:, 1]
    df_test["proba_up"]   = preds_proba[:, 2]

    # ГЕНЕРИРУЕМ СИГНАЛЫ
    conditions = [
        df_test["proba_up"] > threshold,
        df_test["proba_down"] > threshold
    ]
    choices = [1, -1]
    df_test["signal"] = np.select(conditions, choices, default=0)

    print("\n📊 ДИАГНОСТИКА ВЕРОЯТНОСТЕЙ ИИ (Первые 5 строк теста):")
    print(df_test[["proba_down", "proba_flat", "proba_up"]].head())

    print("\n📊 МАКСИМАЛЬНЫЕ ЗНАЧЕНИЯ ВЕРОЯТНОСТЕЙ НА ВСЕМ ТЕСТЕ:")
    print(f"Max Up: {df_test['proba_up'].max():.4f} | Max Down: {df_test['proba_down'].max():.4f}")

    print(f"Количество сигналов LONG (1): {df_test[df_test['signal'] == 1].shape[0]}")
    print(f"Количество сигналов SHORT (-1): {df_test[df_test['signal'] == -1].shape[0]}\n")

    print("⚙️ Запускаю торговую симуляцию...\n")
    balance = initial_balance
    position = 0
    entry_price = 0.0
    position_size = 0.0

    trade_logs = []
    executed_trades = []
    win_trades = 0
    loss_trades = 0
    price_col = "price"

    for i, row in df_test.iterrows():
        signal = row["signal"]
        current_price = row[price_col]
        timestamp = row.get("timestamp", f"Tick_{i}")

        if position != 0:
            is_closed = False
            pnl = 0.0
            action_str = ""

            if position == 1:
                price_change = current_price - entry_price
                if price_change >= tp_sl_size:
                    pnl = tp_sl_size * position_size
                    action_str = "TAKE_PROFIT LONG"
                    win_trades += 1
                    is_closed = True
                elif price_change <= -tp_sl_size:
                    pnl = -tp_sl_size * position_size
                    action_str = "STOP_LOSS LONG"
                    loss_trades += 1
                    is_closed = True
                elif signal == -1:
                    pnl = price_change * position_size
                    action_str = "REVERSE CLOSE LONG"
                    if pnl > 0: win_trades += 1
                    else: loss_trades += 1
                    is_closed = True

            elif position == -1:
                price_change = entry_price - current_price
                if price_change >= tp_sl_size:
                    pnl = tp_sl_size * position_size
                    action_str = "TAKE_PROFIT SHORT"
                    win_trades += 1
                    is_closed = True
                elif price_change <= -tp_sl_size:
                    pnl = -tp_sl_size * position_size
                    action_str = "STOP_LOSS SHORT"
                    loss_trades += 1
                    is_closed = True
                elif signal == 1:
                    pnl = price_change * position_size
                    action_str = "REVERSE CLOSE SHORT"
                    if pnl > 0: win_trades += 1
                    else: loss_trades += 1
                    is_closed = True

            if is_closed:
                balance += pnl
                balance -= (current_price * position_size) * commission_rate
                trade_logs.append({
                    "time": timestamp, "action": action_str,
                    "price": current_price, "pnl": pnl, "balance": balance
                })
                position = 0
                position_size = 0.0
                continue

        if position == 0 and signal != 0:
            position = signal
            entry_price = current_price
            cash_risk = balance * risk_per_trade
            position_size = cash_risk / tp_sl_size

            balance -= (entry_price * position_size) * commission_rate
            action_str = "OPEN LONG" if position == 1 else "OPEN SHORT"
            trade_logs.append({
                "time": timestamp, "action": action_str,
                "price": current_price, "pnl": 0.0, "balance": balance
            })

    if position != 0:
        last_row = df_test.iloc[-1]
        last_price = last_row[price_col]
        pnl = (last_price - entry_price) * position * position_size
        balance += pnl
        balance -= (last_price * position_size) * commission_rate
        trade_logs.append({
            "time": last_row.get("timestamp", "End"),
            "action": "FORCE CLOSE", "price": last_price, "pnl": pnl, "balance": balance
        })
        if pnl > 0: win_trades += 1
        else: loss_trades += 1

    if trade_logs:
        print("📝 Последние сделки из лога:")
        for log in trade_logs[-10:]:
            pnl_str = f"PnL: {log['pnl']:+8.2f}" if log['pnl'] != 0 else "PnL:     0.00"
            print(f"[{log['time']}] {log['action']:<18} | Цена: {log['price']:<8.2f} | {pnl_str} | Баланс: {log['balance']:.2f}")

    total_trades = win_trades + loss_trades
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

    print("\n" + "=" * 50)
    print(f"📊 ИТОГИ МАКРО-ТОРГОВОЙ СИМУЛЯЦИИ:")
    print("=" * 50)
    print(f"Начальный баланс:  {initial_balance:.2f} USDT")
    print(f"Итоговый баланс:   {balance:.2f} USDT")
    print(f"Чистая прибыль:    {balance - initial_balance:+.2f} USDT")
    print(f"Всего сделок:      {total_trades}")
    print(f"Успешных сделок:   {win_trades} ({win_rate:.1f}%)")
    print(f"Убыточных сделок:  {loss_trades}")
    print("=" * 50)

    if trade_logs:
        current_trade = None
        for log in trade_logs:
            if "OPEN" in log["action"]:
                current_trade = {
                    "direction": "LONG" if "LONG" in log["action"] else "SHORT",
                    "entry_time": log["time"],
                    "entry_price": log["price"],
                    "tp_price": log["price"] + tp_sl_size if "LONG" in log["action"] else log["price"] - tp_sl_size,
                    "sl_price": log["price"] - tp_sl_size if "LONG" in log["action"] else log["price"] + tp_sl_size,
                }
            elif current_trade is not None:
                current_trade["exit_time"] = log["time"]
                current_trade["exit_price"] = log["price"]
                current_trade["result"] = "PROFIT" if "TAKE_PROFIT" in log["action"] or log["pnl"] > 0 else "LOSS"
                executed_trades.append(current_trade)
                current_trade = None

    if executed_trades:
        output_dir = "../../data"
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "trades_log.csv")
        pd.DataFrame(executed_trades).to_csv(log_path, index=False)
        print(f"💾 Лог сделок успешно сохранен в {log_path} для отрисовки графиков.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Ошибка запуска. Использование: python tester.py <путь_к_файлу_для_теста.csv>")
        sys.exit(1)

    target_csv = sys.argv[1]
    run_lgbm_fixed_backtester(target_csv)
