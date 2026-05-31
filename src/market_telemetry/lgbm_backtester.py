import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def run_lgbm_fixed_backtester():
    csv_file = "multidim_labeled_market_data.csv"
    model_file = "lgbm_market_model.pkl"

    print(f"📖 Загружаю датасет {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        model = joblib.load(model_file)
    except FileNotFoundError as e:
        print(f"❌ Ошибка загрузки файлов: {e}")
        return

    feature_cols = [
        "imbalance_5",
        "imbalance_20",
        "imbalance_50",
        "market_delta_10s",
        "trade_speed_10s",
        "speed_zscore",
        "delta_rolling_2m",
        "delta_rolling_5m",
        "imb_20_velocity",
    ]

    X = df[feature_cols]
    y = df["label_next_price"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    preds_proba = model.predict(X_test)
    df_test = df.loc[y_test.index].copy()
    df_test["proba"] = preds_proba

    # ==========================================
    # ⚙️ НАСТРОЙКИ РОБОТА-СКАЛЬПЕРА
    # ==========================================
    confidence_threshold = 0.55
    tp_sl_size = 55.0
    commission_rate = 0.0006  # 0.06% Bybit Taker

    start_balance = 1000.0
    balance = start_balance
    position = 0
    entry_price = 0.0

    # Временные переменные для фиксации параметров открываемой сделки
    entry_time = None
    entry_conf = 0.0

    # Новый структурированный массив сделок для визуализатора
    executed_trades = []

    print(
        f"🚀 Скальпер запущен. Порог: {confidence_threshold*100}% | TP/SL: ${tp_sl_size}"
    )

    for i in range(len(df_test)):
        current_row = df_test.iloc[i]
        current_price = current_row["price"]
        proba = current_row["proba"]
        timestamp = current_row["timestamp"]

        # 1. ЛОГИКА ВХОДА
        if proba >= confidence_threshold and position == 0:
            position = 1
            entry_price = current_price
            entry_time = timestamp
            entry_conf = proba
            balance -= balance * commission_rate
            continue

        # 2. ЛОГИКА ВЫХОДА ПО ЖЕСТКИМ МИШЕНЯМ
        if position == 1:
            price_change = current_price - entry_price

            # Выход по Take Profit
            if price_change >= tp_sl_size:
                position = 0
                price_return = (current_price - entry_price) / entry_price
                balance += balance * price_return
                balance -= balance * commission_rate

                executed_trades.append({
                    "direction": "LONG",
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": timestamp,
                    "exit_price": current_price,
                    "tp_price": entry_price + tp_sl_size,
                    "sl_price": entry_price - tp_sl_size,
                    "result": "PROFIT",
                    "conf": entry_conf
                })

            # Выход по Stop Loss
            elif price_change <= -tp_sl_size:
                position = 0
                price_return = (current_price - entry_price) / entry_price
                balance += balance * price_return
                balance -= balance * commission_rate

                executed_trades.append({
                    "direction": "LONG",
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": timestamp,
                    "exit_price": current_price,
                    "tp_price": entry_price + tp_sl_size,
                    "sl_price": entry_price - tp_sl_size,
                    "result": "LOSS",
                    "conf": entry_conf
                })

    # Принудительное закрытие в конце истории
    if position == 1:
        current_row = df_test.iloc[-1]
        current_price = current_row["price"]
        price_return = (current_price - entry_price) / entry_price
        balance += balance * price_return
        balance -= balance * commission_rate

        executed_trades.append({
            "direction": "LONG",
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": current_row["timestamp"],
            "exit_price": current_price,
            "tp_price": entry_price + tp_sl_size,
            "sl_price": entry_price - tp_sl_size,
            "result": "PROFIT" if current_price > entry_price else "LOSS",
            "conf": entry_conf
        })

    # МЕТРИКИ ДЛЯ КОНСОЛИ
    total_trades = len(executed_trades)
    tps = len([t for t in executed_trades if t["result"] == "PROFIT"])
    sls = len([t for t in executed_trades if t["result"] == "LOSS"])

    net_profit_usdt = balance - start_balance
    profit_percent = (net_profit_usdt / start_balance) * 100

    print("\n" + "=" * 50)
    print(f"📊 ИТОГОВЫЙ ОТЧЕТ СКАЛЬПЕРА С TP/SL:")
    print("=" * 50)
    print(f"💰 Стартовый баланс:         {start_balance} USDT")
    print(f"💵 Финальный баланс:         {balance:.2f} USDT")
    print(f"📈 Чистый Профит:            {net_profit_usdt:.2f} USDT ({profit_percent:.2f}%)")
    print(f"🔄 Закрытых сделок:          {total_trades} (🟢 TP: {tps} | 🔴 SL: {sls})")

    # ==========================================
    # 💾 ЧЕСТНЫЙ ЭКСПОРТ ДЛЯ ВИЗУАЛИЗАТОРА
    # ==========================================
    if total_trades > 0:
        trades_df = pd.DataFrame(executed_trades)
        trades_df.to_csv("trades_log.csv", index=False)
        print(f"💾 Лог сделок сохранен в trades_log.csv для зеркальной отрисовки.")


if __name__ == "__main__":
    run_lgbm_fixed_backtester()
