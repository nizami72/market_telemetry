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
    confidence_threshold = 0.55  # Входим только при уверенности 60%+
    tp_sl_size = 55.0  # Жесткий Тейк и Стоп в $6 (чуть выше нашего порога разметки $5)
    commission_rate = 0.0006  # 0.06% Bybit Taker

    start_balance = 1000.0
    balance = start_balance
    position = 0
    entry_price = 0.0

    trades_history = []

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
            balance -= balance * commission_rate
            trades_history.append(
                {
                    "type": "BUY",
                    "price": entry_price,
                    "balance": balance,
                    "time": timestamp,
                    "conf": proba,
                }
            )
            continue

        # 2. ЛОГИКА ВЫХОДА ПО ЖЕСТКИМ МИШЕНЯМ (Внутри позиции)
        if position == 1:
            price_change = current_price - entry_price

            # Выход по Take Profit
            if price_change >= tp_sl_size:
                position = 0
                price_return = (current_price - entry_price) / entry_price
                balance += balance * price_return
                balance -= balance * commission_rate
                trades_history.append(
                    {
                        "type": "TAKE_PROFIT",
                        "price": current_price,
                        "balance": balance,
                        "time": timestamp,
                    }
                )

            # Выход по Stop Loss
            elif price_change <= -tp_sl_size:
                position = 0
                price_return = (current_price - entry_price) / entry_price
                balance += balance * price_return
                balance -= balance * commission_rate
                trades_history.append(
                    {
                        "type": "STOP_LOSS",
                        "price": current_price,
                        "balance": balance,
                        "time": timestamp,
                    }
                )

    # Принудительное закрытие в конце
    if position == 1:
        current_price = df_test.iloc[-1]["price"]
        price_return = (current_price - entry_price) / entry_price
        balance += balance * price_return
        balance -= balance * commission_rate
        trades_history.append(
            {
                "type": "END_CLOSE",
                "price": current_price,
                "balance": balance,
                "time": df_test.iloc[-1]["timestamp"],
            }
        )

    # МЕТРИКИ
    total_trades = len(
        [t for t in trades_history if t["type"] in ["TAKE_PROFIT", "STOP_LOSS", "END_CLOSE"]]
    )
    tps = len([t for t in trades_history if t["type"] == "TAKE_PROFIT"])
    sls = len([t for t in trades_history if t["type"] == "STOP_LOSS"])

    net_profit_usdt = balance - start_balance
    profit_percent = (net_profit_usdt / start_balance) * 100

    print("\n" + "=" * 50)
    print(f"📊 ИТОГОВЫЙ ОТЧЕТ СКАЛЬПЕРА С TP/SL:")
    print("=" * 50)
    print(f"💰 Стартовый баланс:         {start_balance} USDT")
    print(f"💵 Финальный баланс:         {balance:.2f} USDT")
    print(f"📈 Чистый Профит:            {net_profit_usdt:.2f} USDT ({profit_percent:.2f}%)")
    print(f"🔄 Закрытых сделок:          {total_trades} (🟢 TP: {tps} | 🔴 SL: {sls})")


if __name__ == "__main__":
    run_lgbm_fixed_backtester()
