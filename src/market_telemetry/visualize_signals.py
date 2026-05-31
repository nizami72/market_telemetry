import os
import pandas as pd
import numpy as np
import mplfinance as mpf

def generate_mirror_snapshots():
    market_data_file = "../../data/multidim_market_data.csv"
    trades_log_file = "../../data/trades_log.csv"
    output_dir = "signals_snapshots"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("📖 Загружаю честные логи бэктестера и сырые тики...")
    try:
        df_market = pd.read_csv(market_data_file)
        df_trades = pd.read_csv(trades_log_file)
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Не найдены файлы отчетов. Сначала запусти lgbm_backtester.py! {e}")
        return

    # Приводим время к единому формату
    df_market["timestamp"] = pd.to_datetime(df_market["timestamp"])
    df_trades["entry_time"] = pd.to_datetime(df_trades["entry_time"])
    df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])

    # Готовим маркет-датасет для ресемплинга свечей
    df_market.set_index("timestamp", inplace=True)
    ohlc_df = df_market["price"].resample("1Min").ohlc()
    ohlc_df.dropna(inplace=True)

    print(f"🎯 Найдено {len(df_trades)} честных сделок. Начинаю отрисовку...")

    # Отрисовываем абсолютно все сделки, так как их мало и они реальные
    for t_idx, trade in df_trades.iterrows():
        # Округляем тиковое время сделки до минут, чтобы попасть в сетку свечей
        e_min = trade["entry_time"].floor("1Min")
        x_min = trade["exit_time"].floor("1Min")

        if e_min not in ohlc_df.index or x_min not in ohlc_df.index:
            continue

        loc_entry = ohlc_df.index.get_loc(e_min)
        loc_exit = ohlc_df.index.get_loc(x_min)

        # Строим окно: 40 минут предыстории и 15 минут после выхода
        start_idx = max(0, loc_entry - 40)
        end_idx = min(len(ohlc_df), loc_exit + 15)

        slice_df = ohlc_df.iloc[start_idx:end_idx].copy()

        # Маркеры
        slice_df["long_entry"] = np.nan
        slice_df["short_entry"] = np.nan
        slice_df["exit_marker"] = np.nan

        # Проставляем стрелки строго по направлению из лога бэктестера
        if trade["direction"] == "LONG":
            slice_df.at[e_min, "long_entry"] = slice_df.at[e_min, "low"] * 0.9995
            ap_entry = mpf.make_addplot(slice_df["long_entry"], type="scatter", marker="^", markersize=140, color="green")
            tp_color, sl_color = "green", "red"
        else:
            slice_df.at[e_min, "short_entry"] = slice_df.at[e_min, "high"] * 1.0005
            ap_entry = mpf.make_addplot(slice_df["short_entry"], type="scatter", marker="v", markersize=140, color="red")
            tp_color, sl_color = "red", "green"

        # Точка выхода (желтый круг)
        slice_df.at[x_min, "exit_marker"] = slice_df.at[x_min, "high"] * 1.0005
        ap_exit = mpf.make_addplot(slice_df["exit_marker"], type="scatter", marker="o", markersize=100, color="orange")

        h_lines = [trade["entry_price"], trade["tp_price"], trade["sl_price"]]

        # Цветовое оформление (classic = зеленые свечи профита, mike = красные свечи убытка)
        color_theme = "classic" if trade["result"] == "PROFIT" else "mike"
        setup_style = mpf.make_mpf_style(base_mpf_style=color_theme, gridstyle="--")

        time_str = trade["entry_time"].strftime("%Y-%m-%d_%H-%M")
        filename = os.path.join(output_dir, f"real_trade_{t_idx:02d}_{trade['direction']}_{trade['result']}_{time_str}.png")

        mpf.plot(
            slice_df,
            type="candle",
            style=setup_style,
            addplot=[ap_entry, ap_exit],
            hlines=dict(hlines=h_lines, colors=["blue", tp_color, sl_color], linestyle=["-", "--", "--"], linewidths=[1.2, 1, 1]),
            title=f"True Trade {t_idx:02d} [{trade['direction']}]: {trade['result']} | Entry: ${trade['entry_price']:.1f}",
            ylabel="BTC Price (USDT)",
            savefig=filename
        )

    print(f"✅ Зеркальные скриншоты сгенерированы в '{output_dir}/'")

if __name__ == "__main__":
    generate_mirror_snapshots()
