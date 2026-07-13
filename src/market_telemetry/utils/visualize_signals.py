import os
import shutil
import pandas as pd
import numpy as np
import mplfinance as mpf

def generate_mirror_snapshots():
    market_data_file = "../../data/multidim_market_data.csv"
    trades_log_file = "../../../data/trades_log.csv"
    output_dir = "../signals_snapshots"

    # ==========================================
    # 🧹 ШАГ ОЧИСТКИ (CLEAN UP): Удаляем старые рендеры
    # ==========================================
    if os.path.exists(output_dir):
        print(f"🗑️ Обнаружена старая папка '{output_dir}'. Очищаю от старых чартов...")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    print("📖 Загружаю честные логи бэктестера и сырые тики...")
    try:
        df_market = pd.read_csv(market_data_file)
        df_trades = pd.read_csv(trades_log_file)
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Не найдены файлы отчетов. Сначала запусти tester.py! {e}")
        return

    if df_trades.empty:
        print("⚠️ Лог сделок пуст. Нечего отрисовывать.")
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

    # Отрисовываем абсолютно все сделки
    for t_idx, trade in df_trades.iterrows():
        # Округляем тиковое время сделки до минут, чтобы попасть в сетку свечей
        e_min = trade["entry_time"].floor("1Min")
        x_min = trade["exit_time"].floor("1Min")

        # Находим временные рамки окна (40 минут ДО входа и 15 минут ПОСЛЕ выхода)
        try:
            loc_entry = ohlc_df.index.get_loc(e_min)
            loc_exit = ohlc_df.index.get_loc(x_min)
        except KeyError:
            # Предохранитель: если точной минуты нет в сетке OHLC, ищем ближайшую физическую строку
            loc_entry = ohlc_df.index.get_indexer([e_min], method="nearest")[0]
            loc_exit = ohlc_df.index.get_indexer([x_min], method="nearest")[0]

        start_idx = max(0, loc_entry - 40)
        end_idx = min(len(ohlc_df), loc_exit + 15)

        # Вырезаем кусок рынка под конкретную сделку
        slice_df = ohlc_df.iloc[start_idx:end_idx].copy()

        # Защита: если кусок пустой, пропускаем итерацию
        if slice_df.empty:
            continue

        # Инициализируем массивы под кастомные маркеры на графике
        slice_df["long_entry"] = np.nan
        slice_df["short_entry"] = np.nan
        slice_df["exit_marker"] = np.nan

        # Находим точные ключи времени внутри нашего слайса, чтобы маркеры встали без сдвигов
        actual_e_time = slice_df.index[slice_df.index.get_indexer([e_min], method="nearest")[0]]
        actual_x_time = slice_df.index[slice_df.index.get_indexer([x_min], method="nearest")[0]]

        # Жёсткий стандарт цветов: Тейк всегда Зеленый, Стоп всегда Красный!
        tp_color, sl_color = "green", "red"

        # Проставляем стрелки входа строго по направлению из лога бэктестера
        if trade["direction"] == "LONG":
            slice_df.at[actual_e_time, "long_entry"] = slice_df.at[actual_e_time, "low"] * 0.9995
            ap_entry = mpf.make_addplot(slice_df["long_entry"], type="scatter", marker="^", markersize=140, color="green")
        else:
            slice_df.at[actual_e_time, "short_entry"] = slice_df.at[actual_e_time, "high"] * 1.0005
            ap_entry = mpf.make_addplot(slice_df["short_entry"], type="scatter", marker="v", markersize=140, color="red")

        # ЮВЕЛИРНАЯ ТОЧНОСТЬ: Оранжевый шарик встает строго на цену исполнения из лога
        # ИСПРАВЛЕНО: Привязываем шарик к математической линии цели, а не к прыгающему тику
        if "TAKE_PROFIT" in trade["result"] or trade["result"] == "PROFIT":
            # Если профит — сажаем шарик строго на линию Тейка
            slice_df.at[actual_x_time, "exit_marker"] = trade["tp_price"]
        elif "STOP_LOSS" in trade["result"] or trade["result"] == "LOSS":
            # Если стоп — сажаем шарик строго на линию Стопа
            slice_df.at[actual_x_time, "exit_marker"] = trade["sl_price"]
        else:
            # Для FORCE CLOSE или ручного переворота оставляем реальную цену
            slice_df.at[actual_x_time, "exit_marker"] = trade["exit_price"]
        ap_exit = mpf.make_addplot(slice_df["exit_marker"], type="scatter", marker="o", markersize=120, color="orange")

        # Координаты горизонтальных уровней сделки
        h_lines = [trade["entry_price"], trade["tp_price"], trade["sl_price"]]

        # Цветовое оформление (classic = зеленые свечи, если закрылись в профит; mike = красные свечи, если убыток)
        color_theme = "classic" if trade["result"] == "PROFIT" else "mike"
        setup_style = mpf.make_mpf_style(base_mpf_style=color_theme, gridstyle="--")

        time_str = trade["entry_time"].strftime("%Y-%m-%d_%H-%M")
        filename = os.path.join(output_dir, f"real_trade_{t_idx:02d}_{trade['direction']}_{trade['result']}_{time_str}.png")

        # Строим финальный график через движок matplotlib/mplfinance
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

    print(f"\n✅ Папка полностью обновлена! Зеркальные скриншоты сгенерированы в '{output_dir}/'")

if __name__ == "__main__":
    generate_mirror_snapshots()
