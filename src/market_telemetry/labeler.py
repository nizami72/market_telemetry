import pandas as pd


def make_time_machine():
    csv_file = "../../market_data.csv"

    # 1. Загружаем наш собранный датасет
    print(f"📖 Читаем файл {csv_file}...")
    df = pd.read_csv(csv_file)

    if len(df) < 10:
        print("❌ Слишком мало данных в файле. Дай логгеру поработать подольше!")
        return

    print(f"📊 Всего строк найдено: {len(df)}")

    # 2. СТРОИМ МАШИНУ ВРЕМЕНИ
    # Горизонт предсказания: хотим заглянуть на 1 минуту (6 строк по 10 секунд) вперед
    look_ahead_steps = 6

    # .shift(-look_ahead_steps) берет колонку цен и сдвигает ее ВВЕРХ.
    # То есть в строке для 18:00:00 окажется цена, которая физически наступит в 18:01:00.
    df["future_price"] = df["price"].shift(-look_ahead_steps)

    # 3. ФИЗИЧЕСКАЯ РАЗМЕТКА (LABELING)
    # Если будущая цена строго выше текущей — ставим 1 (рост). Иначе 0 (падение или флэт)
    df["label_next_price"] = (df["future_price"] > df["price"]).astype(int)

    # У последних 6 строк не будет будущего (логгер еще не дожил до него),
    # поэтомуpandas запишет туда NaN. Удаляем эти строки, чтобы не путать ИИ.
    df_cleaned = df.dropna(subset=["future_price"]).copy()

    # Удаляем временную колонку future_price, она нам больше не нужна
    df_cleaned = df_cleaned.drop(columns=["future_price"])

    # 4. Сохраняем размеченный датасет в новый файл для обучения
    ready_file = "../../labeled_market_data.csv"
    df_cleaned.to_csv(ready_file, index=False)

    print(f"🎉 Разметка завершена! Чистый датасет сохранен в {ready_file}")
    print(f"🗑️ Удалено крайних строк без будущего: {look_ahead_steps}")
    print(f"📐 Итоговый размер матрицы для ИИ: {df_cleaned.shape}")

    # Выведем превью того, что получилось
    print("\n👀 Первые 5 строк готовой матрицы:")
    print(df_cleaned.head())


if __name__ == "__main__":
    make_time_machine()