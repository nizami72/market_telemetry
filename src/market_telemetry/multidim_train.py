import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_multidim_ai():
    csv_file = "multidim_labeled_market_data.csv"

    # 1. Загружаем нашу многомерную размеченную матрицу
    print(f"📖 Загружаю размеченный датасет {csv_file}...")
    df = pd.read_csv(csv_file)

    print(f"🧠 Доступно {len(df)} строк телеметрии для обучения.")

    # 2. ВЫДЕЛЯЕМ ВЕКТОР АРГУМЕНТОВ (X) И ЦЕЛЬ (Y)
    # Наш список физических датчиков
   # Включаем как моментальные датчики, так и наши новые контекстные скользящие фичи
    feature_cols = [
        "imbalance_5",
        "imbalance_20",
        "imbalance_50",
        "market_delta_10s",
        "trade_speed_10s",
        "speed_zscore",  # Относительный всплеск скорости
        "delta_rolling_2m",  # Кумулятивный напор за 2 мин
        "delta_rolling_5m",  # Кумулятивный напор за 5 мин
        "imb_20_velocity",  # Скорость изменения стакана
    ]

    X = df[feature_cols]
    y = df["label_next_price"]

    # 3. РАЗДЕЛЯЕМ НА TRAIN / TEST (80% на учебу, 20% на честный экзамен)
    # shuffle=False КРИТИЧЕСКИ ВАЖЕН для временных рядов, чтобы модель не заглядывала в будущее
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # 4. ИНИЦИАЛИЗИРУЕМ АНСАМБЛЬ СЛУЧАЙНОГО ЛЕСА
    # Создаем ансамбль из 100 деревьев решений
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

    print("🏋️‍♂️ Запускаю ансамблевое обучение 100 цифровых инженеров...")
    model.fit(X_train, y_train)
    print("✅ Обучение завершено!")

    # 5. ЭКЗАМЕН: ПРОВЕРЯЕМ ТОЧНОСТЬ
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ МНОГОМЕРНОГО ЭКСПЕРИМЕНТА:")
    print(f"Размер обучающей выборки, points:                   {len(X_train)}")
    print(f"Размер тестовой выборки (экзамен) points:           {len(X_test)}")
    print(f"🎯 ЧЕСТНАЯ ТОЧНОСТЬ ПРЕДСКАЗАНИЯ (Accuracy)%:       {accuracy * 100:.1f}")
    print("=" * 50)

    # 6. ВАЖНОСТЬ ДАТЧИКОВ (Feature Importance)
    # Это покажет, какой физический параметр оказался самым ценным для ИИ
    print("\n🎯 Физическая ценность датчиков по мнению ИИ:")
    importances = model.feature_importances_
    for col, importance in zip(feature_cols, importances):
        print(f"  Датчик [{col:<16}] -> Влияние на результат: {importance * 100:.1f}%")


if __name__ == "__main__":
    train_multidim_ai()