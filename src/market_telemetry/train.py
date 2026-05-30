import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_ai():
    csv_file = "../../labeled_market_data.csv"

    # 1. Загружаем размеченную матрицу
    df = pd.read_csv(csv_file)

    # Нам нужно хотя бы немного строк для разделения выборки
    if len(df) < 10:
        print(
            "❌ Матрица слишком мала для обучения. Дай логгеру собрать больше данных!"
        )
        return

    print(f"🧠 Загружено {len(df)} точек телеметрии для обучения ИИ.")

    # 2. РАЗДЕЛЯЕМ ДАННЫЕ НА АРГУМЕНТЫ (X) И ЦЕЛЬ (Y)
    # X — это наши датчики (аргументы). Пока у нас только один: imbalance_20
    # Мы передаем его как список списков [[]], так как модель требует двумерную матрицу
    X = df[["imbalance_20"]]

    # Y — это чистый физический исход (выросло=1, упало=0)
    y = df["label_next_price"]

    # 3. РАЗДЕЛЯЕМ НА УЧЕБНЫЙ И ТЕСТОВЫЙ НАБОРЫ (Train/Test Split)
    # Чтобы проверить честность ИИ, мы обучим его на 80% данных,
    # а на оставшихся 20% проверим его точность. Эти 20% он никогда не видел.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # 4. ИНИЦИАЛИЗИРУЕМ И ОБУЧАЕМ МОДЕЛЬ
    # Используем Дерево Решений (Decision Tree) — это базовый, но очень наглядный алгоритм
    model = DecisionTreeClassifier(max_depth=3)

    print("🏋️‍♂️ Запускаю процесс калибровки весов (обучение модели)...")
    model.fit(X_train, y_train)
    print("✅ Калибровка завершена!")

    # 5. ПРОВЕРЯЕМ ТОЧНОСТЬ (МЕТРИКА РОСТА)
    # Просим модель предсказать исходы для тестовых данных, которые она не видела
    predictions = model.predict(X_test)

    # Считаем процент правильных попаданий (Accuracy)
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА:")
    print(f"Размер обучающей выборки: {len(X_train)} строк")
    print(f"Размер тестовой выборки:  {len(X_test)} строк")
    print(f"🎯 ТОЧНОСТЬ ПРЕДСКАЗАНИЯ (Accuracy): {accuracy * 100:.1f}%")
    print("=" * 50)

    # Физическая интерпретация того, что поняла модель
    # Покажем, при каком значении имбаланса дерево делит рынок
    print("\n💡 Логика, которую вывела модель из твоих данных:")
    threshold = model.tree_.threshold[0]
    if threshold != -2:  # Если дерево смогло найти разделение
        print(
            f"  Если Imbalance_20 > {threshold:.3f} -> модель прогнозирует один исход, если меньше -> другой."
        )
    else:
        print("  Данных слишком мало, модель пока не нашла четкой границы.")


if __name__ == "__main__":
    train_ai()