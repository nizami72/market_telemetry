# Создаем продвинутый скрипт обучения lgbm_train.py
# Changing the model
# Этот скрипт не просто обучит LightGBM, но и сохранит обученный «мозг» в файл lgbm_market_model.pkl, чтобы бэктестер мог его использовать.
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_lgbm():
    csv_file = "multidim_labeled_market_data.csv"

    print(f"📖 Загружаю датасет {csv_file} для LightGBM...")
    df = pd.read_csv(csv_file)

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

    # Хронологическое разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print("🏋️‍♂️ Начинаю градиентный бустинг LightGBM...")

    # Настройки для бинарной классификации с выводом вероятностей
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 31,
        "verbose": -1,
        "random_state": 42,
    }

    # Переводим данные в нативный формат LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)

    # Обучаем модель (100 итераций/деревьев последовательного улучшения)
    model = lgb.train(params, train_data, num_boost_round=100)
    print("✅ LightGBM успешно обучен!")

    # Тестируем (модель выдает вероятности от 0.0 до 1.0)
    preds_proba = model.predict(X_test)

    # Для базовой метрики переводим в 0 и 1 по стандартному порогу 0.5
    preds_binary = [1 if p > 0.5 else 0 for p in preds_proba]

    accuracy = accuracy_score(y_test, preds_binary)
    print("\n" + "=" * 50)
    print(f"🎯 БАЗОВАЯ ТОЧНОСТЬ LIGHTGBM (Порог 0.5): {accuracy * 100:.1f}%")
    print("=" * 50)

    # Сохраняем модель на диск
    model_file = "lgbm_market_model.pkl"
    joblib.dump(model, model_file)
    print(f"💾 Модель сохранена в файл: {model_file}")


if __name__ == "__main__":
    train_lgbm()