import sys
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import configparser
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings('ignore', category=UserWarning)

def prepare_sliced_dataset(csv_file, config_path="config.ini"):
    """
    Выделенная функция загрузки данных.
    Читает переданный файл и вырезает скользящее или фиксированное окно длиной N дней.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    start_date_str = config.get("MARKET_DATA", "start_date")
    slice_days = config.getint("MARKET_DATA", "slice_days")

    # Жестко локализуем границы в UTC, чтобы они совпадали с тиками Bybit
    start_boundary = pd.to_datetime(start_date_str).tz_localize("UTC")
    end_boundary = start_boundary + pd.Timedelta(days=slice_days)

    print(f"Document loading dataset {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {csv_file} не найден!")
        sys.exit(1)

    # Приводим к datetime. Если логгер пишет строки с таймзоной,
    # pd.to_datetime автоматически сделает колонку tz-aware.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # На всякий случай гарантируем, что вся колонка принудительно в UTC
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # Гарантируем строгую хронологию
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"⏳ Фильтрация окна: с {start_boundary} по {end_boundary} ({slice_days} дней)...")

    # 3. Делаем временной срез данных (теперь оба объекта строго в UTC)
    df_sliced = df[(df["timestamp"] >= start_boundary) & (df["timestamp"] <= end_boundary)].reset_index(drop=True)

    if df_sliced.empty or len(df_sliced) < 100:
        print(f"❌ Критическая ошибка: в интервале дней найдено всего {len(df_sliced)} строк! Проверь даты.")
        return None

    # Полезная статистика для контроля тиков в консоли
    total_len = len(df_sliced)
    print(f"🎉 Новая качественная выборка нарезана!")
    print(f"🎯 Итоговый размер датасета для ИИ (включая флэт-паттерны): {total_len}")
    print(f"📅 Фактические границы выборки: с {df_sliced['timestamp'].min()} по {df_sliced['timestamp'].max()}")

    return df_sliced


def train_lgbm(data_file):
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Оставляем только один файл модели
    model_file = config.get("MARKET_DATA", "model_file_prod", fallback="lgbm_market_model.pkl")

    # Вызываем изолированную функцию выборки данных с переданным файлом
    df = prepare_sliced_dataset(data_file)
    if df is None:
        return

    # Список фичей
    feature_cols = [
        "imbalance_5", "imbalance_20", "imbalance_50",
        "market_delta_10s", "trade_speed_10s", "speed_zscore",
        "delta_rolling_2m", "delta_rolling_5m", "imb_20_velocity",
        "delta_rolling_30m", "delta_rolling_1h", "price_velocity_15m",
        "speed_ratio_1m", "speed_ratio_5m", "speed_ratio_15m",
        "cum_delta_1m", "cum_delta_5m", "cum_delta_15m",
        "price_change_5m", "price_change_1h"
    ]

    X = df[feature_cols].values
    y = df["label_next_price"].values + 1  # Сдвиг классов под LightGBM [0, 1, 2]

    tscv = TimeSeriesSplit(n_splits=5)
    print("🏋️‍♂️ Начинаю кросс-валидацию LightGBM на временных рядах...")

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "max_depth": 7,
        "num_leaves": 45,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1
    }

    oof_logloss = []
    oof_auc = []
    best_iterations = []

    # Кросс-валидация идет по всей выборке без искусственного отсечения 80%
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_fold_train, X_fold_val = X[train_idx], X[test_idx]
        y_fold_train, y_fold_val = y[train_idx], y[test_idx]

        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        valid_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        best_iterations.append(model.best_iteration)
        preds_proba = model.predict(X_fold_val)

        loss = log_loss(y_fold_val, preds_proba, labels=[0, 1, 2])
        try:
            auc = roc_auc_score(y_fold_val, preds_proba, multi_class='ovr', labels=[0, 1, 2])
        except ValueError:
            auc = np.nan

        oof_logloss.append(loss)
        oof_auc.append(auc)
        print(f"Fold {fold+1} -> LogLoss: {loss:.4f} | AUC: {auc:.4f} | Trees: {model.best_iteration}")

    print("\n" + "=" * 50)
    print(f"🎯 СРЕДНИЙ ROC-AUC (Валидация): {np.nanmean(oof_auc):.4f}")
    print(f"🎯 СРЕДНИЙ LOGLOSS (Валидация): {np.mean(oof_logloss):.4f}")
    print("=" * 50)

    optimal_trees = int(np.mean(best_iterations))

    print(f"\n🚀 Сборка единой боевой модели на 100% данных (Оптимальное кол-во деревьев: {optimal_trees})...")
    full_train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, full_train_data, num_boost_round=optimal_trees)

    joblib.dump(final_model, model_file)
    print(f"💾 Модель '{model_file}' успешно создана и готова к деплою.")

if __name__ == "__main__":
    # Проверяем, передан ли аргумент с названием файла при запуске
    if len(sys.argv) < 2:
        print("❌ Ошибка запуска. Использование: python lgbm_train.py <путь_к_файлу_с_данными.csv>")
        sys.exit(1)

    target_csv = sys.argv[1]
    train_lgbm(target_csv)
