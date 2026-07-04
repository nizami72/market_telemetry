import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import configparser
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings('ignore', category=UserWarning)

def prepare_sliced_dataset(config_path="config.ini"):
    """
    Выделенная функция загрузки данных.
    Вырезает скользящее или фиксированное окно длиной N дней, начиная с даты d.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    csv_file = config.get("MARKET_DATA", "csv_file")
    start_date_str = config.get("MARKET_DATA", "start_date")
    slice_days = config.getint("MARKET_DATA", "slice_days")

    print(f"辨 Загружаю датасет {csv_file}...")
    df = pd.read_csv(csv_file)

    # Приводим к datetime и гарантируем строгую хронологию
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Точка старта d
    start_boundary = pd.to_datetime(start_date_str)

    # 2. Вычисляем точку окончания: start_date + N дней
    end_boundary = start_boundary + pd.Timedelta(days=slice_days)

    print(f"⏳ Фильтрация окна: с {start_boundary} по {end_boundary} ({slice_days} дней)...")

    # 3. Делаем временной срез данных
    df_sliced = df[(df["timestamp"] >= start_boundary) & (df["timestamp"] <= end_boundary)].reset_index(drop=True)

    if df_sliced.empty or len(df_sliced) < 100:
        print(f"❌ Критическая ошибка: в интервале дней найдено всего {len(df_sliced)} строк! Проверь даты.")
        return None

    # Полезная статистика для контроля тиков в консоли
    total_ticks = len(df_sliced)
    print(f"✅ Срез сформирован. Получено строк: {total_ticks}")
    print(f"📅 Фактические границы выборки: с {df_sliced['timestamp'].min()} по {df_sliced['timestamp'].max()}")

    return df_sliced


def train_lgbm():
    config = configparser.ConfigParser()
    config.read("config.ini")

    model_file_prod = config.get("MARKET_DATA", "model_file_prod")
    model_file_test = config.get("MARKET_DATA", "model_file_test")

    # Вызываем нашу новую изолированную функцию выборки данных
    df = prepare_sliced_dataset()
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

    # Разделение на тренировочную выборку (80%) для кросс-валидации
    train_size = int(len(X) * 0.8)
    X_train_80, y_train_80 = X[:train_size], y[:train_size]

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

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_80)):
        X_fold_train, X_fold_val = X_train_80[train_idx], X_train_80[test_idx]
        y_fold_train, y_fold_val = y_train_80[train_idx], y_train_80[test_idx]

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

    print(f"\n🛡️ Сборка модели для БЭКТЕСТЕРА (80% выборки)...")
    backtest_dataset = lgb.Dataset(X_train_80, label=y_train_80)
    backtest_model = lgb.train(params, backtest_dataset, num_boost_round=optimal_trees)
    joblib.dump(backtest_model, model_file_test)
    print(f"💾 Файл '{model_file_test}' успешно создан.")

    print(f"\n🚀 Сборка финальной модели для ПРОДАКШЕНА (100% выборки)...")
    full_train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, full_train_data, num_boost_round=optimal_trees)
    joblib.dump(final_model, model_file_prod)
    print(f"💾 Файл '{model_file_prod}' успешно создан.")

if __name__ == "__main__":
    train_lgbm()
