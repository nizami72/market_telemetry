import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import configparser
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings('ignore', category=UserWarning)

def run_grid_search(csv_file):
    print(f"📥 Загрузка размеченного массива: {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Файл {csv_file} не найден!")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 🔧 ЗАПЛАТКА ДЛЯ СТРУКТУРЫ ДАННЫХ: Авто-поиск базовой цены Mid-Price
    if "bid" in df.columns and "ask" in df.columns and "mid_price" not in df.columns:
        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        print("💡 [DATA FIXED] Колонка 'mid_price' автоматически собрана из 'bid' и 'ask'.")
    elif "price" in df.columns and "mid_price" not in df.columns:
        df = df.rename(columns={"price": "mid_price"})
        print("💡 [DATA FIXED] Колонка 'price' переименована в 'mid_price'.")
    elif "close" in df.columns and "mid_price" not in df.columns:
        df = df.rename(columns={"close": "mid_price"})
        print("💡 [DATA FIXED] Колонка 'close' переименована в 'mid_price'.")

    if "mid_price" not in df.columns:
        print(f"❌ Критическая ошибка: Не найдена базовая цена! Доступные колонки: {list(df.columns)}")
        sys.exit(1)

    # Целевой список фичей
    all_target_features = [
        "imbalance_5", "imbalance_20", "imbalance_50",
        "market_delta_10s", "trade_speed_10s", "speed_zscore",
        "delta_rolling_2m", "delta_rolling_5m", "imb_20_velocity",
        "delta_rolling_30m", "delta_rolling_1h", "price_velocity_15m",
        "speed_ratio_1m", "speed_ratio_5m", "speed_ratio_15m",
        "cum_delta_1m", "cum_delta_5m", "cum_delta_15m",
        "price_change_5m", "price_change_1h"
    ]

    # 🔧 ДИНАМИЧЕСКИЙ ФИЛЬТР: Скрипт адаптируется под доступные колонки
    feature_cols = [col for col in all_target_features if col in df.columns]
    print(f"📊 Доступно фичей для обучения: {len(feature_cols)} из {len(all_target_features)}")

    if len(feature_cols) == 0:
        print("❌ Критическая ошибка: В файле вообще нет необходимых фичей для обучения!")
        sys.exit(1)

    # Сетка параметров R&D исследования
    time_horizons = [15, 30, 45, 60]  # В минутах реального времени
    price_targets = [150, 250, 350, 450, 600]  # В USDT

    results = []
    print(f"🔬 Запуск сеточного поиска по {len(time_horizons) * len(price_targets)} комбинациям...")

    # Базовые сбалансированные параметры LightGBM
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "max_depth": 7,
        "num_leaves": 45,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"  # 💥 ЗАЩИТА ОТ ДИСБАЛАНСА КЛАССОВ
    }

    for horizon in time_horizons:
        for target in price_targets:
            print(f"⏳ Тест: Горизонт {horizon} мин | Порог {target} USDT", end=" ")

            # 🔧 ЖЕЛЕЗОБЕТОННЫЙ МЭРДЖ: Ищем цену через N минут строго по timestamp, а не по индексам строк
            df_temp = df[["timestamp", "mid_price"]].copy()
            df_temp["target_time"] = df_temp["timestamp"] + pd.Timedelta(minutes=horizon)
            df_temp['orig_index'] = df_temp.index

            df_merged = pd.merge_asof(
                df_temp.sort_values("target_time"),
                df[["timestamp", "mid_price"]].rename(columns={"mid_price": "future_mid"}).sort_values("timestamp"),
                left_on="target_time",
                right_on="timestamp",
                direction="nearest"
            )

            # Возвращаем исходную геометрию и сортировку датасета через orig_index
            df_merged = df_merged.set_index('orig_index').sort_index()

            df["future_mid"] = df_merged["future_mid"]
            df["price_diff"] = df["future_mid"] - df["mid_price"]

            # Трехклассовая разметка под текущий шаг сетки
            conditions = [
                (df["price_diff"] <= -target),
                (df["price_diff"] > -target) & (df["price_diff"] < target),
                (df["price_diff"] >= target)
            ]
            choices = [0, 1, 2]
            df["temp_label"] = np.select(conditions, choices, default=1)

            # Удаляем NaN на краях, которые возникли из-за заглядывания в будущее
            df_clean = df.dropna(subset=["future_mid"]).reset_index(drop=True)

            X = df_clean[feature_cols].values
            y = df_clean["temp_label"].values

            # Валидация на временных рядах 5-Fold TSCV
            tscv = TimeSeriesSplit(n_splits=5)
            fold_aucs = []
            fold_losses = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[test_idx]
                y_train, y_val = y[train_idx], y[test_idx]

                # Защита фолда от критического отсутствия одного из классов
                if len(np.unique(y_val)) < 3 or len(np.unique(y_train)) < 3:
                    continue

                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                model = lgb.train(
                    lgb_params,
                    train_data,
                    num_boost_round=300,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
                )

                preds = model.predict(X_val)

                fold_losses.append(log_loss(y_val, preds, labels=[0, 1, 2]))
                fold_aucs.append(roc_auc_score(y_val, preds, multi_class='ovr', labels=[0, 1, 2]))

            mean_auc = np.mean(fold_aucs) if fold_aucs else np.nan
            mean_loss = np.mean(fold_losses) if fold_losses else np.nan

            print(f"-> ROC-AUC: {mean_auc:.4f} | LogLoss: {mean_loss:.4f}")

            if not np.isnan(mean_auc):
                results.append({
                    "horizon_minutes": horizon,
                    "price_target_usdt": target,
                    "mean_auc": mean_auc,
                    "mean_logloss": mean_loss
                })

    # Сводный отчет
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by="mean_auc", ascending=False).reset_index(drop=True)

    print("\n" + "🏆" * 25)
    print(" 🔥 ИТОГОВЫЙ ТОП КОМБИНАЦИЙ ПО МЕТРИКЕ ROC-AUC (БЕЗ ОШИБОК СДВИГА):")
    print("🏆" * 25)
    print(res_df.to_string(index=False))

    res_df.to_csv("grid_search_report.csv", index=False)
    print("\n💾 Полный лог-отчет сохранен в 'grid_search_report.csv'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Ошибка запуска. Использование: python grid_search_optimizer.py <путь_к_файлу.csv>")
        sys.exit(1)

    run_grid_search(sys.argv[1])
