import sys
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import configparser
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
from telegram_alerts import send_telegram_alert_sync

warnings.filterwarnings('ignore', category=UserWarning)

def prepare_sliced_dataset(csv_file, config_path="config.ini"):
    """
    Улучшенная функция загрузки данных.
    Если start_date == 'auto', рассчитывает плавающее окно за последние slice_days.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    start_date_str = config.get("MARKET_DATA", "start_date")
    slice_days = config.getint("MARKET_DATA", "slice_days")

    # 🤖 АВТОМАТИЗАЦИЯ ДЛЯ НОЧНОГО ЗАПУСКА
    if start_date_str.lower() == "auto":
        # Берем текущее время сервера в UTC и отнимаем slice_days
        now_utc = pd.Timestamp.now(tz="UTC")
        start_boundary = (now_utc - pd.Timedelta(days=slice_days)).floor("D")
        end_boundary = now_utc
        print(f"🔄 [AUTO DATE] Обнаружен ночной режим. Авто-расчет окна за последние {slice_days} дней.")
    else:
        # Если в конфиге жесткая дата (например для R&D исследований за июнь)
        start_boundary = pd.to_datetime(start_date_str).tz_localize("UTC")
        end_boundary = start_boundary + pd.Timedelta(days=slice_days)

    print(f"Document loading dataset {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {csv_file} не найден!")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"⏳ Фильтрация скользящего окна: с {start_boundary} по {end_boundary}...")

    # Делаем временной срез данных
    df_sliced = df[(df["timestamp"] >= start_boundary) & (df["timestamp"] <= end_boundary)].reset_index(drop=True)

    if df_sliced.empty or len(df_sliced) < 100:
        print(f"❌ Критическая ошибка: в интервале найдено всего {len(df_sliced)} строк! Проверь базу.")
        return None

    print(f"🎉 Новая качественная выборка нарезана! Размер: {len(df_sliced)} строк.")
    print(f"📅 Границы выборки: с {df_sliced['timestamp'].min()} по {df_sliced['timestamp'].max()}")

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
    # Notify vis Telegram
    mean_auc = np.nanmean(oof_auc)
    mean_logloss = np.mean(oof_logloss)
    min_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
    max_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M')

    # Чистый MarkdownV2 с моноширинным блоком данных
    alert_text = (
        "🤖 *MLOps Pipeline: Модель успешно обновлена\\!*\n\n"
        "\n"
        f"Период:    c {min_date} по {max_date} UTC\n"
        f"Выборка:   {len(df):,} строк\n"
        f"Ансамбль:  {optimal_trees} деревьев\n\n"
        "Метрики кросс-валидации (5-Fold TSCV):\n"
        f"• Mean ROC-AUC:  {mean_auc:.4f}\n"
        f"• Mean LogLoss:  {mean_logloss:.4f}\n\n"
        "Статус:    Файл .pkl перезаписан на VPS.\n"
        "\n"
        "⚙️ `paper_trader.py` переключился на новые веса\\."
    )
    send_telegram_alert_sync(alert_text)
    print(f"💾 Модель '{model_file}' успешно создана и готова к деплою.")

if __name__ == "__main__":
    # Проверяем, передан ли аргумент с названием файла при запуске
    if len(sys.argv) < 2:
        print("❌ Ошибка запуска. Использование: python lgbm_train.py <путь_к_файлу_с_данными.csv>")
        sys.exit(1)

    target_csv = sys.argv[1]
    train_lgbm(target_csv)
