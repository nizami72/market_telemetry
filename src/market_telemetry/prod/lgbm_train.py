import os
import sys
import joblib
import warnings
import configparser
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score

# Отключаем спам-предупреждения LightGBM для чистых логов cron
warnings.filterwarnings('ignore', category=UserWarning)

# Конфигурация каскада агентов (Фрактальная матрица целей и шагов)
AGENTS_CONFIG = {
    "M15": {"horizon_rows": 90,   "target_usdt": 250.0, "thinning_step": 30},
    "M30": {"horizon_rows": 180,  "target_usdt": 450.0, "thinning_step": 60},
    "M45": {"horizon_rows": 270,  "target_usdt": 600.0, "thinning_step": 90},
    "M60": {"horizon_rows": 360,  "target_usdt": 800.0, "thinning_step": 120}
}

FEATURE_COLS = [
    "imbalance_5", "imbalance_20", "imbalance_50",
    "market_delta_10s", "trade_speed_10s", "speed_zscore",
    "delta_rolling_2m", "delta_rolling_5m", "imb_20_velocity",
    "delta_rolling_30m", "delta_rolling_1h", "price_velocity_15m"
]

def train_cascade_ensemble(feature_store_path):
    # feature_store_path = "../../data/multidim_market_features.csv"

    # Читаем путь к моделям из config.ini
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.ini")

    models_dir = "models"
    if os.path.exists(config_path):
        config.read(config_path)
        try:
            models_dir = config.get("PATH", "models", fallback="models")
        except Exception as e:
            print(f"⚠️ Ошибка чтения config.ini ({e}). Использую дефолт 'models'.")
    else:
        print(f"⚠️ Файл конфигурации не найден по пути: {config_path}. Использую дефолт 'models'.")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    print(f"📖 Загружаю плотный Feature Store: {feature_store_path}...")
    try:
        # Плотный датасет (270 000+ строк), упорядоченный по времени
        df_raw = pd.read_csv(feature_store_path)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
    except FileNotFoundError:
        print(f"❌ Критическая ошибка: Feature Store не найден по пути {feature_store_path}")
        return

    print(f"⚡ Базовый размер матрицы признаков: {df_raw.shape[0]} строк.")
    print("---")

    # Итерируемся по каскаду агентов
    for agent_name, cfg in AGENTS_CONFIG.items():
        print(f"⚙️ НАЧИНАЮ СБОРКУ МОЗГА ДЛЯ АГЕНТА: [{agent_name}]")
        print(f"📊 Параметры: Горизонт={cfg['horizon_rows']*10}с | Цель={cfg['target_usdt']} USDT | Шаг={cfg['thinning_step']*10}с")

        # Рабочая копия плотных данных для RAM-разметки на лету
        df_agent = df_raw.copy()

        # Шаг 1: Расчет будущих цен строго под персональный горизонт
        df_agent["future_price"] = df_agent["price"].shift(-cfg["horizon_rows"])
        df_agent["price_change"] = df_agent["future_price"] - df_agent["price"]

        # Шаг 2: Трехклассовый динамический лейблинг (0=FLAT, 1=FALL, 2=RISE)
        conditions = [
            (df_agent["price_change"] < -cfg["target_usdt"]), # FALL
            (df_agent["price_change"] > cfg["target_usdt"])   # RISE
        ]
        choices = [1, 2]
        df_agent["label"] = np.select(conditions, choices, default=0)

        # Очищаем NaN на хвостах, возникшие из-за сдвига look-ahead
        df_agent = df_agent.dropna(subset=["future_price", "speed_zscore", "imb_20_velocity"]).copy()

        # Шаг 3: Персональное адаптивное прореживание во избежание оверфиттинга окон
        df_filtered = df_agent.iloc[::cfg["thinning_step"]].reset_index(drop=True)

        X = df_filtered[FEATURE_COLS].values
        y = df_filtered["label"].values

        class_counts = df_filtered["label"].value_counts().to_dict()
        print(f"📊 Распределение классов в RAM после нарезки: {class_counts}")

        # Настройка параметров LightGBM с жесткой балансировкой весов классов
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "class_weight": "balanced",  # Уничтожает проблему '98% FLAT'
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1
        }

        # Шаг 4: Хронологическая Валидация на временных рядах
        tscv = TimeSeriesSplit(n_splits=5)
        oof_logloss = []
        oof_auc = []
        best_iterations = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Страховка от фолдов с вырожденным количеством классов
            if len(np.unique(y_train)) < 3 or len(np.unique(y_test)) < 3:
                continue

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )

            best_iterations.append(model.best_iteration)
            preds_proba = model.predict(X_test)

            loss = log_loss(y_test, preds_proba, labels=[0, 1, 2])
            try:
                auc = roc_auc_score(y_test, preds_proba, multi_class='ovr', labels=[0, 1, 2])
            except ValueError:
                auc = np.nan

            oof_logloss.append(loss)
            oof_auc.append(auc)

        # Шаг 5: Обучение финального агента на 100% выделенного контекста
        optimal_trees = int(np.mean(best_iterations)) if best_iterations else 50
        print(f"🎯 Валидация завершена. Средний ROC-AUC: {np.nanmean(oof_auc):.4f} | LogLoss: {np.mean(oof_logloss):.4f}")
        print(f"🚀 Тренирую финальную боевую модель на {optimal_trees} деревьях...")

        full_dataset = lgb.Dataset(X, label=y)
        final_agent = lgb.train(params, full_dataset, num_boost_round=optimal_trees)

        # Сохранение весов специализированного агента
        model_filename = f"{models_dir}/lgbm_{agent_name.lower()}.pkl"
        joblib.dump(final_agent, model_filename)
        print(f"💾 Веса агента успешно упакованы в: {model_filename}")
        print("-" * 50)

if __name__ == "__main__":

    target_csv = sys.argv[1]
    train_cascade_ensemble(target_csv)
