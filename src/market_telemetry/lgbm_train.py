import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score

# Отключаем спам-предупреждения, если в фолде попался только один класс
warnings.filterwarnings('ignore', category=UserWarning)

def train_lgbm():
    csv_file = "../../data/multidim_labeled_market_data.csv"
    print(f"📖 Загружаю датасет {csv_file} для LightGBM...")
    df = pd.read_csv(csv_file)

    feature_cols = [
        "imbalance_5", "imbalance_20", "imbalance_50",
        "market_delta_10s", "trade_speed_10s", "speed_zscore",
        "delta_rolling_2m", "delta_rolling_5m", "imb_20_velocity",
        "delta_rolling_30m", "delta_rolling_1h", "price_velocity_15m"
        ,
        # Наш новый Feature Engineering:
        "speed_ratio_1m", "speed_ratio_5m", "speed_ratio_15m",
        "cum_delta_1m", "cum_delta_5m", "cum_delta_15m",
        "price_change_5m", "price_change_1h"
    ]

    X = df[feature_cols].values

    # Сдвигаем классы из [-1, 0, 1] в [0, 1, 2] для LightGBM
    y = df["label_next_price"].values + 1

    tscv = TimeSeriesSplit(n_splits=5)
    print("🏋️‍♂️ Начинаю валидацию LightGBM на временных рядах...")

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

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

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

        # Явно указываем labels=[0, 1, 2], чтобы избежать ошибки размерностей
        loss = log_loss(y_test, preds_proba, labels=[0, 1, 2])

        try:
            auc = roc_auc_score(y_test, preds_proba, multi_class='ovr', labels=[0, 1, 2])
        except ValueError:
            auc = np.nan # Если класс всё-таки один, пишем nan, чтобы не ронять скрипт

        oof_logloss.append(loss)
        oof_auc.append(auc)
        print(f"Fold {fold+1} -> LogLoss: {loss:.4f} | AUC: {auc:.4f} | Trees: {model.best_iteration}")

    print("\n" + "=" * 50)
    # np.nanmean корректно считает среднее, игнорируя 'nan'
    print(f"🎯 СРЕДНИЙ ROC-AUC: {np.nanmean(oof_auc):.4f}")
    print(f"🎯 СРЕДНИЙ LOGLOSS: {np.mean(oof_logloss):.4f}")
    print("=" * 50)

    optimal_trees = int(np.mean(best_iterations))
    print(f"🚀 Обучаю финальный 'мозг' на 100% данных (Оптимальное кол-во деревьев: {optimal_trees})...")

    full_train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, full_train_data, num_boost_round=optimal_trees)

    model_file = "lgbm_market_model.pkl"
    joblib.dump(final_model, model_file)
    print(f"💾 Модель сохранена в файл: {model_file}")

if __name__ == "__main__":
    train_lgbm()
