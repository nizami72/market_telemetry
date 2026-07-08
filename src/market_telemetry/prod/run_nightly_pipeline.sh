#!/bin/bash

# Переходим в рабочую директорию проекта
cd /root/projects/market_telemetry || exit 1

# Активируем наше изолированное виртуальное окружение python
source venv/bin/activate

echo "⏰ [$(date)] Запуск ночного MLOps-конвейера..."

# 1. Запуск разметчика (передаем путь к сырому файлу)
echo "⚡ Шаг 1: Начинаю генерацию макро-фичей и разметку..."
python3 labeler.py ../../multidim_market_data.csv

if [ $? -ne 0 ]; then
    echo "❌ Ошибка на этапе разметки данных! Пайплайн остановлен."
    exit 1
fi

# 2. Запуск обучения LightGBM
echo "🏋️‍♂️ Шаг 2: Запускаю переобучение модели LightGBM..."
python3 lgbm_train.py ../../multidim_market_data_labeled.csv

if [ $? -ne 0 ]; then
    echo "❌ Ошибка на этапе обучения модели! Пайплайн остановлен."
    exit 1
fi

echo "✅ [$(date)] Пайплайн успешно завершен. Модель lgbm_market_model.pkl обновлена!"