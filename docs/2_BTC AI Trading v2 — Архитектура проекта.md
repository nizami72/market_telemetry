# BTC AI Trading v2 — Архитектура проекта

## Общая концепция

Система представляет собой конвейер машинного обучения, который ежедневно переобучает каскад моделей на последних 30 днях истории рынка Bitcoin.

Каждая модель специализируется на своем временном горизонте и отвечает на один и тот же вопрос:

> **Какова вероятность того, что в течение заданного временного горизонта первым будет достигнут уровень Take Profit в направлении LONG, SHORT или не будет достигнут ни один из уровней?**

Система состоит из независимых модулей, каждый из которых выполняет одну строго определенную задачу.

---

# Архитектура

```
Биржа
    │
    ▼
Collector
    │
    ▼
Feature Store
    │
    ▼
TP Optimizer
    │
    ▼
Label Generator
    │
    ▼
Dataset Builder
    │
    ▼
Model Trainer
    │
    ▼
Model Repository
    │
    ▼
Prediction Engine
    │
    ▼
Trading Bot
```

---

# Компоненты системы

## 1. Collector

Назначение:

Сбор рыночных данных с биржи.

Результат:

Формирование непрерывного потока 10-секундных свечей OHLC и вычисление рыночных признаков.

Выход:

Feature Store.

---

## 2. Feature Store

Источник истины.

Хранит исключительно рыночные данные.

Не содержит никаких Label.

Содержимое:

* timestamp
* open
* high
* low
* close
* imbalance_5
* imbalance_20
* imbalance_50
* market_delta_10s
* trade_speed_10s
* остальные вычисленные признаки

Feature Store никогда не изменяется алгоритмами обучения.

---

## 3. TP Optimizer

Запускается один раз перед каждым ежедневным обучением.

Использует:

последние 30 дней Feature Store.

Для каждого горизонта:

* M15
* M30
* M45
* M60

строит статистику движения рынка и автоматически определяет оптимальное значение TP.

Полученные значения сохраняются в Meta-файлах.

---

## 4. Label Generator

Использует:

* Feature Store
* TP выбранный TP Optimizer

Для каждой записи:

* определяет направление LONG / SHORT / FLAT / UNKNOWN;
* вычисляет время достижения цели;
* создает отдельный Label Store для каждого горизонта.

Feature Store при этом не изменяется.

---

## 5. Label Store

Хранится отдельно от Feature Store.

Для каждого горизонта существует собственный файл.

Например:

* m15.parquet
* m30.parquet
* m45.parquet
* m60.parquet

Каждый файл содержит:

* timestamp
* label
* hit_after_rows

Отдельно хранится Meta-файл.

---

## 6. Dataset Builder

Объединяет:

Feature Store

*

Label Store

по timestamp.

Во время сборки:

* исключаются UNKNOWN;
* формируется итоговая обучающая выборка;
* подготавливаются признаки и целевые значения.

После обучения промежуточный Dataset может быть удален.

---

## 7. Model Trainer

Получает:

готовый Dataset.

Выполняет:

* обучение модели;
* кросс-валидацию;
* подбор количества деревьев;
* расчет метрик качества.

Для каждого горизонта обучается собственная независимая модель.

---

## 8. Model Repository

Хранит:

* обученные модели;
* параметры обучения;
* Meta-информацию.

Каждая модель связана со своим Meta-файлом.

---

## 9. Prediction Engine

Используется во время торговли.

Получает:

текущие признаки рынка.

Загружает:

соответствующую модель и ее Meta-файл.

Возвращает вероятности классов:

* LONG
* SHORT
* FLAT

---

## 10. Trading Bot

Использует вероятности модели.

При превышении установленного порога:

* открывает позицию;
* сразу выставляет TP;
* сразу выставляет SL.

Закрытие позиции полностью выполняется биржей.

Модель не сопровождает открытую сделку.

---

# Структура каталогов

```
btc-ai-trading/
│
├── config/
│   ├── config.ini
│   └── agents.json
│
├── data/
│   ├── raw/
│   │   └── market_ohlc.parquet
│   │
│   ├── features/
│   │   └── feature_store.parquet
│   │
│   ├── labels/
│   │   ├── m15.parquet
│   │   ├── m15_meta.json
│   │   ├── m30.parquet
│   │   ├── m30_meta.json
│   │   ├── m45.parquet
│   │   ├── m45_meta.json
│   │   ├── m60.parquet
│   │   └── m60_meta.json
│   │
│   └── datasets/
│       ├── train_m15.parquet
│       ├── train_m30.parquet
│       ├── train_m45.parquet
│       └── train_m60.parquet
│
├── models/
│   ├── lgbm_m15.pkl
│   ├── lgbm_m30.pkl
│   ├── lgbm_m45.pkl
│   ├── lgbm_m60.pkl
│   ├── lgbm_m15_meta.json
│   ├── lgbm_m30_meta.json
│   ├── lgbm_m45_meta.json
│   └── lgbm_m60_meta.json
│
├── logs/
│   ├── label_generator.log
│   ├── training.log
│   └── prediction.log
│
├── scripts/
│   ├── feature_builder.py
│   ├── tp_optimizer.py
│   ├── label_generator.py
│   ├── dataset_builder.py
│   ├── train.py
│   ├── predict.py
│   └── backtest.py
│
├── docs/
│   ├── PROJECT_SPECIFICATION.md
│   ├── DATASET_SPECIFICATION.md
│   ├── LABEL_GENERATOR.md
│   ├── TRAINING_PIPELINE.md
│   └── IDEAS.md
│
├── tests/
│   ├── test_label_generator.py
│   ├── test_tp_optimizer.py
│   └── test_training.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Последовательность ежедневного обучения

```
Collector

↓

Обновление Feature Store

↓

TP Optimizer

↓

Label Generator

↓

Dataset Builder

↓

Model Trainer

↓

Model Repository
```

---

# Последовательность работы в реальном времени

```
Collector

↓

Feature Engineering

↓

Prediction Engine

↓

Trading Bot

↓

Биржа
```

---

# Независимость компонентов

Каждый модуль является независимым.

Изменение:

* алгоритма Label Generator;
* способа выбора TP;
* модели машинного обучения;
* структуры Feature Engineering;

не должно требовать изменения остальных компонентов системы.

Все взаимодействие между модулями осуществляется исключительно через файлы данных и Meta-файлы.

Это позволяет независимо тестировать, заменять и совершенствовать любой компонент без изменения общей архитектуры проекта.

