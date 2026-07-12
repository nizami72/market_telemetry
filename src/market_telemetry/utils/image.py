import re
import matplotlib.pyplot as plt
import pandas as pd

# 1. Чтение лога из файла
parsed_records = []

try:
    with open('bot.log', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("Файл bot.log не найден. Убедитесь, что он лежит в той же папке.")
    exit()

# 2. Парсинг данных парами
for i in range(0, len(lines) - 1, 2):
    prob_line = lines[i]
    btc_line = lines[i+1]

    if "Probabilities" in prob_line and "BTC:" in btc_line:
        time_match = re.search(r'\[(.*?)\]', btc_line)
        btc_match = re.search(r'BTC:\s*([\d\.]+)', btc_line)
        m15_match = re.search(r'M15:\s*\[L:([\d\.]+)\s*\|\s*S:([\d\.]+)\s*\|\s*N:([\d\.]+)\]', prob_line)

        if time_match and btc_match and m15_match:
            parsed_records.append({
                'Time': time_match.group(1),
                'BTC': float(btc_match.group(1)),
                'M15_Long': float(m15_match.group(1)),
                'M15_Short': float(m15_match.group(2))
            })

df = pd.DataFrame(parsed_records)

# Преобразуем время в удобный формат для расчетов
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

# 3. Очистка и сглаживание данных
df['BTC_Smooth'] = df['BTC'].rolling(window=30, min_periods=1).mean()

# Задаем порог для фиксации сигналов бота
THRESHOLD = 0.12
df_long_signals = df[df['M15_Long'] >= THRESHOLD]
df_short_signals = df[df['M15_Short'] >= THRESHOLD]

# 4. Построение графика
fig, ax1 = plt.subplots(figsize=(16, 8))

# Ось цены BTC (Сглаженная линия)
color_btc = '#1f77b4'
ax1.set_xlabel('Время', fontsize=11, labelpad=10)
ax1.set_ylabel('Цена BTC ($) [Сглаженная]', color=color_btc, fontsize=11)
ax1.plot(df['Time'].astype(str), df['BTC_Smooth'], color=color_btc, linewidth=2.5, label='BTC (5m SMA)')
ax1.tick_params(axis='y', labelcolor=color_btc)

# Разреживаем подписи по оси X
x_ticks = range(0, len(df), max(1, len(df) // 15))
ax1.set_xticks(x_ticks)
ax1.set_xticklabels([str(df['Time'].iloc[t]) for t in x_ticks], rotation=45, ha='right')

# Накладываем точки "Сигналов"
ax1.scatter(df_long_signals['Time'].astype(str), df_long_signals['BTC_Smooth'],
            color='#ff7f0e', marker='^', s=120, label=f'Сигнал LONG (Prob >= {THRESHOLD})', zorder=3)

ax1.scatter(df_short_signals['Time'].astype(str), df_short_signals['BTC_Short'] if 'BTC_Short' in df else df_short_signals['BTC_Smooth'],
            color='#2ca02c', marker='v', s=120, label=f'Сигнал SHORT (Prob >= {THRESHOLD})', zorder=3)

# 5. Добавление текстовых лейблов со временем возникновения
for idx, row in df_long_signals.iterrows():
    ax1.annotate(f"{str(row['Time'])}\n(L:{row['M15_Long']})",
                 (str(row['Time']), row['BTC_Smooth']),
                 textcoords="offset points",
                 xytext=(0, 12),  # Смещение текста чуть выше точки
                 ha='center',
                 fontsize=8,
                 fontweight='bold',
                 color='#d35400',
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7, ec="orange"))

for idx, row in df_short_signals.iterrows():
    ax1.annotate(f"{str(row['Time'])}\n(S:{row['M15_Short']})",
                 (str(row['Time']), row['BTC_Smooth']),
                 textcoords="offset points",
                 xytext=(0, -25),  # Смещение текста чуть ниже точки
                 ha='center',
                 fontsize=8,
                 fontweight='bold',
                 color='#27ae60',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="green"))

# Настройка сетки и дизайна
plt.title(f'Анализ лога BTC (Агрегировано {len(df)} записей)', fontsize=14, pad=15)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.legend(loc='upper left', fontsize=10)

fig.tight_layout()

# Сохранение результата
plt.savefig('btc_large_log_chart.png', dpi=300)
plt.show()
