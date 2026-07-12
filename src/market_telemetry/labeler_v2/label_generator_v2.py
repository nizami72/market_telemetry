import numpy as np
import pandas as pd

LABEL_FLAT = 0
LABEL_FALL = 1
LABEL_RISE = 2

def generate_labels(
        df: pd.DataFrame,
        horizon_rows: int,
        tp_usdt: float,
        sl_usdt: float = None
):
    """
    Генерация label по принципу:
        - сначала достигнут TP -> RISE
        - сначала достигнут SL -> FALL
        - ничего не произошло -> FLAT

    Parameters
    ----------
    df : DataFrame
        Должен содержать колонку price

    horizon_rows : int
        Например 90 строк = 15 минут

    tp_usdt : float
        Размер Take Profit

    sl_usdt : float
        Размер Stop Loss.
        Если None -> равен tp_usdt
    """

    if sl_usdt is None:
        sl_usdt = tp_usdt

    prices = df["price"].values
    labels = np.full(len(df), np.nan)

    for i in range(len(df) - horizon_rows):

        entry = prices[i]
        label = LABEL_FLAT

        for j in range(i + 1, i + horizon_rows + 1):

            delta = prices[j] - entry
            if delta >= tp_usdt:
                label = LABEL_RISE
                break

            if delta <= -sl_usdt:
                label = LABEL_FALL
                break

        labels[i] = label

    df = df.copy()
    df["label"] = labels

    return df.dropna(subset=["label"])