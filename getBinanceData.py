from binance.client import Client, HistoricalKlinesType
from datetime import datetime
import pandas as pd
import os
try:
    fp_root = os.path.join(os.path.split(__file__)[0], 'data')
except NameError:
    fp_root = 'data'
def fp_p(*args):
    path = fp_root
    for x in args[:-1]:
        path = os.path.join(path, x)
        if not os.path.exists(path):
            os.mkdir(path)
    return os.path.join(path, args[-1])

proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}
client = Client(requests_params={'proxies': proxies})

# 输入您要获取历史数据的虚拟货币
symbol = 'ETHUSDC'

def get_klines(start_date, end_date=None, t=None):
    klines = client.get_historical_klines(
        symbol, t,
        start_date, end_date,
        klines_type = HistoricalKlinesType.FUTURES
    )

    df = pd.DataFrame(
        klines,
        columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ]
    )

    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    df = df.drop([
        'quote_asset_volume',
        'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ], axis=1)

    cols = ['open', 'close', 'low', 'high', 'volume']
    for col in cols:
        df[col] = df[col].astype(float)
    return df

def clean_df_rm(df, min_ms = 3599999):
    if len(df) < 1:
        return df
    last_row = df.iloc[-1]
    if datetime.now().timestamp() < last_row['close_time']:
        print(datetime.now().timestamp(), last_row['close_time'])
        return df.iloc[:-1]
    diff = last_row['close_time'] - last_row['timestamp']
    df = df.drop(['close_time', 'timestamp'], axis=1)
    if diff < min_ms:
        return df.iloc[:-1]
    else:
        return df

def get_last_time(df):
    dt = pd.to_datetime(df.iloc[-1]['date'])
    return int(dt.value // 10**6)

def update_df(df, t=None, t_factor=15):
    sub_df = get_klines(
        start_date = get_last_time(df) + 1,
        t = t
    )
    return clean_df_rm(sub_df, 1000 * 60 * t_factor - 1)

if __name__ == '__main':
    # 15分钟K线
    df = get_klines(
        start_date = '2024-01-04',
        end_date = '2025-07-21',
        t = client.KLINE_INTERVAL_15MINUTE
    )

    df = clean_df_rm(df, 1000 * 60 * 15-1)

    df.to_parquet(fp_p('15m.pq', '000.parquet'))

    # 30分钟K线
    df = get_klines(
        start_date = '2024-01-04',
        end_date = '2025-07-21',
        t = client.KLINE_INTERVAL_30MINUTE
    )

    df = clean_df_rm(df, 1000 * 60 * 30-1)

    df.to_parquet(fp_p('30m.pq', '000.parquet'))

    # 1小时K线
    df = get_klines(
        start_date = '2024-01-04',
        end_date = '2025-07-21',
        t = client.KLINE_INTERVAL_1HOUR
    )

    df = clean_df_rm(df, 1000 * 60 * 60-1)

    df.to_parquet(fp_p('1h.pq', '000.parquet'))

    # 4小时K线
    df = get_klines(
        start_date = '2024-01-04',
        end_date = '2025-07-21',
        t = client.KLINE_INTERVAL_4HOUR
    )

    df = clean_df_rm(df, 1000 * 60 * 60 * 4 -1)

    df.to_parquet(fp_p('4h.pq', '000.parquet'))

    # 日线
    df = get_klines(
        start_date = '2024-01-04',
        end_date = '2025-07-21',
        t = client.KLINE_INTERVAL_1DAY
    )

    df = clean_df_rm(df, 1000 * 60 * 60 * 24 -1)

    df.to_parquet(fp_p('1d.pq', '000.parquet'))

    # 周线
    df = get_klines(
        start_date = '2024-01-04',
        end_date = '2025-07-21',
        t = client.KLINE_INTERVAL_1WEEK
    )

    df = clean_df_rm(df, 1000 * 60 * 60 * 24 * 7 -1)

    df.to_parquet(fp_p('1w.pq', '000.parquet'))
