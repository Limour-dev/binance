from binance.client import Client, HistoricalKlinesType
import pandas as pd
import os
fp_root = os.path.join(os.path.split(__file__)[0], 'data')
def fp_p(*args):
    return os.path.join(fp_root, *args)

proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}
client = Client(requests_params={'proxies': proxies})

# 输入您要获取历史数据的虚拟货币
symbol = 'ETHUSDC'

# 输入您要获取历史数据的时间范围
start_date = '2025-07-20'
end_date = '2025-07-21'
# 获取历史数据

klines = client.get_historical_klines(
    symbol,
    client.KLINE_INTERVAL_1HOUR,
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
    'timestamp', 'close_time', 'quote_asset_volume',
    'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
], axis=1)

df.to_parquet(fp_p('1h.pq'))