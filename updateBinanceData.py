import pandas as pd
import os
try:
    fp_root = os.path.join(os.path.split(__file__)[0], 'data')
except NameError:
    fp_root = 'data'
def fp_p(*args):
    return os.path.join(fp_root, *args)

from getBinanceData import update_df, client

df_15m = pd.read_parquet(fp_p('15m.pq'))
df_30m = pd.read_parquet(fp_p('30m.pq'))
df_1h = pd.read_parquet(fp_p('1h.pq'))
df_4h = pd.read_parquet(fp_p('4h.pq'))
df_1d = pd.read_parquet(fp_p('1d.pq'))
df_1w = pd.read_parquet(fp_p('1w.pq'))

tmp = update_df(df_15m, client.KLINE_INTERVAL_15MINUTE, 15)
print(len(tmp))
if len(tmp) > 255:
    fs = os.listdir(fp_p('15m.pq'))
    tmp.to_parquet(fp_p('15m.pq', f'{len(fs):03}.parquet'))

tmp = update_df(df_30m, client.KLINE_INTERVAL_30MINUTE, 30)
print(len(tmp))
if len(tmp) > 255:
    fs = os.listdir(fp_p('30m.pq'))
    tmp.to_parquet(fp_p('30m.pq', f'{len(fs):03}.parquet'))

tmp = update_df(df_1h, client.KLINE_INTERVAL_1HOUR, 60)
print(len(tmp))
if len(tmp) > 255:
    fs = os.listdir(fp_p('1h.pq'))
    tmp.to_parquet(fp_p('1h.pq', f'{len(fs):03}.parquet'))

tmp = update_df(df_4h, client.KLINE_INTERVAL_4HOUR, 60*4)
print(len(tmp))
if len(tmp) > 255:
    fs = os.listdir(fp_p('4h.pq'))
    tmp.to_parquet(fp_p('4h.pq', f'{len(fs):03}.parquet'))

tmp = update_df(df_1d, client.KLINE_INTERVAL_1DAY, 60*24)
print(len(tmp))
if len(tmp) > 255:
    fs = os.listdir(fp_p('1d.pq'))
    tmp.to_parquet(fp_p('1d.pq', f'{len(fs):03}.parquet'))

tmp = update_df(df_1w, client.KLINE_INTERVAL_1WEEK, 60*24*7)
print(len(tmp))
if len(tmp) > 255:
    fs = os.listdir(fp_p('1w.pq'))
    tmp.to_parquet(fp_p('1w.pq', f'{len(fs):03}.parquet'))
