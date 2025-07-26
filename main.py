import pandas as pd
import os
try:
    fp_root = os.path.join(os.path.split(__file__)[0], 'data')
except NameError:
    fp_root = 'data'
def fp_p(*args):
    return os.path.join(fp_root, *args)

from getBinanceData import update_df, client

import matplotlib.ticker as mticker
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')

# 设置mplfinance的蜡烛颜色，up为阳线颜色，down为阴线颜色
my_color = mpf.make_marketcolors(
    up='g',
    down='r',
    edge='inherit',
    wick='inherit',
    volume='inherit'
)

# 设置图表的背景色
my_style = mpf.make_mpf_style(
    marketcolors=my_color,
    gridstyle='--',
    y_on_right=True,
    figcolor='(0.82, 0.83, 0.85)',
    gridcolor='(0.82, 0.83, 0.85)',
    base_mpf_style='yahoo'
)

from io import BytesIO
from PIL import Image, ImageTk

def get_candle_img(
        df,
        limit=24, datetime_format='%H',
        xlabel='1 hour',
        xlocator=None,
        ylocator=None
):
    sub_df = df.iloc[-limit:].copy()
    sub_df['date'] = pd.to_datetime(sub_df['date'], utc=True).dt.tz_convert('Asia/Shanghai')
    sub_df = sub_df.set_index('date')
    fig, axes = mpf.plot(
        sub_df, datetime_format=datetime_format,
        style=my_style,
        figsize=(6.41, 5),
        type='candle', volume=False,
        returnfig=True, block=True
    )
    # 价格主图坐标轴通常是axes[0]
    ax = axes[0]

    ax.set_ylabel('')
    # 将y轴刻度移动到右侧
    # ax.yaxis.tick_right()
    # 设置主图 y 轴每 5 个单位一个刻度
    if ylocator:
        ax.yaxis.set_major_locator(ylocator)

    # 设置x轴标题
    ax.set_xlabel(xlabel)
    # 设置 x 轴刻度数量
    if xlocator:
        ax.xaxis.set_major_locator(xlocator)

    fig_buf = BytesIO()
    fig.savefig(fig_buf, format='png', bbox_inches='tight', pad_inches=0, dpi=125)
    fig_buf.seek(0)
    img = Image.open(fig_buf)

    tk_img = ImageTk.PhotoImage(img)

    matplotlib.pyplot.close()

    return tk_img

import tkinter as tk
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)

root = tk.Tk()
root.title('K线图预测')
root.attributes("-topmost", True)
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
root.geometry(f'{w}x{h}')

cv00 = tk.Label(root)
cv00.grid(row=0, column=0, padx=0, pady=0)
cv01 = tk.Label(root)
cv01.grid(row=1, column=0, padx=0, pady=0)
cv02 = tk.Label(root)
cv02.grid(row=0, column=1, padx=0, pady=0)

cv10 = tk.Label(root)
cv10.grid(row=0, column=2, padx=0, pady=0)
cv11 = tk.Label(root)
cv11.grid(row=1, column=1, padx=0, pady=0)
cv12 = tk.Label(root)
cv12.grid(row=1, column=2, padx=0, pady=0)

from datetime import datetime, timedelta
def seconds_to_next_15min():
    now = datetime.now()
    # 计算当前分钟属于哪个15分钟段
    next_minute = (now.minute // 15 + 1) * 15
    if next_minute == 60:
        # 跨小时
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)
    delta = next_time - now
    return int(delta.total_seconds()) + 10

df_15m = pd.read_parquet(fp_p('15m.pq'))
df_30m = pd.read_parquet(fp_p('30m.pq'))
df_1h = pd.read_parquet(fp_p('1h.pq'))
df_4h = pd.read_parquet(fp_p('4h.pq'))
df_1d = pd.read_parquet(fp_p('1d.pq'))
df_1w = pd.read_parquet(fp_p('1w.pq'))

def update_all_df():
    global df_15m, df_30m, df_1h, df_4h, df_1d, df_1w
    tmp = update_df(df_15m, client.KLINE_INTERVAL_15MINUTE, 15)
    if len(tmp):
        df_15m = pd.concat([df_15m, tmp], ignore_index=True)
    tmp = update_df(df_30m, client.KLINE_INTERVAL_30MINUTE, 30)
    if len(tmp):
        df_30m = pd.concat([df_30m, tmp], ignore_index=True)
    tmp = update_df(df_1h, client.KLINE_INTERVAL_1HOUR, 60)
    if len(tmp):
        df_1h = pd.concat([df_1h, tmp], ignore_index=True)
    tmp = update_df(df_4h, client.KLINE_INTERVAL_4HOUR, 60*4)
    if len(tmp):
        df_4h = pd.concat([df_4h, tmp], ignore_index=True)
    tmp = update_df(df_1d, client.KLINE_INTERVAL_1DAY, 60*24)
    if len(tmp):
        df_1d = pd.concat([df_1d, tmp], ignore_index=True)
    tmp = update_df(df_1w, client.KLINE_INTERVAL_1WEEK, 60*24*7)
    if len(tmp):
        df_1w = pd.concat([df_1w, tmp], ignore_index=True)

import requests
def flush():
    try:
        update_all_df()
        cv00.tk_img = get_candle_img(
            df_15m,
            limit=76, datetime_format='%H',
            xlabel='15MINUTE',
            ylocator=mticker.MultipleLocator(10)
        )
        cv00.config(image=cv00.tk_img)
        cv01.tk_img = get_candle_img(
            df_30m,
            limit=76, datetime_format='%H',
            xlabel='30MINUTE'
        )
        cv01.config(image=cv01.tk_img)
        cv02.tk_img = get_candle_img(
            df_1h,
            limit=76, datetime_format='%H',
            xlabel='1HOUR'
        )
        cv02.config(image=cv02.tk_img)
        cv10.tk_img = get_candle_img(
            df_4h,
            limit=76, datetime_format='%dd',
            xlabel='4HOUR'
        )
        cv10.config(image=cv10.tk_img)
        cv11.tk_img = get_candle_img(
            df_1d,
            limit=60, datetime_format='%dd',
            xlabel='1DAY'
        )
        cv11.config(image=cv11.tk_img)
        cv12.tk_img = get_candle_img(
            df_1w,
            limit=48, datetime_format='%mm',
            xlabel='1WEEK'
        )
        cv12.config(image=cv12.tk_img)
    except requests.exceptions.SSLError:
        root.after(3000, flush)
    finally:
        print(seconds_to_next_15min() / 60)
        root.after(seconds_to_next_15min() * 1000, flush)

root.after(50, flush)
root.mainloop()
