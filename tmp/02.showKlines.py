import pandas as pd
import os, sys
fp_root = os.path.join(os.path.split(__file__)[0], 'data')
def fp_p(*args):
    return os.path.join(fp_root, *args)

df = pd.read_parquet(fp_p('1h.pq'))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

cols = ['open','close','low','high','volume']
for col in cols:
    df[col] = df[col].astype(float)

fig, axes = mpf.plot(
    df.set_index('date'), datetime_format='%H',
    style=my_style,
    figsize=(6.41, 5.15),
    type='candle', volume=False,
    returnfig=True, block=True
)
# 价格主图坐标轴通常是axes[0]
ax = axes[0]

ax.set_ylabel('')
# 将y轴刻度移动到右侧
# ax.yaxis.tick_right()
# 设置主图 y 轴每 5 个单位一个刻度
ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

ax.set_xlabel('15 minutes')
# 设置 x 轴刻度数量
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,24]))

is_PyCharm = '_pydev_imps' in sys.modules
# print('绘图结束', is_PyCharm)

from io import BytesIO
from PIL import Image, ImageTk
fig_buf = BytesIO()
fig.savefig(fig_buf, format='png', bbox_inches='tight', pad_inches=0, dpi=125)
fig_buf.seek(0)
img = Image.open(fig_buf)

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
cv01.grid(row=0, column=1, padx=0, pady=0)
cv02 = tk.Label(root)
cv02.grid(row=0, column=2, padx=0, pady=0)

cv10 = tk.Label(root)
cv10.grid(row=1, column=0, padx=0, pady=0)
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
    return int(delta.total_seconds()) + 1

def flush():
    try:
        tk_img = ImageTk.PhotoImage(img)
        cv00.config(image=tk_img)
        cv01.config(image=tk_img)
        cv02.config(image=tk_img)
        cv10.config(image=tk_img)
        cv11.config(image=tk_img)
        cv12.config(image=tk_img)
        cv12.img = tk_img
    finally:
        print(seconds_to_next_15min() / 60)
        root.after(seconds_to_next_15min() * 1000, flush)

root.after(50, flush)
root.mainloop()
