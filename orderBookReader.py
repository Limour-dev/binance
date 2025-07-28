import pickle
import os, time
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

class OB:
    def __init__(self, asks, bids, ts, price):
        self.asks = asks
        self.bids = bids
        self.ts = ts
        self.price = price

obs = []
for name in os.listdir(fp_p('ob'))[:-1]:
    with open(fp_p('ob', name), 'rb') as rf:
        while True:
            try:
                obs.append(pickle.load(rf))
            except EOFError:
                print(name, '读取完毕')
                break

# for obo in obs:
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(obo.ts // 1000)), obo.price)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

date_rng = np.zeros(len(obs))
price = np.zeros(len(obs))
for i,obo in enumerate(obs):
    date_rng[i] = obo.ts
    price[i] = obo.price
df = pd.DataFrame({'timestamp': date_rng, 'price': price})
df['date'] = pd.to_datetime(df['timestamp'], utc=True, unit='ms').dt.tz_convert('Asia/Shanghai')
df = df.drop('timestamp', axis=1)

df.set_index('date', inplace=True)

resampled = df['price'].resample('5min').agg([
    lambda x: np.percentile(x, 75) if len(x) > 20 else np.nan,
    lambda x: np.percentile(x, 25) if len(x) > 20 else np.nan
]).dropna(how='all')

resampled.columns = ['Q75', 'Q25']

plt.figure(figsize=(12, 6))
plt.plot(resampled.index, resampled['Q75'], label='Q75')
plt.plot(resampled.index, resampled['Q25'], label='Q25')
plt.fill_between(resampled.index, resampled['Q25'], resampled['Q75'], color='gray', alpha=0.2)
plt.xticks(rotation=45)
plt.gca().yaxis.tick_right()
plt.show(block=True)