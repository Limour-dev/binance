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

for obo in obs:
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(obo.ts // 1000)), obo.price)