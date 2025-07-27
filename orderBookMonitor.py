from binance.client import Client, BinanceAPIException, BinanceRequestException
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

proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}
_client = Client(requests_params={'proxies': proxies})
def client(max_retries=3, delay=2):
    global _client
    for attempt in range(1, max_retries + 1):
        try:
            _client.futures_ping()
            return _client
        except (BinanceAPIException, BinanceRequestException, Exception) as e:
            _client = Client(requests_params={'proxies': proxies})
            print(f"第 {attempt} 次尝试失败: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise BinanceRequestException("已达最大重试次数，Ping 失败")
symbol = 'ETHUSDC'

def all2float(_l):
    return [(float(x),float(y)) for x,y in _l]

def merge_to_integers(sequence, price):
    """
    将形如 [(value, quantity), ...] 的序列合并成整数值序列

    参数:
    sequence -- 原始序列，每个元素为 (value, quantity) 的元组

    返回:
    合并后的序列，每个元素为 (integer_value, merged_quantity) 的元组，
    其中第一个整数值不小于原序列中的第一个值
    """
    reverse = (sequence[0][0] < price)
    # 按整数范围分组
    integer_groups = {}

    for value, quantity in sequence:
        # 确定该值所属的整数范围
        lower_int = int(value)
        upper_int = lower_int + 1

        # 计算该值在两个整数之间的位置比例
        lower_weight = upper_int - value
        upper_weight = value - lower_int

        # 将数量按比例分配到相邻的两个整数
        if lower_int in integer_groups:
            integer_groups[lower_int] += quantity * lower_weight
        else:
            integer_groups[lower_int] = quantity * lower_weight

        if upper_weight > 0:  # 如果value不是整数
            if upper_int in integer_groups:
                integer_groups[upper_int] += quantity * upper_weight
            else:
                integer_groups[upper_int] = quantity * upper_weight
    result = [[key, integer_groups[key]] for key in sorted(integer_groups.keys(), reverse=reverse)]
    if reverse:
        if result[0][0] <= price:
            return result
    else:
        if result[0][0] >= price:
            return result
    result[1][1] += (result[0][0] * result[0][1]) / result[1][0]
    return result[1:]

class OB:
    def __init__(self, asks, bids, ts, price):
        self.asks = asks
        self.bids = bids
        self.ts = ts
        self.price = price

def get_order_book():
    ob = client().futures_order_book(symbol=symbol, limit=1000)
    asks = all2float(ob['asks'])
    bids = all2float(ob['bids'])
    ts = ob['T']
    price = (asks[0][0]*asks[0][1] + bids[0][0]*bids[0][1]) / (asks[0][1] + bids[0][1])
    asks = merge_to_integers(asks, price)
    bids = merge_to_integers(bids, price)
    return OB(asks, bids, ts, price)


with open(fp_p('ob',str(int(time.time()))), 'wb') as wf:
    while True:
        obo = get_order_book()
        pickle.dump(obo, wf)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(obo.ts // 1000)), obo.price)
        time.sleep(10)
