import pandas as pd
import torch
import os
try:
    fp_root = os.path.join(os.path.split(__file__)[0], 'data')
except NameError:
    fp_root = 'data'
def fp_p(*args):
    return os.path.join(fp_root, *args)

from models import MultiTimeScaleTransformer1Day, \
    MultiTimeScaleTransformer4Hour, MultiTimeScaleTransformer1Hour, MultiTimeScaleTransformer15MINUTE

df_15m = pd.read_parquet(fp_p('15m.pq'))
df_1h = pd.read_parquet(fp_p('1h.pq'))
df_4h = pd.read_parquet(fp_p('4h.pq'))
df_1d = pd.read_parquet(fp_p('1d.pq'))
df_1w = pd.read_parquet(fp_p('1w.pq'))

m_1d = MultiTimeScaleTransformer1Day()
m_4h = MultiTimeScaleTransformer4Hour()
m_1h = MultiTimeScaleTransformer1Hour()
m_15m = MultiTimeScaleTransformer15MINUTE()

nw = df_1w.iloc[7]['date'] + pd.Timedelta(minutes=1)

def apd_e(_c, _e):
    return torch.cat([_c, _e.unsqueeze(0)], dim=0)
def apd_t(_d, _t):
    return  pd.concat([_d, pd.Series(_t)], ignore_index=True)

def get_data_1d(now):
    # 8 周
    # 31 天

    df_before_now = df_1w[df_1w['date'] < now]
    result_1w = df_before_now.tail(8)

    df_before_now = df_1d[df_1d['date'] <= now]
    result_1d = df_before_now.tail(31)

    selected_columns = ['open', 'close', 'low', 'high', 'volume']
    # 只选这五列
    result_1w_a = result_1w[selected_columns].values  # shape: (seq_len, 5)
    result_1d_a = result_1d[selected_columns].values

    # 转为 torch 张量
    tensor_1w = torch.tensor(result_1w_a, dtype=torch.float32)
    tensor_1d = torch.tensor(result_1d_a, dtype=torch.float32)

    return tensor_1d, result_1d['date'], tensor_1w

# a,b,c = get_data_1d(nw)
# d, t = m_1d.forward(a, c, b)

def get_data_4h(now):
    # 30 天
    # 168 4h

    df_before_now = df_1d[df_1d['date'] < now]
    result_1w = df_before_now.tail(30)

    df_before_now = df_4h[df_4h['date'] <= now]
    result_1d = df_before_now.tail(168)

    selected_columns = ['open', 'close', 'low', 'high', 'volume']
    # 只选这五列
    result_1w_a = result_1w[selected_columns].values  # shape: (seq_len, 5)
    result_1d_a = result_1d[selected_columns].values

    # 转为 torch 张量
    tensor_1w = torch.tensor(result_1w_a, dtype=torch.float32)
    tensor_1d = torch.tensor(result_1d_a, dtype=torch.float32)

    return tensor_1d, result_1d['date'], tensor_1w, result_1w['date']

# a,b,c,d = get_data_4h(nw)
# e,t = m_4h.forward(a, c, b, d)

def get_data_1h(now):
    # 168 4h
    # 250 1h

    df_before_now = df_4h[df_4h['date'] < now]
    result_1w = df_before_now.tail(168)

    df_before_now = df_1h[df_1h['date'] <= now]
    result_1d = df_before_now.tail(250)

    selected_columns = ['open', 'close', 'low', 'high', 'volume']
    # 只选这五列
    result_1w_a = result_1w[selected_columns].values  # shape: (seq_len, 5)
    result_1d_a = result_1d[selected_columns].values

    # 转为 torch 张量
    tensor_1w = torch.tensor(result_1w_a, dtype=torch.float32)
    tensor_1d = torch.tensor(result_1d_a, dtype=torch.float32)

    return tensor_1d, result_1d['date'], tensor_1w, result_1w['date']

# a,b,c,d = get_data_1h(nw)
# e = m_1h.forward(a, c, b, d)

def get_data_15m(now):
    # 250 1h
    # 360 1h

    df_before_now = df_1h[df_1h['date'] < now]
    result_1w = df_before_now.tail(250)

    df_before_now = df_15m[df_15m['date'] <= now]
    result_1d = df_before_now.tail(360)

    selected_columns = ['open', 'close', 'low', 'high', 'volume']
    # 只选这五列
    result_1w_a = result_1w[selected_columns].values  # shape: (seq_len, 5)
    result_1d_a = result_1d[selected_columns].values

    # 转为 torch 张量
    tensor_1w = torch.tensor(result_1w_a, dtype=torch.float32)
    tensor_1d = torch.tensor(result_1d_a, dtype=torch.float32)

    return tensor_1d, result_1d['date'], tensor_1w, result_1w['date']

def get_tg(now, df):
    df_now = df[df['date'] > now]
    res = df_now.iloc[0][['open', 'close', 'low', 'high', 'volume']].values.astype(float)
    return torch.tensor(res, dtype=torch.float32)

# a,b,c,d = get_data_15m(nw)
# e = m_15m.forward(a, c, b, d)
import torch.nn as nn

stop = pd.Timestamp('2024-07-24 12:00:00')

lr_base = 0.05
op_1d = torch.optim.Adam(m_1d.parameters(), lr=lr_base*2)
op_4h = torch.optim.Adam(m_4h.parameters(), lr=lr_base)
op_1h = torch.optim.Adam(m_1h.parameters(), lr=lr_base)
op_15m = torch.optim.Adam(m_15m.parameters(), lr=lr_base)
criteria = nn.MSELoss()

m_1d.train()
m_4h.train()
m_1h.train()
m_15m.train()

nw_1d = nw
nw_4h = nw
nw_1h = nw

step = pd.Timedelta(minutes=15)
step_1d = pd.Timedelta(days=1)
step_4h = pd.Timedelta(hours=4)
step_1h = pd.Timedelta(hours=1)

while nw_1d <= stop:
    op_1d.zero_grad()
    a, b, c = get_data_1d(nw)
    p_1d, ts_1d = m_1d.forward(a/1000, c/1000, b)
    loss_1d = criteria(p_1d, get_tg(nw, df_1d) / 1000)
    print('1d', loss_1d.item())
    loss_1d.backward()
    op_1d.step()
    nw_1d += step_1d

nw_1d = nw

while nw_4h <= stop:
    op_4h.zero_grad()
    a, b, c, d = get_data_4h(nw)
    p_4h, ts_4h = m_4h.forward(a/1000, c/1000, b, d)
    loss_4h = criteria(p_4h, get_tg(nw, df_4h) / 1000)
    print('4h', loss_4h.item())
    loss_4h.backward()
    op_4h.step()
    nw_4h += step_4h

nw_4h = nw

while nw_1h <= stop:
    op_1h.zero_grad()
    a, b, c, d = get_data_1h(nw)
    p_1h, ts_1h = m_1h.forward(a/1000, c/1000, b, d)
    a_1h = get_tg(nw, df_1h) / 1000
    loss_1h = criteria(p_1h, a_1h)
    print('1h', loss_1h.item())
    loss_1h.backward()
    op_1h.step()
    nw_1h += step_1h


nw_1h = nw

torch.autograd.set_detect_anomaly(True)
while nw <= stop:
    # 清空所有优化器的梯度
    op_1d.zero_grad()
    op_4h.zero_grad()
    op_1h.zero_grad()
    op_15m.zero_grad()

    # 1天模型 - 始终运行，无论是否需要更新
    a, b, c = get_data_1d(nw)
    p_1d, ts_1d = m_1d.forward(a/1000, c/1000, b)

    # 4小时模型 - 始终运行
    a, b, c, d = get_data_4h(nw)
    c = apd_e(c, p_1d)
    d = apd_t(d, ts_1d)
    p_4h, ts_4h = m_4h.forward(a/1000, c/1000, b, d)

    # 1小时模型 - 始终运行
    a, b, c, d = get_data_1h(nw)
    c = apd_e(c, p_4h)
    d = apd_t(d, ts_4h)
    p_1h, ts_1h = m_1h.forward(a/1000, c/1000, b, d)

    # 15分钟模型 - 始终运行
    a, b, c, d = get_data_15m(nw)
    c = apd_e(c, p_1h)
    d = apd_t(d, ts_1h)
    p_15m, ts_15m = m_15m.forward(a/1000, c/1000, b, d)

    # 计算所有损失
    a_15m = get_tg(nw, df_15m)/1000
    print(p_15m*1000)
    print(a_15m*1000)
    total_loss = criteria(p_15m, a_15m)
    print('15m', total_loss.item())

    # 打印损失
    if nw_1d <= nw:
        loss_1d = criteria(p_1d, get_tg(nw, df_1d)/1000)
        print('1d', loss_1d.item())
        nw_1d += step_1d
    if nw_4h <= nw:
        loss_4h = criteria(p_4h, get_tg(nw, df_4h)/1000)
        print('4h', loss_4h.item())
        nw_4h += step_4h
    if nw_1h <= nw:
        a_1h = get_tg(nw, df_1h)/1000
        loss_1h = criteria(p_1h, a_1h)
        print('1h', loss_1h.item())
        nw_1h += step_1h

    # 只进行一次反向传播
    total_loss.backward()

    # 更新所有优化器
    op_1d.step()
    op_4h.step()
    op_1h.step()
    op_15m.step()

    nw += step
