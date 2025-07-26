import pandas as pd
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        self.d_model = d_model

        # 基础位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('base_pe', pe)

    def forward(self, x):
        seq_len, _ = x.size()
        position_encodings = self.base_pe[:seq_len]
        return x + position_encodings

class TimeAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model_3, max_len=5000):
        super().__init__()
        self.d_model = d_model_3

        # 基础位置编码
        pe = torch.zeros(max_len, d_model_3)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_3, 2).float() * (-math.log(10000.0) / d_model_3))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('base_pe', pe)

        # 日周期编码 (24小时)
        day_pe = torch.zeros(24, d_model_3)
        hour_pos = torch.arange(0, 24, dtype=torch.float).unsqueeze(1)
        day_div_term = torch.exp(torch.arange(0, d_model_3 // 4, 2).float() * (-math.log(10000.0) / (d_model_3 // 4)))
        day_pe[:, 0::2] = torch.sin(hour_pos * day_div_term)
        day_pe[:, 1::2] = torch.cos(hour_pos * day_div_term)
        self.register_buffer('day_pe', day_pe)

        # 周周期编码 (7天)
        week_pe = torch.zeros(7, d_model_3)
        day_pos = torch.arange(0, 7, dtype=torch.float).unsqueeze(1)
        week_div_term = torch.exp(torch.arange(0, d_model_3 // 4, 2).float() * (-math.log(10000.0) / (d_model_3 // 4)))
        week_pe[:, 0::2] = torch.sin(day_pos * week_div_term)
        week_pe[:, 1::2] = torch.cos(day_pos * week_div_term)
        self.register_buffer('week_pe', week_pe)

    def forward(self, x, timestamps):
        """
        x: 输入张量 [seq_len, d_model]
        timestamps: 每个位置的时间戳 [seq_len]，可以是datetime对象或Unix时间戳
        """
        seq_len, _ = x.size()

        # 提取小时和星期几信息
        hours = torch.tensor([timestamp.hour for timestamp in timestamps])
        weekdays = torch.tensor([timestamp.weekday() for timestamp in timestamps])

        # 获取基础位置编码
        position_encodings = self.base_pe[:seq_len]

        # 获取时间周期编码
        day_encodings = torch.stack([self.day_pe[h] for h in hours])
        week_encodings = torch.stack([self.week_pe[w] for w in weekdays])

        # 组合所有编码
        combined_encoding = torch.cat([
            position_encodings,
            day_encodings,
            week_encodings
        ], dim=1)

        return x + combined_encoding

pos_encoder_1w = PositionalEncoding(24, max_len=8)

pos_encoder_1d = TimeAwarePositionalEncoding(8, 32)
class MultiTimeScaleTransformer1Day(nn.Module):
    def __init__(self, feature_dim=5, d_model=24, nhead=3, num_encoder_layers=2, num_decoder_layers=4,
                 dim_feedforward=48, dropout=0.1):
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model

        # 日线数据特征投影
        self.input_projection = nn.Linear(feature_dim, d_model)

        # 周期性K线数据特征投影
        self.weekly_input_projection = nn.Linear(feature_dim, d_model)

        # 日线数据Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 周期性K线数据Transformer编码器
        weekly_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.weekly_transformer_encoder = nn.TransformerEncoder(weekly_encoder_layer, num_encoder_layers)

        # Cross-Attention层 - 用于融合两种数据
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出预测头
        self.output_1day = nn.Linear(d_model, feature_dim)

    def forward(self, src, weekly_src, timestamps):
        """
        src: 日线输入序列 [seq_len, feature_dim]
        weekly_src: 周期性K线输入 [weekly_seq_len, feature_dim]
        timestamps: 日线序列的时间戳 [seq_len]
        tgt_timestamps: 目标预测的时间戳 [1].
        """

        # 特征投影
        src = self.input_projection(src)  # [seq_len, d_model]
        weekly_src = self.weekly_input_projection(weekly_src)  # [weekly_seq_len, d_model]

        # 添加位置编码
        src = pos_encoder_1d(src, timestamps)  # [seq_len, d_model]
        weekly_src = pos_encoder_1w(weekly_src)  # [weekly_seq_len, d_model]

        # Transformer编码
        memory = self.transformer_encoder(src.unsqueeze(1))  # [seq_len, 1, d_model]
        weekly_memory = self.weekly_transformer_encoder(weekly_src.unsqueeze(1))  # [weekly_seq_len, 1, d_model]

        # 应用Cross-Attention - 让日线数据关注周期性数据中的重要信息

        src_attended = memory  # [1, seq_len, d_model]
        weekly_attended = weekly_memory  # [1, weekly_seq_len, d_model]
        # print(src_attended.shape , weekly_attended.shape)

        enhanced_src, _ = self.cross_attention(
            query=src_attended,
            key=weekly_attended,
            value=weekly_attended
        )  # [1, seq_len, d_model]
        # print(enhanced_src.shape)

        enhanced_memory = enhanced_src  # [seq_len, 1, d_model]

        # 为解码器创建查询向量
        query = torch.zeros(1, 1, self.d_model)
        query_reshaped = query.squeeze(1)  # [1, d_model]
        tgt_timestamps = [timestamps.values[-1] + pd.Timedelta(days=1)]
        query_with_pos = pos_encoder_1d(query_reshaped, tgt_timestamps)
        query = query_with_pos.unsqueeze(1)  # [1, 1, d_model]
        # print(query.shape, enhanced_memory.shape)
        # Transformer解码 - 使用融合了周期数据的增强记忆
        output = self.transformer_decoder(query.transpose(0, 1) , enhanced_memory)  # [1, 1, d_model]
        output = output.squeeze(1)  # [1, d_model]

        # 生成预测
        pred_1day = self.output_1day(output[0])  # [feature_dim]

        return pred_1day, tgt_timestamps

pos_encoder_4h = TimeAwarePositionalEncoding(8, 192)
class MultiTimeScaleTransformer4Hour(nn.Module):
    def __init__(self, feature_dim=5, d_model=24, nhead=3, num_encoder_layers=2, num_decoder_layers=4,
                 dim_feedforward=48, dropout=0.1):
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model

        # 日线数据特征投影
        self.input_projection = nn.Linear(feature_dim, d_model)

        # 周期性K线数据特征投影
        self.weekly_input_projection = nn.Linear(feature_dim, d_model)

        # 日线数据Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 周期性K线数据Transformer编码器
        weekly_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.weekly_transformer_encoder = nn.TransformerEncoder(weekly_encoder_layer, num_encoder_layers)

        # Cross-Attention层 - 用于融合两种数据
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出预测头
        self.output_1day = nn.Linear(d_model, feature_dim)

    def forward(self, src, weekly_src, timestamps, weekly_timestamps):
        """
        src: 日线输入序列 [seq_len, feature_dim]
        weekly_src: 周期性K线输入 [weekly_seq_len, feature_dim]
        timestamps: 日线序列的时间戳 [seq_len]
        weekly_timestamps: 周期性K线的时间戳 [weekly_seq_len]
        """

        # 特征投影
        src = self.input_projection(src)  # [seq_len, d_model]
        weekly_src = self.weekly_input_projection(weekly_src)  # [weekly_seq_len, d_model]

        # 添加位置编码
        src = pos_encoder_4h(src, timestamps)  # [seq_len, d_model]
        weekly_src = pos_encoder_1d(weekly_src, weekly_timestamps)  # [weekly_seq_len, d_model]

        # Transformer编码
        memory = self.transformer_encoder(src.unsqueeze(1))  # [seq_len, 1, d_model]
        weekly_memory = self.weekly_transformer_encoder(weekly_src.unsqueeze(1))  # [weekly_seq_len, 1, d_model]

        # 应用Cross-Attention - 让日线数据关注周期性数据中的重要信息
        src_attended = memory  # [1, seq_len, d_model]
        weekly_attended = weekly_memory  # [1, weekly_seq_len, d_model]

        enhanced_src, _ = self.cross_attention(
            query=src_attended,
            key=weekly_attended,
            value=weekly_attended
        )  # [1, seq_len, d_model]

        enhanced_memory = enhanced_src  # [seq_len, 1, d_model]

        # 为解码器创建查询向量
        query = torch.zeros(1, 1, self.d_model)
        query_reshaped = query.squeeze(1)  # [1, d_model]
        tgt_timestamps = [timestamps.values[-1] + pd.Timedelta(hours=4)]
        query_with_pos = pos_encoder_4h(query_reshaped, tgt_timestamps)
        query = query_with_pos.unsqueeze(1)  # [1, 1, d_model]

        # Transformer解码 - 使用融合了周期数据的增强记忆
        output = self.transformer_decoder(query.transpose(0, 1) , enhanced_memory)  # [1, 1, d_model]
        output = output.squeeze(1)  # [1, d_model]

        # 生成预测
        pred_1day = self.output_1day(output[0])  # [feature_dim]

        return pred_1day, tgt_timestamps

pos_encoder_1h = TimeAwarePositionalEncoding(8, 256)
class MultiTimeScaleTransformer1Hour(nn.Module):
    def __init__(self, feature_dim=5, d_model=24, nhead=3, num_encoder_layers=2, num_decoder_layers=4,
                 dim_feedforward=48, dropout=0.1):
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model

        # 日线数据特征投影
        self.input_projection = nn.Linear(feature_dim, d_model)

        # 周期性K线数据特征投影
        self.weekly_input_projection = nn.Linear(feature_dim, d_model)

        # 日线数据Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 周期性K线数据Transformer编码器
        weekly_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.weekly_transformer_encoder = nn.TransformerEncoder(weekly_encoder_layer, num_encoder_layers)

        # Cross-Attention层 - 用于融合两种数据
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出预测头
        self.output_1day = nn.Linear(d_model, feature_dim)

    def forward(self, src, weekly_src, timestamps, weekly_timestamps):
        """
        src: 日线输入序列 [seq_len, feature_dim]
        weekly_src: 周期性K线输入 [weekly_seq_len, feature_dim]
        timestamps: 日线序列的时间戳 [seq_len]
        weekly_timestamps: 周期性K线的时间戳 [weekly_seq_len]
        """

        # 特征投影
        src = self.input_projection(src)  # [seq_len, d_model]
        weekly_src = self.weekly_input_projection(weekly_src)  # [weekly_seq_len, d_model]

        # 添加位置编码
        src = pos_encoder_1h(src, timestamps)  # [seq_len, d_model]
        weekly_src = pos_encoder_4h(weekly_src, weekly_timestamps)  # [weekly_seq_len, d_model]

        # Transformer编码
        memory = self.transformer_encoder(src.unsqueeze(1))  # [seq_len, 1, d_model]
        weekly_memory = self.weekly_transformer_encoder(weekly_src.unsqueeze(1))  # [weekly_seq_len, 1, d_model]

        # 应用Cross-Attention - 让日线数据关注周期性数据中的重要信息
        src_attended = memory  # [1, seq_len, d_model]
        weekly_attended = weekly_memory  # [1, weekly_seq_len, d_model]

        enhanced_src, _ = self.cross_attention(
            query=src_attended,
            key=weekly_attended,
            value=weekly_attended
        )  # [1, seq_len, d_model]

        enhanced_memory = enhanced_src  # [seq_len, 1, d_model]

        # 为解码器创建查询向量
        query = torch.zeros(1, 1, self.d_model)
        query_reshaped = query.squeeze(1)  # [1, d_model]
        tgt_timestamps = [timestamps.values[-1] + pd.Timedelta(hours=1)]
        query_with_pos = pos_encoder_1h(query_reshaped, tgt_timestamps)
        query = query_with_pos.unsqueeze(1)  # [1, 1, d_model]

        # Transformer解码 - 使用融合了周期数据的增强记忆
        output = self.transformer_decoder(query.transpose(0, 1) , enhanced_memory)  # [1, 1, d_model]
        output = output.squeeze(1)  # [1, d_model]

        # 生成预测
        pred_1day = self.output_1day(output[0])  # [feature_dim]

        return pred_1day, tgt_timestamps

pos_encoder_15m = TimeAwarePositionalEncoding(8, 384)
class MultiTimeScaleTransformer15MINUTE(nn.Module):
    def __init__(self, feature_dim=5, d_model=24, nhead=3, num_encoder_layers=2, num_decoder_layers=4,
                 dim_feedforward=48, dropout=0.1):
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model

        # 日线数据特征投影
        self.input_projection = nn.Linear(feature_dim, d_model)

        # 周期性K线数据特征投影
        self.weekly_input_projection = nn.Linear(feature_dim, d_model)

        # 日线数据Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 周期性K线数据Transformer编码器
        weekly_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.weekly_transformer_encoder = nn.TransformerEncoder(weekly_encoder_layer, num_encoder_layers)

        # Cross-Attention层 - 用于融合两种数据
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出预测头
        self.output_1day = nn.Linear(d_model, feature_dim)

    def forward(self, src, weekly_src, timestamps, weekly_timestamps):
        """
        src: 日线输入序列 [seq_len, feature_dim]
        weekly_src: 周期性K线输入 [weekly_seq_len, feature_dim]
        timestamps: 日线序列的时间戳 [seq_len]
        weekly_timestamps: 周期性K线的时间戳 [weekly_seq_len]
        """

        # 特征投影
        src = self.input_projection(src)  # [seq_len, d_model]
        weekly_src = self.weekly_input_projection(weekly_src)  # [weekly_seq_len, d_model]

        # 添加位置编码
        src = pos_encoder_15m(src, timestamps)  # [seq_len, d_model]
        weekly_src = pos_encoder_1h(weekly_src, weekly_timestamps)  # [weekly_seq_len, d_model]

        # Transformer编码
        memory = self.transformer_encoder(src.unsqueeze(1))  # [seq_len, 1, d_model]
        weekly_memory = self.weekly_transformer_encoder(weekly_src.unsqueeze(1))  # [weekly_seq_len, 1, d_model]

        # 应用Cross-Attention - 让日线数据关注周期性数据中的重要信息
        src_attended = memory  # [1, seq_len, d_model]
        weekly_attended = weekly_memory  # [1, weekly_seq_len, d_model]

        enhanced_src, _ = self.cross_attention(
            query=src_attended,
            key=weekly_attended,
            value=weekly_attended
        )  # [1, seq_len, d_model]

        enhanced_memory = enhanced_src  # [seq_len, 1, d_model]

        # 为解码器创建查询向量
        query = torch.zeros(1, 1, self.d_model)
        query_reshaped = query.squeeze(1)  # [1, d_model]
        tgt_timestamps = [timestamps.values[-1] + pd.Timedelta(minutes=15)]
        query_with_pos = pos_encoder_15m(query_reshaped, tgt_timestamps)
        query = query_with_pos.unsqueeze(1)  # [1, 1, d_model]

        # Transformer解码 - 使用融合了周期数据的增强记忆
        output = self.transformer_decoder(query.transpose(0, 1) , enhanced_memory)  # [1, 1, d_model]
        output = output.squeeze(1)  # [1, d_model]

        # 生成预测
        pred_1day = self.output_1day(output[0])  # [feature_dim]

        return pred_1day, tgt_timestamps