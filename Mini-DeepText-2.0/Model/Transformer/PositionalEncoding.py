
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    '''位置编码层模块'''

    def __init__(self, max_seq_len, d_model):
        '''
        max_seq_len: 最大句子长度
        d_model: 词嵌入向量维度
        '''
        super(PositionalEncoding, self).__init__()
        pos_encoding = np.array([[pos / (np.power(10000, 2.0 * (i // 2) / d_model))
                                  for i in range(d_model)] for pos in range(max_seq_len+1)])
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        pos_encoding = torch.from_numpy(pos_encoding).float()
        self.positional_encoding = nn.Embedding(max_seq_len+1, d_model)
        self.positional_encoding.weight = nn.Parameter(
            pos_encoding, requires_grad=False)

    def forward(self, inputs):
        '''
        inputs: 编码器或解码器的输入 [batch_size, seq_len]
        outputs: 编码器或解码器的输出 [batch_size, seq_len, emb_dim]
        '''
        batch_size, seq_len = inputs.size()
        mask = inputs.data.numpy() == 0
        absolute_encode = np.array([[i for i in range(1, seq_len+1)]
                                    for _ in range(batch_size)])
        absolute_encode[mask] = 0
        absolute_encode = torch.from_numpy(absolute_encode).long()
        outputs = self.positional_encoding(absolute_encode)
        return outputs
