
import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    '''多头中的自注意力机制模块'''

    def __init__(self, dk):
        '''
        dk: 多头注意力向量的维度
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.dk = dk

    def forward(self, Q, K, V, attn_mask):
        '''
        QKV: 查询矩阵，键矩阵，值矩阵 [batch_size, n_heads, len_q, dqkv]
        attn_mask: 掩盖pad在softmax中计算的层 [batch_size, n_heads, len_q, len_k]
        context: 注意力机制生成的向量 [batch_size, n_heads, len_q, dv]
        attn: 注意力机制的权重矩阵 [batch_size, n_heads, len_q, len_k]
        '''
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk)
        scores.masked_fill_(attn_mask, -1e9)
        # attn: [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        # context: [batch_size, n_heads, len_q, dv]
        context = torch.matmul(attn, V)
        return context, attn
