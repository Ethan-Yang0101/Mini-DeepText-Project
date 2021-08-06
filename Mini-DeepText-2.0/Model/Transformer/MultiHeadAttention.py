
import torch
import torch.nn as nn
from Model.Transformer.SelfAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    '''多头注意力机制模块'''

    def __init__(self, d_model, n_heads):
        '''
        d_model: 词嵌入向量的维度
        n_heads: 自注意力机制模块的个数
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dq = d_model // n_heads
        self.dk = d_model // n_heads
        self.dv = d_model // n_heads
        self.W_Q = nn.Linear(d_model, n_heads * self.dq, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * self.dk, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * self.dv, bias=False)
        self.fc = nn.Linear(n_heads * self.dv, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_QKV: 词嵌入+位置编码后的数据表示 [batch_size, len_q, d_model]
        attn_mask: 掩盖pad在softmax中计算的层 [batch_size, len_q, len_k]
        output: 多头注意力机制生成的向量 [batch_size, len_q, d_model]
        attn: 注意力机制的权重矩阵 [batch_size, n_heads, len_q, len_k]
        '''
        # input_Q作为残差块的输入，编码解码都一样
        residual = input_Q
        # 准备变换数据的维度
        batch_size = input_Q.size(0)
        len_q, len_k, len_v = input_Q.size(1), input_K.size(1), input_V.size(1)
        # Q: [batch_size, n_heads, len_q, dq]
        Q = self.W_Q(input_Q).view(batch_size, len_q,
                                   self.n_heads, self.dq).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, dk]
        K = self.W_K(input_K).view(batch_size, len_k,
                                   self.n_heads, self.dk).transpose(1, 2)
        # V: [batch_size, n_heads, len_v, dv]
        V = self.W_V(input_V).view(batch_size, len_v,
                                   self.n_heads, self.dv).transpose(1, 2)
        # attn_mask: [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [batch_size, n_heads, len_q, dv], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dk)(Q, K, V, attn_mask)
        # context: [batch_size, len_q, n_heads * dv]
        context = context.transpose(1, 2).reshape(
            batch_size, len_q, self.n_heads * self.dq)
        # output: [batch_size, len_q, d_model]
        output = self.fc(context) + residual
        output = self.layernorm(self.dropout(output))
        return output, attn
