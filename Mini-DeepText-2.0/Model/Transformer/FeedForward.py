
import torch
import torch.nn as nn


class PoswiseFeedForwardNet(nn.Module):
    '''两个全链接组成的残差模块'''

    def __init__(self, d_model, hid_dim):
        '''
        d_model: 词嵌入向量维度
        hid_dim: 隐藏层的维度
        '''
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hid_dim, d_model, bias=False)
        )
        self.d_model = d_model
        self.hid_dim = hid_dim
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        '''
        inputs: 多头注意力机制的输出 [batch_size, len_q, d_model]
        output: 残差块的输出 [batch_size, len_q, d_model]
        '''
        residual = inputs
        output = self.fc(inputs) + residual
        output = self.layernorm(self.dropout(output))
        return output
