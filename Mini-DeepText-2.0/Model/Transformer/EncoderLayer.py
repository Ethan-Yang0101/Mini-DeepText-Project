
import torch
import torch.nn as nn
from Model.Transformer.FeedForward import PoswiseFeedForwardNet
from Model.Transformer.MultiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    '''单个Encoder模块'''

    def __init__(self, source_embed_dim, encoder_n_heads, encoder_hid_dim):
        '''
        source_embed_dim: 词嵌入向量的维度
        encoder_n_heads: 自注意力机制模块的个数
        encoder_hid_dim: 残差模块的隐藏层大小
        '''
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            source_embed_dim, encoder_n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(source_embed_dim, encoder_hid_dim)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: 词嵌入+位置编码后的数据表示 [batch_size, source_len, source_embed_dim]
        enc_self_attn_mask: 编码器自注意力机制的掩码层 [batch_size, source_len, source_len]
        enc_outputs: 编码器的输出 [batch_size, source_len, source_embed_dim]
        attn: 注意力机制矩阵 [batch_size, encoder_n_heads, source_len, source_len]
        '''
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
