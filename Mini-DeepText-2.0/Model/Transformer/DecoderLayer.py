
import torch
import torch.nn as nn
from Model.Transformer.FeedForward import PoswiseFeedForwardNet
from Model.Transformer.MultiHeadAttention import MultiHeadAttention


class DecoderLayer(nn.Module):
    '''单个Decoder模块'''

    def __init__(self, target_embed_dim, decoder_n_heads, decoder_hid_dim):
        '''
        target_embed_dim: 词嵌入向量的维度
        decoder_n_heads: 自注意力机制模块的个数
        decoder_hid_dim: 残差模块的隐藏层大小
        '''
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            target_embed_dim, decoder_n_heads)
        self.dec_enc_attn = MultiHeadAttention(
            target_embed_dim, decoder_n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(target_embed_dim, decoder_hid_dim)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: 解码器的输入数据 [batch_size, target_len, target_embed_dim]
        enc_outputs: 编码器的输出数据 [batch_size, source_len, source_embed_dim]
        dec_self_attn_mask: 解码器下面注意力掩码层 [batch_size, target_len, target_len]
        dec_enc_attn_mask: 解码器上面注意力掩码层 [batch_size, target_len, source_len]
        dec_self_attn: 解码器下面注意力矩阵 [batch_size, decoder_n_heads, target_len, target_len]
        dec_enc_attn: 解码器上面注意力矩阵 [batch_size, decoder_n_heads, target_len, source_len]
        dec_outputs: 解码器的输出数据 [batch_size, target_len, target_embed_dim]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
