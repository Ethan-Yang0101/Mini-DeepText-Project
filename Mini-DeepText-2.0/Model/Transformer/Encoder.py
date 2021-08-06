
import torch
import torch.nn as nn
from Model.Transformer.PositionalEncoding import PositionalEncoding
from Model.Transformer.MaskLayer import get_attn_pad_mask
from Model.Transformer.EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    '''整个Encoder模块'''

    def __init__(self, source_vocab_size, source_embed_dim, encoder_n_heads,
                 encoder_hid_dim, encoder_n_layers, encoder_max_seq_len):
        '''
        source_vocab_size: 需要翻译词典的单词数量
        source_embed_dim: 需要翻译词嵌入向量维度
        encoder_n_heads: 编码器注意力头数
        encoder_hid_dim: 编码器全链接残差块的隐藏层大小
        encoder_n_layers: 编码器模块的个数
        encoder_max_seq_len: 编码器最大句子长度
        '''
        super(Encoder, self).__init__()
        self.source_emb = nn.Embedding(source_vocab_size, source_embed_dim)
        self.pos_emb = PositionalEncoding(
            encoder_max_seq_len, source_embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(
            source_embed_dim, encoder_n_heads, encoder_hid_dim) for _ in range(encoder_n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: 编码器的句子索引输入 [batch_size, source_len]
        enc_outputs: 编码器的输出数据 [batch_size, source_len, source_emb_dim]
        enc_self_attns: 编码器的注意力矩阵 [encoder_n_layer, batch_size, encoder_n_heads, source_len, source_len]
        '''
        # word_emb: [batch_size, source_len, source_emb_dim]
        word_emb = self.source_emb(enc_inputs)
        # pos_emb: [batch_size, source_len, source_emb_dim]
        pos_emb = self.pos_emb(enc_inputs)
        # enc_outputs: [batch_size, source_len, source_emb_dim]
        enc_outputs = nn.Dropout(p=0.1)(word_emb + pos_emb)
        # enc_self_attn_mask: [batch_size, source_len, source_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, source_len, source_emb_dim]
            # enc_self_attn: [batch_size, encoder_n_heads, source_len, source_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
