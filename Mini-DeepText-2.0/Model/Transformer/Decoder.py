
import torch
import torch.nn as nn
from Model.Transformer.DecoderLayer import DecoderLayer
from Model.Transformer.PositionalEncoding import PositionalEncoding
from Model.Transformer.MaskLayer import get_attn_pad_mask
from Model.Transformer.MaskLayer import get_attn_subsequence_mask


class Decoder(nn.Module):
    '''整个Decoder模块'''

    def __init__(self, target_vocab_size, target_embed_dim, target_n_heads,
                 target_hid_dim, target_n_layers, decoder_max_seq_len):
        '''
        target_vocab_size: 目标翻译词典的单词数量
        target_embed_dim: 目标翻译词嵌入向量维度
        decoder_n_heads: 解码器注意力头数
        decoder_hid_dim: 解码器全链接残差块的隐藏层大小
        decoder_n_layers: 解码器模块的个数
        decoder_max_seq_len: 解码器最大句子长度
        '''
        super(Decoder, self).__init__()
        self.target_emb = nn.Embedding(target_vocab_size, target_embed_dim)
        self.pos_emb = PositionalEncoding(
            decoder_max_seq_len, target_embed_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(target_embed_dim, target_n_heads, target_hid_dim) for _ in range(target_n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: 解码器的句子索引输入 [batch_size, target_len]
        enc_inputs: 编码器的句子索引输入 [batch_size, source_len]
        enc_outputs: 编码器的输出 [batch_size, source_len, source_emb_dim]
        dec_outputs: 解码器的输出 [batch_size, target_len, target_emb_dim]
        dec_self_attns: 解码器下注意力矩阵 [decoder_n_layers, batch_size, decoder_n_heads, target_len, target_len]
        dec_enc_attns: 解码器上注意力矩阵 [decoder_n_layers, batch_size, decoder_n_heads, target_len, source_len]
        '''
        # word_emb: [batch_size, target_len, target_emb_dim]
        word_emb = self.target_emb(dec_inputs)
        # pos_emb: [batch_size, target_len, target_emb_dim]
        pos_emb = self.pos_emb(dec_inputs)
        # dec_outputs: [batch_size, target_len, target_emb_dim]
        dec_outputs = nn.Dropout(p=0.1)(word_emb + pos_emb)
        # dec_self_attn_pad_mask: [batch_size, target_len, target_len]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # dec_self_attn_subsequence_mask: [batch_size, target_len, target_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        # dec_self_attn_mask: [batch_size, target_len, target_len]
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)
        # dec_enc_attn_mask: [batch_size, target_len, source_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, target_len, target_emb_dim]
            # dec_self_attn: [batch_size, decoder_n_heads, target_len, target_len]
            # dec_enc_attn: [batch_size, decoder_n_heads, target_len, source_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
