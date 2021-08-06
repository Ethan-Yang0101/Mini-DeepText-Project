
from Model.Transformer.Encoder import Encoder
from Model.Transformer.Decoder import Decoder
import torch.nn as nn


class Transformer(nn.Module):
    '''变压器神经网络'''

    def __init__(self, source_vocab_size, target_vocab_size, source_embed_dim,
                 target_embed_dim, encoder_n_heads, decoder_n_heads,
                 encoder_hid_dim, decoder_hid_dim, encoder_n_layers, decoder_n_layers,
                 encoder_max_seq_len, decoder_max_seq_len):
        '''
        source_vocab_size: 需要翻译词典的单词数量
        target_vocab_size: 目标翻译词典的单词数量
        source_embed_dim: 需要翻译词嵌入向量维度
        target_embed_dim: 目标翻译词嵌入向量维度
        encoder_n_heads: 编码器注意力头数
        decoder_n_heads: 解码器注意力头数
        encoder_hid_dim: 编码器全链接残差块的隐藏层大小
        decoder_hid_dim: 解码器全链接残差块的隐藏层大小
        encoder_n_layers: 编码器模块的个数
        decoder_n_layers: 解码器模块的个数
        encoder_max_seq_len: 编码器最大句子长度
        decoder_max_seq_len: 解码器最大句子长度
        '''
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            source_vocab_size, source_embed_dim, encoder_n_heads, encoder_hid_dim,
            encoder_n_layers, encoder_max_seq_len+2)
        self.decoder = Decoder(
            target_vocab_size, target_embed_dim, decoder_n_heads, decoder_hid_dim,
            decoder_n_layers, decoder_max_seq_len+1)
        self.projection = nn.Linear(
            target_embed_dim, target_vocab_size, bias=False)

    def forward(self, source_sequence, source_lengths, target_sequence):
        '''
        source_sequence: 编码器的句子索引输入 [batch_size, source_len]
        source_lengths: 统一API而保留的接口，暂时不使用
        target_sequence: 解码器的句子索引输入 [batch_size, target_len]
        outputs: 解码器的输出 [batch_size, target_len, target_vocab_size]
        enc_mul_attns: 编码器注意力矩阵 [encoder_n_layer, batch_size, encoder_n_heads, source_len, source_len]
        dec_mul_attns: 解码器下注意力矩阵 [decoder_n_layer, batch_size, decoder_n_heads, target_len, target_len]
        dec_enc_attns: 解码器上注意力矩阵 [decoder_n_layer, batch_size, decoder_n_heads, target_len, source_len]
        '''
        # enc_outputs: [batch_size, source_len, source_emb_dim]
        # enc_self_attns: [encoder_n_layer, batch_size, encoder_n_heads, source_len, source_len]
        enc_outputs, enc_self_attns = self.encoder(source_sequence)
        # dec_outputs: [batch_size, target_len, target_emb_dim]
        # dec_self_attns: [decoder_n_layer, batch_size, decoder_n_heads, target_len, target_len]
        # dec_enc_attns: [decoder_n_layer, batch_size, decoder_n_heads, target_len, source_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            target_sequence, source_sequence, enc_outputs)
        # output: [batch_size, target_len, target_vocab_size]
        outputs = self.projection(dec_outputs)
        return outputs
