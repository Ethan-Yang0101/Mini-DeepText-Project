
import torch
import numpy as np
import torch.nn as nn


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: 编码器和解码器都是编码器的输入数据 [batch_size, len_q]
    seq_k: 根据编码器和解码器使用不同输入数据 [batch_size, len_k]
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(dec_inputs):
    '''
    dec_inputs: 解码器的输入数据 [batch_size, target_len]
    '''
    attn_shape = [dec_inputs.size(0), dec_inputs.size(1), dec_inputs.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask
