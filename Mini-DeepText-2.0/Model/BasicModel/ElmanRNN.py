
import torch
import torch.nn as nn


class ElmanRNN(nn.Module):

    '''使用RNNCell创建一个RNN层'''

    def __init__(self, input_size, hidden_size, batch_first=True):
        '''
        Args:
            input_size: RNN输入数据的维度
            hidden_size: RNN的隐藏层的大小
            batch_first: batch是否为数据集的第0维
        '''
        super(ElmanRNN, self).__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.batch_first = batch_first
        self.hidden_size = hidden_size

    def _initial_hidden(self, batch_size):
        '''初始化隐藏层数值'''
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, input_batch, initial_hidden=None):
        '''输入数据批，返回每一个时间步长上的隐藏层数值'''
        if self.batch_first:
            batch_size, seq_size, emb_size = input_batch.size()
            input_batch = input_batch.permute(1, 0, 2)
        else:
            seq_size, batch_size, emb_size = input_batch.size()
        hiddens = []
        if initial_hidden is None:
            initial_hidden = self._initial_hidden(batch_size)
        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn_cell(input_batch[t], hidden_t)
            hiddens.append(hidden_t)
        hiddens = torch.stack(hiddens)
        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)
        return hiddens
