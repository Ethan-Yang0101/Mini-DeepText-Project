
from Model.BasicModel.ElmanRNN import ElmanRNN
import torch.nn.functional as F
import torch.nn as nn


class TextCLRModel(nn.Module):

    '''创建文本分类模型'''

    def __init__(self, num_embeddings, embedding_dim, rnn_hidden_size,
                 num_classes, padding_idx=0, batch_first=True):
        '''
        Args:
            num_embeddings: 词嵌入矩阵的行数，等于词典单词的数量
            embedding_dim: 词嵌入矩阵的维度，人为规定大小
            rnn_hidden_size: RNN的隐藏层大小
            num_classes: 输出层的大小
            padding_idx: 将某个index作为padding对象
            batch_first: batch是否为数据集的第0维
        '''
        super(TextCLRModel, self).__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_dim,
                                padding_idx=padding_idx)
        self.rnn = ElmanRNN(input_size=embedding_dim,
                            hidden_size=rnn_hidden_size,
                            batch_first=batch_first)
        self.fc1 = nn.Linear(in_features=rnn_hidden_size,
                             out_features=rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size,
                             out_features=num_classes)

    def forward(self, input_batch, apply_softmax=False):
        '''输入数据批，返回批的最后一个时间步长上的隐藏层数值'''
        # input_emb: [batch_size, seq_len, emb_dim]
        input_emb = self.emb(input_batch)
        # rnn_out: [batch_size, seq_len, hid_dim]
        rnn_out = self.rnn(input_emb)
        # rnn_out: [batch_size, hid_dim]
        rnn_out = rnn_out[:, -1, :]
        # output_batch: [batch_size, hid_dim]
        output_batch = F.relu(self.fc1(F.dropout(rnn_out, 0.5)))
        # output_batch: [batch_size, num_classes]
        output_batch = self.fc2(F.dropout(output_batch, 0.5))
        if apply_softmax:
            output_batch = F.softmax(output_batch, dim=1)
        return output_batch
