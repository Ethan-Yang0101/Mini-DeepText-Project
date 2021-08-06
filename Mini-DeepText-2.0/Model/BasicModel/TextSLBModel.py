
import torch.nn.functional as F
import torch.nn as nn


class TextSLBModel(nn.Module):

    '''创建序列标注模型'''

    def __init__(self, num_embeddings, embedding_dim, rnn_hidden_size,
                 padding_idx=0, batch_first=True):
        '''
        Args:
            num_embeddings: 词嵌入矩阵的行数，等于词典单词的数量
            embedding_dim: 词嵌入矩阵的维度，人为规定大小
            rnn_hidden_size: RNN的隐藏层大小
            padding_idx: 将某个index作为padding对象
            batch_first: batch是否为数据集的第0维
        '''
        super(TextSLBModel, self).__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_dim,
                                padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=rnn_hidden_size,
                          batch_first=batch_first)
        self.fc = nn.Linear(in_features=rnn_hidden_size,
                            out_features=num_embeddings)

    def forward(self, input_batch, apply_softmax=False):
        '''输入数据批，返回每一个时间步长上的隐藏层数值'''
        # input_emb: [batch_size, seq_len, emb_dim]
        input_emb = self.emb(input_batch)
        # y_out: [batch_size, seq_len, hid_dim]
        y_out, _ = self.rnn(input_emb)
        batch_size, seq_size, feat_size = y_out.size()
        # y_out: [batch_size * seq_len, hid_dim]
        y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)
        # y_out: [batch_size * seq_len, num_emb]
        y_out = self.fc(F.dropout(y_out, 0.5))
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        new_feat_size = y_out.shape[-1]
        # y_out: [batch_size * seq_len, num_emb]
        y_out = y_out.view(batch_size, seq_size, new_feat_size)
        return y_out
