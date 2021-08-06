
import torch.nn as nn
import torch.nn.functional as F
import torch


class TextDSMModel(nn.Module):

    '''创建语义匹配模型'''

    def __init__(self, num_embeddings1, num_embeddings2, embedding_dim,
                 rnn_hidden_size, padding_idx=0, batch_first=True):
        '''
        Args:
            num_embeddings1: 词嵌入矩阵的行数，等于词典单词的数量
            num_embeddings2: 词嵌入矩阵的行数，等于词典单词的数量
            embedding_dim: 词嵌入矩阵的维度，人为规定大小
            rnn_hidden_size: RNN的隐藏层大小
            padding_idx: 将某个index作为padding对象
            batch_first: batch是否为数据集的第0维
        '''
        super(TextDSMModel, self).__init__()
        self.emb1 = nn.Embedding(num_embeddings=num_embeddings1,
                                 embedding_dim=embedding_dim,
                                 padding_idx=padding_idx)
        self.emb2 = nn.Embedding(num_embeddings=num_embeddings2,
                                 embedding_dim=embedding_dim,
                                 padding_idx=padding_idx)
        self.birnn1 = nn.GRU(embedding_dim, rnn_hidden_size,
                             bidirectional=True, batch_first=True)
        self.birnn2 = nn.GRU(embedding_dim, rnn_hidden_size,
                             bidirectional=True, batch_first=True)
        self.fc = nn.Linear(in_features=rnn_hidden_size * 4,
                            out_features=2)

    def forward(self, input_batch1, input_batch2, apply_softmax=False):
        '''输入数据批，返回匹配分数'''
        # input_emb12: [batch_size, seq_len, emb_dim]
        input_emb1 = self.emb1(input_batch1)
        input_emb2 = self.emb2(input_batch2)
        # hid_output12: [2 * 1, batch_size, hid_dim]
        _, hid_output1 = self.birnn1(input_emb1)
        _, hid_output2 = self.birnn2(input_emb2)
        # hid_output12: [batch_size, 2 * hid_dim]
        hid_output1 = hid_output1.permute(
            1, 0, 2).reshape(input_batch1.size(0), -1)
        hid_output2 = hid_output2.permute(
            1, 0, 2).reshape(input_batch2.size(0), -1)
        # cat_hid_output: [batch_size, 4 * hid_dim]
        cat_hid_output = torch.cat((hid_output1, hid_output2), dim=1)
        # output_scores: [batch_size, 2]
        output_scores = self.fc(cat_hid_output)
        if apply_softmax == True:
            output_scores = F.softmax(output_scores, dim=1)
        return output_scores
