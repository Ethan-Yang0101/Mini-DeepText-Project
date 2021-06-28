
import torch.nn.functional as F
import torch.nn as nn

class TextGenModel(nn.Module):

    '''创建文本生成模型'''

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
        super(TextGenModel, self).__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_dim,
                                padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=rnn_hidden_size,
                          batch_first=batch_first)
        self.fc = nn.Linear(in_features=rnn_hidden_size,
                            out_features=num_embeddings)

    # 输入数据批，返回每一个时间步长上的隐藏层数值
    def forward(self, input_batch, apply_softmax=False):
        input_emb = self.emb(input_batch)
        y_out, _ = self.rnn(input_emb)
        batch_size, seq_size, feat_size = y_out.size()
        y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)
        y_out = self.fc(F.dropout(y_out, 0.5))
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        new_feat_size = y_out.shape[-1]
        y_out = y_out.view(batch_size, seq_size, new_feat_size)
        return y_out
