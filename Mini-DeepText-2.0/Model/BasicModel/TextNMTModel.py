
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch


class NMTEncoder(nn.Module):

    '''创建一个编码器'''

    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        """
        Args:
            num_embeddings: 词嵌入矩阵的行数，等于词典单词的数量
            embedding_size: 词嵌入矩阵的维度，人为规定大小
            rnn_hidden_size: RNN的隐藏层大小
        """
        super(NMTEncoder, self).__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_size,
                                padding_idx=0)
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size,
                            bidirectional=True, batch_first=True)

    def forward(self, input_batch, input_lengths):
        '''
        input_batch: 编码器输入数据 [batch_size, seq_len]
        input_lengths: 编码器输入非零长度 [batch_size]
        input_unpacked: 编码器每个步长的输出 [batch_size, seq_len, 2 * hid_dim]
        input_birnn_h: 编码器最后隐藏层的输出 [batch_size, 2 * 1 * hid_dim]
        '''
        # input_emb: [batch_size, seq_len, emb_dim]
        input_emb = self.emb(input_batch)
        input_packed = pack_padded_sequence(input_emb, input_lengths.detach().cpu().numpy(),
                                            enforce_sorted=False, batch_first=True)
        # input_birnn_out: [batch_size, seq_len, 2 * hid_dim]
        # input_birnn_h: [2 * 1, batch_size, hid_dim]
        input_birnn_out, input_birnn_h = self.birnn(input_packed)
        # input_birnn_h: [batch_size, 2 * 1, hid_dim]
        input_birnn_h = input_birnn_h.permute(1, 0, 2)
        # input_birnn_h: [batch_size, 2 * 1 * hid_dim]
        input_birnn_h = input_birnn_h.contiguous().view(input_birnn_h.size(0), -1)
        # input_unpacked = [batch_size, seq_len, 2 * hid_dim]
        input_unpacked, _ = pad_packed_sequence(
            input_birnn_out, batch_first=True)
        return input_unpacked, input_birnn_h


class NMTDecoder(nn.Module):

    '''创建一个解码器'''

    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        """
        Args:
            num_embeddings: 词嵌入矩阵的行数，等于词典单词的数量
            embedding_size: 词嵌入矩阵的维度，人为规定大小
            rnn_hidden_size: RNN的隐藏层大小
        """
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_size,
                                padding_idx=0)
        self.gru_cell = nn.GRUCell(
            embedding_size+rnn_hidden_size, rnn_hidden_size)
        self.encoder_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings)

    def _init_context_vectors(self, batch_size):
        '''生成初始化的注意力机制的上下文向量'''
        return torch.zeros(batch_size, self._rnn_hidden_size)

    def terse_attention(self, encoder_state_vectors, query_vector):
        '''
        encoder_state_vectors: 编码器每个步长的输出 [batch_size, seq_len, rnn_hid_size]
        query_vector: 上一个步长的隐藏层输出 [batch_size, rnn_hid_size] 
        context_vectors: 加在本步长输入的注意力向量 [batch_size, rnn_hid_size]
        '''
        # encoder_state_vectors: [batch_size, seq_len, rnn_hid_size]
        # query_vector: [batch_size, rnn_hid_size]
        # vector_scores: [batch_size, seq_len]
        vector_scores = torch.matmul(
            encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
        vector_probabilities = F.softmax(vector_scores, dim=-1)
        # context_vector: [batch_size, rnn_hid_size]
        context_vectors = torch.matmul(encoder_state_vectors.transpose(-2, -1),
                                       vector_probabilities.unsqueeze(dim=2)).squeeze()
        return context_vectors

    def forward(self, encoder_states, encoder_output, target_sequence):
        '''
        encoder_states: 编码器每个步长的输出 [batch_size, seq_len, rnn_hid_size]
        encoder_output: 编码器最后一个隐藏层输出 [batch_size, rnn_hid_size]
        target_sequence: 解码器每个步长的输入 [batch_size, seq_len]
        output_vectors: 解码器每个步长的输出 [batch_size, seq_len, num_embeddings]
        '''
        # target_sequence: [seq_len, batch_size]
        target_sequence = target_sequence.permute(1, 0)
        seq_length = target_sequence.size(0)
        # h_t: [batch_size, rnn_hid_size]
        h_t = self.encoder_map(encoder_output)
        batch_size = encoder_states.size(0)
        # context_vectors: [batch_size, rnn_hid_size]
        context_vectors = self._init_context_vectors(batch_size)
        output_vectors = []
        for t in range(seq_length):
            # batch_indices: [batch_size]
            batch_indices = target_sequence[t]
            # batch_vectors: [batch_size, emb_dim]
            batch_vectors = self.emb(batch_indices)
            # rnn_output: [batch_size, emb_dim+rnn_hid_dim]
            rnn_input = torch.cat([batch_vectors, context_vectors], dim=1)
            # h_t: [batch_size, rnn_hid_size]
            h_t = self.gru_cell(rnn_input, h_t)
            # context_vectors: [batch_size, rnn_hid_size]
            context_vectors = self.terse_attention(
                encoder_state_vectors=encoder_states, query_vector=h_t)
            # prediction_vectors: [batch_size, rnn_hid_size * 2]
            prediction_vectors = torch.cat((context_vectors, h_t), dim=1)
            # prediction_scores: [batch_size, num_embeddings]
            prediction_scores = self.classifier(
                F.dropout(prediction_vectors, 0.3))
            output_vectors.append(prediction_scores)
        # output_vectors: [batch_size, seq_len, num_embeddings]
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        return output_vectors


class TextNMTModel(nn.Module):

    '''创建机器翻译模型'''

    def __init__(self, source_num_embeddings, source_embedding_size,
                 target_num_embeddings, target_embedding_size, encoding_size):
        """
        Args:
            source_num_embeddings: 词嵌入矩阵的行数，等于词典单词的数量
            source_embedding_size: 词嵌入矩阵的维度，人为规定大小
            target_num_embeddings: 词嵌入矩阵的行数，等于词典单词的数量
            target_embedding_size: 词嵌入矩阵的维度，人为规定大小
            encoding_size: 编码器的隐藏值向量大小
        """
        super(TextNMTModel, self).__init__()
        self.encoder = NMTEncoder(num_embeddings=source_num_embeddings,
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoder(num_embeddings=target_num_embeddings,
                                  embedding_size=target_embedding_size,
                                  rnn_hidden_size=decoding_size)

    def forward(self, source_sequence, source_lengths, target_sequence):
        '''
        source_sequence: 需要翻译数据输入 [batch_size, seq_len]
        source_lengths: 需要翻译数据非零长度 [batch_size]
        target_sequence: 目标翻译数据输入 [batch_size, seq_len]
        decoder_states: 解码器的输出 [batch_size, seq_len, num_embeddings]
        '''
        encoder_states, final_hidden_state = self.encoder(input_batch=source_sequence,
                                                          input_lengths=source_lengths)
        decoded_states = self.decoder(encoder_states=encoder_states,
                                      encoder_output=final_hidden_state,
                                      target_sequence=target_sequence)
        return decoded_states
