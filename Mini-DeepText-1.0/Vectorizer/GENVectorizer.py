
from Vocabulary.SequenceVocabulary import SequenceVocabulary
from Vocabulary.Vocabulary import Vocabulary
import numpy as np

class GENVectorizer(object):

    '''创建一个矢量化器类将文本句子转换为句子索引矢量'''

    def __init__(self, source_vocab, target_vocab):
        '''
        Args:
            source_vocab: 包含数据集中所有文本的词典
            target_vocab: 包含数据集中所有标签的词典
        '''
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    # 矢量化文本句子，将句子中的每个词用索引表示，生成用于训练和预测的句子索引矢量
    def vectorize(self, row_dict, max_seq_length):
        max_vector_length = max_seq_length + 1
        begin_index = [self.source_vocab.begin_index]
        end_index = [self.source_vocab.end_index]
        indices = [self.source_vocab.lookup_token(token) for token in row_dict['source']]
        source_vector = np.zeros(max_vector_length, dtype=np.int64)
        source_indices = begin_index + indices
        source_vector[:len(source_indices)] = source_indices
        source_vector[len(source_indices):] = self.source_vocab.mask_index
        target_vector = np.zeros(max_vector_length, dtype=np.int64)
        target_indices = indices + end_index
        target_vector[:len(target_indices)] = target_indices
        target_vector[len(target_indices):] = self.source_vocab.mask_index
        return source_vector, target_vector

    # 通过词频集创建一个矢量化器
    @classmethod
    def from_freq_dict(cls, source_freq_dict, target_freq_dict, cutoff):
        source_vocab = SequenceVocabulary()
        target_vocab = Vocabulary()
        for token in list(source_freq_dict.keys()):
            if source_freq_dict[token] >= cutoff:
                source_vocab.add_token(token)
        for topic in list(target_freq_dict.keys()):
            target_vocab.add_token(topic)
        return cls(source_vocab, target_vocab)

    # 生成序列化信息，方便使用JSON保存初始化信息
    def to_serializable(self):
        return {'source_vocab': self.source_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}

    # 通过使用contents(序列化后的初始化信息)重建实例
    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents['source_vocab'])
        target_vocab = Vocabulary.from_serializable(contents['target_vocab'])
        return cls(source_vocab, target_vocab)
