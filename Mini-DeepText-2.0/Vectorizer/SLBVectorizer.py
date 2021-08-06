
from Vocabulary.SequenceVocabulary import SequenceVocabulary
from Vocabulary.Vocabulary import Vocabulary
import numpy as np


class SLBVectorizer(object):

    '''创建一个矢量化器类将文本句子转换为句子索引矢量'''

    def __init__(self, source_vocab, target_vocab):
        '''
        Args:
            source_vocab: 包含数据集中所有文本的词典
            target_vocab: 包含数据集中所有标注文本的词典
        '''
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def _vectorize(self, indices, vector_length, mask_index):
        '''基本的句子矢量化过程'''
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index
        return vector

    def _get_source_indices(self, sentence):
        '''从文本中获取句子索引'''
        begin_index = [self.source_vocab.begin_index]
        indices = [self.source_vocab.lookup_token(token) for token in sentence]
        return begin_index + indices

    def _get_target_indices(self, sentence):
        '''从文本中获取句子索引'''
        end_index = [self.target_vocab.end_index]
        indices = [self.target_vocab.lookup_token(token) for token in sentence]
        return indices + end_index

    def vectorize(self, row_dict, max_seq_length):
        '''矢量化文本句子，将句子中的每个单词用索引表示，生成句子索引矢量'''
        source_vector_length = max_seq_length + 1
        target_vector_length = max_seq_length + 1
        source_indices = self._get_source_indices(row_dict['source'])
        source_vector = self._vectorize(indices=source_indices,
                                        vector_length=source_vector_length,
                                        mask_index=self.source_vocab.mask_index)
        target_indices = self._get_target_indices(row_dict['target'])
        target_vector = self._vectorize(indices=target_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)
        return {'source_vector': source_vector,
                'target_vector': target_vector}

    @classmethod
    def from_freq_dict(cls, source_freq_dict, target_freq_dict, cutoff):
        '''通过词频集创建一个矢量化器'''
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()
        for token in list(source_freq_dict.keys()):
            if source_freq_dict[token] >= cutoff:
                source_vocab.add_token(token)
        for topic in list(target_freq_dict.keys()):
            if target_freq_dict[token] >= cutoff:
                target_vocab.add_token(token)
        return cls(source_vocab, target_vocab)

    def to_serializable(self):
        '''生成序列化信息，方便使用JSON保存初始化信息'''
        return {'source_vocab': self.source_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}

    @classmethod
    def from_serializable(cls, contents):
        '''通过使用contents(序列化后的初始化信息)重建实例'''
        source_vocab = SequenceVocabulary.from_serializable(
            contents['source_vocab'])
        target_vocab = SequenceVocabulary.from_serializable(
            contents['target_vocab'])
        return cls(source_vocab, target_vocab)
