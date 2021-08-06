
from Vocabulary.SequenceVocabulary import SequenceVocabulary
from Vocabulary.Vocabulary import Vocabulary
import numpy as np


class CLRVectorizer(object):

    '''创建一个矢量化器类将文本句子转换为句子索引矢量'''

    def __init__(self, source_vocab, label_vocab):
        '''
        Args:
            source_vocab: 包含数据集中所有文本的词典
            label_vocab: 包含数据集中所有标签的词典
        '''
        self.source_vocab = source_vocab
        self.label_vocab = label_vocab

    def _vectorize(self, indices, vector_length, mask_index):
        '''基本的句子矢量化过程'''
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index
        return vector

    def _get_indices(self, sentence):
        '''从文本中获取句子索引'''
        begin_index = [self.source_vocab.begin_index]
        end_index = [self.source_vocab.end_index]
        indices = [self.source_vocab.lookup_token(token) for token in sentence]
        input_indices = begin_index + indices + end_index
        return input_indices

    def vectorize(self, row_dict, max_seq_length):
        '''矢量化文本句子，将句子中的每个单词用索引表示，生成句子索引矢量'''
        max_vector_length = max_seq_length + 2
        input_indices = self._get_indices(row_dict['source'])
        source_vector = self._vectorize(indices=input_indices,
                                        vector_length=max_vector_length,
                                        mask_index=self.source_vocab.mask_index)
        label_index = self.label_vocab.lookup_token(row_dict['label'][0])
        return {'source_vector': source_vector,
                'label_index': label_index}

    @classmethod
    def from_freq_dict(cls, source_freq_dict, label_freq_dict, cutoff):
        '''通过词频集创建一个矢量化器'''
        source_vocab = SequenceVocabulary()
        label_vocab = Vocabulary()
        for token in list(source_freq_dict.keys()):
            if source_freq_dict[token] >= cutoff:
                source_vocab.add_token(token)
        for label in list(label_freq_dict.keys()):
            label_vocab.add_token(label)
        return cls(source_vocab, label_vocab)

    def to_serializable(self):
        '''生成序列化信息，方便使用JSON保存初始化信息'''
        return {'source_vocab': self.source_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}

    @classmethod
    def from_serializable(cls, contents):
        '''通过使用contents(序列化后的初始化信息)重建实例'''
        source_vocab = SequenceVocabulary.from_serializable(
            contents['source_vocab'])
        label_vocab = Vocabulary.from_serializable(contents['label_vocab'])
        return cls(source_vocab, label_vocab)
