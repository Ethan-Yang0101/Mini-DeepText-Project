
from Vocabulary.Vocabulary import Vocabulary


class SequenceVocabulary(Vocabulary):

    '''创建一个词典类来管理数据集中每个词和对应索引的关系'''

    def __init__(self, token_index_dict={}, unk_token='<UNK>', mask_token='<MASK>',
                 begin_token='<BOS>', end_token='<EOS>'):
        '''
        Args:
            token_index_dict: 载入预先生成好的词典，若没有会自动生成空词典
            unk_token，mask_token，begin_token, end_token: 文本中的特殊token
        '''
        super(SequenceVocabulary, self).__init__(token_index_dict)
        self._unk_token = unk_token
        self._mask_token = mask_token
        self._begin_token = begin_token
        self._end_token = end_token
        self.unk_index = self.add_token(self._unk_token)
        self.mask_index = self.add_token(self._mask_token)
        self.begin_index = self.add_token(self._begin_token)
        self.end_index = self.add_token(self._end_token)

    def lookup_token(self, token):
        '''查找token在词典中对应的索引，如果token不存在，则返回UNK索引'''
        return self._token_index_dict.get(token, self.unk_index)

    def to_serializable(self):
        '''生成序列化信息，方便使用JSON保存初始化信息'''
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_token': self._begin_token,
                         'end_token': self._end_token})
        return contents
