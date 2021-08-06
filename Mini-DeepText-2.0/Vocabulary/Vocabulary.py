
class Vocabulary(object):

    '''创建一个词典类来管理数据集中每个词和对应索引的关系'''

    def __init__(self, token_index_dict={}):
        '''
        Args:
            token_index_dict: 载入预先生成好的词典，若没有会自动生成空词典
        '''
        self._token_index_dict = token_index_dict
        self._index_token_dict = {
            index: token for token, index in token_index_dict.items()}

    def add_token(self, token):
        '''向词典中加入token，并返回token在词典中所在的索引，若token已存在，直接返回索引'''
        if token in self._token_index_dict:
            return self._token_index_dict[token]
        else:
            index = len(self._token_index_dict)
            self._token_index_dict[token] = index
            self._index_token_dict[index] = token
            return index

    def lookup_token(self, token):
        '''查找token在词典中的对应索引'''
        return self._token_index_dict[token]

    def lookup_index(self, index):
        '''查找索引在词典中对应的token'''
        return self._index_token_dict[index]

    def to_serializable(self):
        '''生成序列化信息，方便使用JSON保存初始化信息'''
        return {'token_index_dict': self._token_index_dict}

    @classmethod
    def from_serializable(cls, contents):
        '''通过使用contents(序列化后的初始化信息)重建实例'''
        return cls(**contents)

    def __str__(self):
        '''Print打印实例的输出结果'''
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        '''定义实例的长度信息为词典的长度'''
        return len(self._token_index_dict)
