
from torch.utils.data import Dataset
from Vectorizer.CLRVectorizer import CLRVectorizer
from Vectorizer.GENVectorizer import GENVectorizer
from Vectorizer.NMTVectorizer import NMTVectorizer
import json

class TextDataset(Dataset):

    '''创建一个数据集类来对数据进行矢量化和划分'''

    def __init__(self, dataset, vectorizer, split_ratio, max_seq_length, task):
        '''
        Args:
            dataset: 数据集
            vectorizer: 由训练集生成的向量化器
            split_ratio: 划分比例
            max_seq_length: 最大句子长度
            task: 任务类型
        '''
        self._dataset = dataset
        self._vectorizer = vectorizer
        self._split_ratio = split_ratio
        self._max_seq_length = max_seq_length
        self._task = task
        self._train_data = self._dataset[0:int(len(self._dataset)*split_ratio[0])]
        self._train_size = len(self._train_data)
        val_ratio = self._split_ratio[0] + self._split_ratio[1]
        self._val_data = self._dataset[int(len(self._dataset)*split_ratio[0]):int(len(self._dataset)*val_ratio)]
        self._val_size = len(self._val_data)
        self._test_data = self._dataset[int(len(self._dataset)*val_ratio):]
        self._test_size = len(self._test_data)
        # 将数据集分划后保存在dict中，通过set_split调取需要使用的数据集
        self._lookup_dict = {'train': (self._train_data, self._train_size),
                             'val': (self._val_data, self._val_size),
                             'test': (self._test_data, self._test_size)}
        self.set_split('train')

    # 根据情况选择当前要使用的数据集，默认使用训练集
    def set_split(self, split='train'):
        self._target_split = split
        self._target_data, self._target_size = self._lookup_dict[split]

    # 定义数据集的长度，用于DataLoader的batch数量计算
    def __len__(self):
        return self._target_size

    # 定义数据集的输出，用于DataLoader的batch数据生成
    def __getitem__(self, index):
        row_dict = self._target_data[index]
        if self._task == 'classification':
            source_vector = self._vectorizer.vectorize(row_dict, self._max_seq_length)
            label_index = self._vectorizer.target_vocab.lookup_token(row_dict["target"][0])
            return {'source_vector': source_vector, 'target': label_index}
        if self._task == 'generation':
            source_vector, target_vector = self._vectorizer.vectorize(row_dict, self._max_seq_length)
            return {'source_vector': source_vector, 'target': target_vector}
        if self._task == 'translation':
            vector_dict = self._vectorizer.vectorize(row_dict, self._max_seq_length)
            return vector_dict

    # 用于之后的vectorizer提取使用
    def get_vectorizer(self):
        return self._vectorizer

    # 通过数据集创建数据集实例
    @classmethod
    def dataset_make_vectorizer(cls, dataset, split_ratio, max_seq_length, task, source_freq_dict,
                                target_freq_dict, cutoff):
        vectorizer = None
        if task == 'classification':
            vectorizer = CLRVectorizer.from_freq_dict(source_freq_dict, target_freq_dict, cutoff)
        if task == 'generation':
            vectorizer = GENVectorizer.from_freq_dict(source_freq_dict, target_freq_dict, cutoff)
        if task == 'translation':
            vectorizer = NMTVectorizer.from_freq_dict(source_freq_dict, target_freq_dict, cutoff)
        return cls(dataset, vectorizer, split_ratio, max_seq_length, task)

    # 通过数据集以及保存好的矢量化器来创建数据集实例
    @classmethod
    def dataset_load_vectorizer(cls, dataset, split_ratio, max_seq_length, task, vectorizer_file):
        vectorizer = cls.load_vectorizer_only(vectorizer_file, task)
        return cls(dataset, vectorizer, split_ratio, max_seq_length, task)

    # 从JSON文件中加载保存好矢量化器
    @staticmethod
    def load_vectorizer_only(vectorizer_file, task):
        with open(vectorizer_file) as fp:
            if task == 'classification':
                return CLRVectorizer.from_serializable(json.load(fp))
            if task == 'generation':
                return GENVectorizer.from_serializable(json.load(fp))
            if task == 'translation':
                return NMTVectorizer.from_serializable(json.load(fp))

    # 将矢量化器保存到JSON文件中
    def save_vectorizer(self, vectorizer_file):
        with open(vectorizer_file, "w", encoding='utf-8') as fp:
            json.dump(self._vectorizer.to_serializable(), fp, ensure_ascii=False)
