
from torch.utils.data import Dataset
from Vectorizer.CLRVectorizer import CLRVectorizer
from Vectorizer.SLBVectorizer import SLBVectorizer
from Vectorizer.NMTVectorizer import NMTVectorizer
from Vectorizer.DSMVectorizer import DSMVectorizer
from Utils.Data import generate_token_freq_dict
import json


class TextDataset(Dataset):

    '''创建一个数据集类来对数据进行矢量化和划分'''

    def __init__(self, dataset, vectorizer, split_ratio, max_seq_length, task):
        '''
        Args:
            dataset: 数据集 (List[Dict])
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
        self._lookup_dict = self.train_val_test_split(dataset, split_ratio)
        self.set_split('train')

    def set_split(self, split='train'):
        '''根据情况选择当前要使用的数据集，默认使用训练集'''
        self._target_split = split
        self._target_data, self._target_size = self._lookup_dict[split]

    def train_val_test_split(self, dataset, split_ratio):
        '''按比例将数据集划分为训练集，验证集和测试集'''
        train_data = dataset[0:int(len(dataset)*split_ratio[0])]
        train_size = len(train_data)
        val_ratio = split_ratio[0] + split_ratio[1]
        val_data = dataset[int(len(dataset)*split_ratio[0])
                               :int(len(dataset)*val_ratio)]
        val_size = len(val_data)
        test_data = dataset[int(len(dataset)*val_ratio):]
        test_size = len(test_data)
        return {'train': (train_data, train_size),
                'val': (val_data, val_size),
                'test': (test_data, test_size)}

    def __len__(self):
        '''定义数据集的长度，用于DataLoader的batch数量计算'''
        return self._target_size

    def __getitem__(self, index):
        '''定义数据集的输出，用于DataLoader的batch数据生成'''
        row_dict = self._target_data[index]
        vector_dict = self._vectorizer.vectorize(
            row_dict, self._max_seq_length)
        return vector_dict

    def get_vectorizer(self):
        '''用于之后的vectorizer提取使用'''
        return self._vectorizer

    @classmethod
    def dataset_make_vectorizer(cls, dataset, split_ratio, max_seq_length, task, cutoff):
        '''通过数据集创建数据集实例'''
        vectorizer = None
        source_freq_dict, target_freq_dict, label_freq_dict = generate_token_freq_dict(
            dataset)
        if task == 'classification':
            vectorizer = CLRVectorizer.from_freq_dict(
                source_freq_dict, label_freq_dict, cutoff)
        if task == 'matching':
            vectorizer = DSMVectorizer.from_freq_dict(
                source_freq_dict, target_freq_dict, label_freq_dict, cutoff)
        if task == 'labeling':
            vectorizer = SLBVectorizer.from_freq_dict(
                source_freq_dict, target_freq_dict, cutoff)
        if task == 'translation':
            vectorizer = NMTVectorizer.from_freq_dict(
                source_freq_dict, target_freq_dict, cutoff)
        return cls(dataset, vectorizer, split_ratio, max_seq_length, task)

    @classmethod
    def dataset_load_vectorizer(cls, dataset, split_ratio, max_seq_length, task, vectorizer_file):
        '''通过数据集以及保存好的矢量化器来创建数据集实例'''
        vectorizer = cls.load_vectorizer_only(vectorizer_file, task)
        return cls(dataset, vectorizer, split_ratio, max_seq_length, task)

    @staticmethod
    def load_vectorizer_only(vectorizer_file, task):
        '''从JSON文件中加载保存好矢量化器'''
        with open(vectorizer_file) as fp:
            if task == 'classification':
                return CLRVectorizer.from_serializable(json.load(fp))
            if task == 'matching':
                return DSMVectorizer.from_serializable(json.load(fp))
            if task == 'labeling':
                return SLBVectorizer.from_serializable(json.load(fp))
            if task == 'translation':
                return NMTVectorizer.from_serializable(json.load(fp))

    # 将矢量化器保存到JSON文件中
    def save_vectorizer(self, vectorizer_file):
        with open(vectorizer_file, "w", encoding='utf-8') as fp:
            json.dump(self._vectorizer.to_serializable(),
                      fp, ensure_ascii=False)
