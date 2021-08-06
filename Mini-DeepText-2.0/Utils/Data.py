
import json
from collections import Counter


def read_json_dataset(filepath, max_seq_length):
    '''读取JSON格式的数据集'''
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as fp:
        for json_string in fp.readlines():
            json_dict = json.loads(json_string)
            if len(json_dict['source']) > max_seq_length:
                json_dict['source'] = json_dict['source'][:max_seq_length]
            if len(json_dict['target']) > max_seq_length:
                json_dict['target'] = json_dict['target'][:max_seq_length]
            dataset.append(json_dict)
    return dataset


def generate_token_freq_dict(dataset):
    '''根据数据集获取词频字典'''
    source_token_count = Counter()
    target_token_count = Counter()
    label_token_count = Counter()
    for row_dict in dataset:
        for token in row_dict['source']:
            source_token_count[token] += 1
        for token in row_dict['target']:
            target_token_count[token] += 1
        for token in row_dict['label']:
            label_token_count[token] += 1
    source_freq_dict = dict(source_token_count.items())
    target_freq_dict = dict(target_token_count.items())
    label_freq_dict = dict(label_token_count.items())
    return source_freq_dict, target_freq_dict, label_freq_dict
