
import json
from argparse import Namespace


class Config(object):

    '''创建一个配置类来管理系统的所有变量'''

    def __init__(self, config_dict):
        '''
        Args:
            config_dict: 配置文件转换生成的字典
        '''
        configure_dict = dict()
        data_info_dict = config_dict['data_info']
        train_info_dict = config_dict['train_info']
        save_info_dict = config_dict['save_info']
        task = config_dict['task']
        model_name = config_dict['model_name']
        model_dict = config_dict['model'][model_name]
        configure_dict.update(data_info_dict)
        configure_dict.update(train_info_dict)
        configure_dict.update(save_info_dict)
        configure_dict.update({'task': task})
        configure_dict.update({'model_name': model_name})
        configure_dict.update(model_dict)
        self.args = Namespace(**configure_dict)

    # 读取JSON格式的配置文件，并创建Config实例
    @classmethod
    def from_config_json(cls, json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            config_dict = json.load(json_file)
        return cls(config_dict)
