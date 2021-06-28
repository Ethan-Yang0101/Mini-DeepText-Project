
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from TextDataset import TextDataset
from Model.TextClrModel import TextClrModel
from Model.TextGenModel import TextGenModel
from Model.TextNMTModel import TextNMTModel
from Vectorizer.CLRVectorizer import CLRVectorizer
from Vectorizer.GENVectorizer import GENVectorizer
from Vectorizer.NMTVectorizer import NMTVectorizer
from Data import read_json_dataset
from Data import generate_token_freq_dict
from ModelTrainer import ModelTrainer
from Config import Config
import json
import sys
import os

# 通过数据集和词频字典创建用于训练，验证和测试的数据批生成器
def get_data_loader(args, data, source_freq_dict, target_freq_dict):
    dataset = None
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if os.path.exists(args.vectorizer_file):
        parameters = {'dataset': data,
                      'split_ratio': args.split_ratio,
                      'max_seq_length': args.max_seq_length,
                      'task': args.task,
                      'vectorizer_file': args.vectorizer_file}
        dataset = TextDataset.dataset_load_vectorizer(**parameters)
    else:
        parameters = {'dataset': data,
                      'split_ratio': args.split_ratio,
                      'max_seq_length': args.max_seq_length,
                      'task': args.task,
                      'source_freq_dict': source_freq_dict,
                      'target_freq_dict': target_freq_dict,
                      'cutoff': args.cutoff}
        dataset = TextDataset.dataset_make_vectorizer(**parameters)
        dataset.save_vectorizer(args.vectorizer_file)
    dataset.set_split('train')
    train_data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset.set_split('val')
    val_data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset.set_split('test')
    test_data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data_loaders = (train_data_loader, val_data_loader, test_data_loader)
    return data_loaders

# 根据任务类型获取用于训练的模型类型
def get_task_model(args, vectorizer):
    model = None
    if args.task == 'classification':
        model = TextClrModel(
            num_embeddings=len(vectorizer.source_vocab),
            embedding_dim=args.embedding_size,
            rnn_hidden_size=args.rnn_hidden_size,
            num_classes=len(vectorizer.target_vocab),
            padding_idx=vectorizer.source_vocab.mask_index,
            batch_first=True)
    if args.task == 'generation':
        model = TextGenModel(
            num_embeddings=len(vectorizer.source_vocab),
            embedding_dim=args.embedding_size,
            rnn_hidden_size=args.rnn_hidden_size,
            padding_idx=vectorizer.source_vocab.mask_index,
            batch_first=True)
    if args.task == 'translation':
        model = TextNMTModel(
            source_num_embeddings=len(vectorizer.source_vocab),
            source_embedding_size=args.source_embedding_size,
            target_num_embeddings=len(vectorizer.target_vocab),
            target_embedding_size=args.target_embedding_size,
            encoding_size=args.encoding_size)
    return model

# 获取想要使用的优化器
def get_optimizer(args, model):
    if args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate)

# 根据任务类型获取损失函数
def get_loss_func(args):
    if args.task == 'classification':
        return nn.CrossEntropyLoss()
    if args.task == 'generation':
        return sequence_loss
    if args.task == 'translation':
        return sequence_loss

# 用于计算序列模型的损失函数
def sequence_loss(pred, target, mask_index):
    if len(pred.size()) == 3:
        pred = pred.contiguous().view(-1, pred.size(2))
    if len(target.size()) == 2:
        target = target.contiguous().view(-1)
    return F.cross_entropy(pred, target, ignore_index=mask_index)

# 根据任务获取矢量化器
def get_vectorizer(args):
    with open(args.vectorizer_file, "r") as fp:
        if args.task == 'classification':
            return CLRVectorizer.from_serializable(json.load(fp))
        if args.task == 'generation':
            return GENVectorizer.from_serializable(json.load(fp))
        if args.task == 'translation':
            return NMTVectorizer.from_serializable(json.load(fp))

if __name__ == '__main__':
    # 获取配置文件信息
    config_filename = sys.argv[1]
    config = Config.from_config_json(config_filename)
    args = config.args
    # 获取数据集和词频字典
    data = read_json_dataset(args.data_filepath, args.max_seq_length)
    source_freq_dict, target_freq_dict = generate_token_freq_dict(data)
    # 获取数据批生成器
    data_loaders = get_data_loader(args, data, source_freq_dict, target_freq_dict)
    # 获取模型
    vectorizer = get_vectorizer(args)
    model = get_task_model(args, vectorizer)
    # 获取优化器
    optimizer = get_optimizer(args, model)
    # 获取损失函数
    loss_func = get_loss_func(args)
    # 获取训练器
    model_trainer = ModelTrainer(args, data_loaders, model, optimizer, loss_func)
    # 训练模型
    model_trainer.train_val_test_model()
