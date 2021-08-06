
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from TextDataset import TextDataset
from Model.BasicModel.TextCLRModel import TextCLRModel
from Model.BasicModel.TextSLBModel import TextSLBModel
from Model.BasicModel.TextNMTModel import TextNMTModel
from Model.BasicModel.TextDSMModel import TextDSMModel
from Model.Transformer.Transformer import Transformer
from Vectorizer.CLRVectorizer import CLRVectorizer
from Vectorizer.SLBVectorizer import SLBVectorizer
from Vectorizer.NMTVectorizer import NMTVectorizer
from Vectorizer.DSMVectorizer import DSMVectorizer
from Utils.Data import read_json_dataset
from ModelTrainer import ModelTrainer
from Utils.Config import Config
import json
import sys
import os


def get_data_loaders(args, dataset):
    '''通过数据集创建用于训练，验证和测试的数据批生成器'''
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if os.path.exists(args.vectorizer_file):
        parameters = {'dataset': dataset,
                      'split_ratio': args.split_ratio,
                      'max_seq_length': args.max_seq_length,
                      'task': args.task,
                      'vectorizer_file': args.vectorizer_file}
        dataset = TextDataset.dataset_load_vectorizer(**parameters)
    else:
        parameters = {'dataset': dataset,
                      'split_ratio': args.split_ratio,
                      'max_seq_length': args.max_seq_length,
                      'task': args.task,
                      'cutoff': args.cutoff}
        dataset = TextDataset.dataset_make_vectorizer(**parameters)
        dataset.save_vectorizer(args.vectorizer_file)
    dataset.set_split('train')
    train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True)
    dataset.set_split('val')
    val_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                 shuffle=True, drop_last=True)
    dataset.set_split('test')
    test_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
    data_loaders = (train_data_loader, val_data_loader, test_data_loader)
    return data_loaders


def get_task_model(args, vectorizer):
    '''根据任务类型获取用于训练的模型类型'''
    model = None
    if args.task == 'classification':
        if args.model_name == 'TextCLRModel':
            model = TextCLRModel(
                num_embeddings=len(vectorizer.source_vocab),
                embedding_dim=args.embedding_size,
                rnn_hidden_size=args.rnn_hidden_size,
                num_classes=len(vectorizer.label_vocab),
                padding_idx=vectorizer.source_vocab.mask_index,
                batch_first=True)
    if args.task == 'labeling':
        if args.model_name == 'TextSLBModel':
            model = TextSLBModel(
                num_embeddings=len(vectorizer.source_vocab),
                embedding_dim=args.embedding_size,
                rnn_hidden_size=args.rnn_hidden_size,
                padding_idx=vectorizer.source_vocab.mask_index,
                batch_first=True)
    if args.task == 'matching':
        if args.model_name == 'TextDSMModel':
            model = TextDSMModel(
                num_embeddings1=len(vectorizer.source_vocab),
                num_embeddings2=len(vectorizer.target_vocab),
                embedding_dim=args.embedding_size,
                rnn_hidden_size=args.rnn_hidden_size,
                padding_idx=vectorizer.source_vocab.mask_index,
                batch_first=True)
    if args.task == 'translation':
        if args.model_name == 'Transformer':
            model = Transformer(
                source_vocab_size=len(vectorizer.source_vocab),
                target_vocab_size=len(vectorizer.target_vocab),
                source_embed_dim=args.source_embed_dim,
                target_embed_dim=args.target_embed_dim,
                encoder_n_heads=args.encoder_n_heads,
                decoder_n_heads=args.decoder_n_heads,
                encoder_hid_dim=args.encoder_hid_dim,
                decoder_hid_dim=args.decoder_hid_dim,
                encoder_n_layers=args.encoder_n_layers,
                decoder_n_layers=args.decoder_n_layers,
                encoder_max_seq_len=args.max_seq_length,
                decoder_max_seq_len=args.max_seq_length
            )
        if args.model_name == 'TextNMTModel':
            model = TextNMTModel(
                source_num_embeddings=len(vectorizer.source_vocab),
                source_embedding_size=args.source_embedding_size,
                target_num_embeddings=len(vectorizer.target_vocab),
                target_embedding_size=args.target_embedding_size,
                encoding_size=args.encoding_size)
    return model


def get_optimizer(args, model):
    '''获取想要使用的优化器'''
    if args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate)


def get_loss_func(args):
    '''根据任务类型获取损失函数'''
    if args.task == 'classification':
        return nn.CrossEntropyLoss()
    if args.task == 'matching':
        return nn.CrossEntropyLoss()
    if args.task == 'labeling':
        return sequence_loss
    if args.task == 'translation':
        return sequence_loss


def sequence_loss(pred, target, mask_index):
    '''用于计算序列模型的损失函数'''
    pred = pred.contiguous().view(-1, pred.size(2))
    target = target.contiguous().view(-1)
    return F.cross_entropy(pred, target, ignore_index=mask_index)


def get_vectorizer(args):
    '''根据任务获取矢量化器'''
    with open(args.vectorizer_file, "r") as fp:
        if args.task == 'classification':
            return CLRVectorizer.from_serializable(json.load(fp))
        if args.task == 'matching':
            return DSMVectorizer.from_serializable(json.load(fp))
        if args.task == 'labeling':
            return GENVectorizer.from_serializable(json.load(fp))
        if args.task == 'translation':
            return NMTVectorizer.from_serializable(json.load(fp))


if __name__ == '__main__':
    # 获取配置文件信息
    config_filename = sys.argv[1]
    config = Config.from_config_json(config_filename)
    args = config.args
    # 获取数据集
    dataset = read_json_dataset(args.data_filepath, args.max_seq_length)
    # 获取数据批生成器
    data_loaders = get_data_loaders(args, dataset)
    # 获取模型
    vectorizer = get_vectorizer(args)
    model = get_task_model(args, vectorizer)
    # 获取优化器
    optimizer = get_optimizer(args, model)
    # 获取损失函数
    loss_func = get_loss_func(args)
    # 获取训练器
    model_trainer = ModelTrainer(
        args, data_loaders, model, optimizer, loss_func)
    # 训练模型
    model_trainer.train_val_test_model()
