
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

class ModelTrainer(object):

    '''创建模型训练器'''

    def __init__(self, args, dataloaders, model, optimizer, loss_func):
        '''
        Args:
            args: 模型训练的配置参数集
            dataloders: 3个数据批生成器
            model: 训练模型
            optimizer: 优化器
            loss_func: 损失函数
            train_state: 训练状态
        '''
        self.args = args
        self.dataloaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_state = self.make_train_state()

    # 训练，验证和测试模型
    def train_val_test_model(self):
        try:
            for epoch_index in range(self.args.num_epochs):
                self.train_state['epoch_index'] = epoch_index
                self.run_model('train')
                self.run_model('eval')
                self.train_state = self.update_train_state(
                    self.model, self.train_state)
                if self.train_state['stop_early']:
                    print('Early Stop Training!')
                    break
            self.run_model('test')
        except KeyboardInterrupt:
            print("Exiting loop")

    # 初始化训练状态
    def make_train_state(self):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8,
                'learning_rate': self.args.learning_rate,
                'epoch_index': 0,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': [],
                'test_acc': [],
                'model_filename': self.args.model_state_file}

    # 更新训练状态
    def update_train_state(self, model, train_state):
        if train_state['epoch_index'] == 0:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['stop_early'] = False
        elif train_state['epoch_index'] >= 1:
            loss_pre_t, loss_t = train_state['val_loss'][-2:]
            if loss_t >= loss_pre_t:
                train_state['early_stopping_step'] += 1
            else:
                if loss_t < train_state['early_stopping_best_val']:
                    torch.save(model.state_dict(), train_state['model_filename'])
                    train_state['early_stopping_best_val'] = loss_t
                    train_state['early_stopping_step'] = 0
        train_state['stop_early'] = train_state['early_stopping_step'] >= \
             self.args.early_stopping_criteria
        return train_state

    # 计算模型准确度
    def compute_accuracy(self, pred, target, mask, task):
        if task == 'classification':
            _, pred_indices = pred.max(dim=1)
            n_correct = torch.eq(pred_indices, target).sum().item()
            return n_correct / len(pred_indices) * 100
        if task == 'generation' or task == 'translation':
            if len(pred.size()) == 3:
                pred = pred.contiguous().view(-1, pred.size(2))
            if len(target.size()) == 2:
                target = target.contiguous().view(-1)
            _, pred_indices = pred.max(dim=1)
            correct_indices = torch.eq(pred_indices, target).float()
            valid_indices = torch.ne(target, mask).float()
            n_correct = (correct_indices * valid_indices).sum().item()
            n_valid = valid_indices.sum().item()
        return n_correct / n_valid * 100

    # 根据训练模式来训练模型
    def run_model(self, train_mode):
        dataloader, loss_type, acc_type = None, None, None
        if train_mode == 'train':
            self.model.train()
            dataloader = self.dataloaders[0]
            loss_type = 'train_loss'
            acc_type = 'train_acc'
        if train_mode == 'eval':
            self.model.eval()
            dataloader = self.dataloaders[1]
            loss_type = 'val_loss'
            acc_type = 'val_acc'
        if train_mode == 'test':
            self.model.eval()
            dataloader = self.dataloaders[2]
            loss_type = 'test_loss'
            acc_type = 'test_acc'
        running_loss, running_acc = 0.0, 0.0
        for batch_index, batch_dict in enumerate(dataloader):
            pred, loss = None, None
            if self.args.task == 'classification':
                pred = self.model(batch_dict['source_vector'])
                loss = self.loss_func(pred, batch_dict['target'])
            if self.args.task == 'generation':
                pred = self.model(batch_dict['source_vector'])
                loss = self.loss_func(pred, batch_dict['target'], mask_index=0)
            if self.args.task == 'translation':
                pred = self.model(batch_dict['source_vector'], batch_dict['source_length'],
                                  batch_dict['target_input_vector'])
                loss = self.loss_func(pred, batch_dict['target'], mask_index=0)
            if train_mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            running_loss += (loss.item() - running_loss) / (batch_index + 1)
            acc_t = self.compute_accuracy(pred, batch_dict['target'], mask=0, task=self.args.task)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        self.train_state[loss_type].append(running_loss)
        self.train_state[acc_type].append(running_acc)
        print("Epoch: {} / {} -- {} Loss: {:.3f}, {} Accuracy: {:.3f}%".format(
            self.train_state['epoch_index'] +1, self.args.num_epochs, train_mode.capitalize(),
            self.train_state[loss_type][-1], train_mode.capitalize(), self.train_state[acc_type][-1]))
