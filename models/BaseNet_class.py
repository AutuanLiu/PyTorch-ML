#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  A abstract class for establish network
    1. https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    2. https://github.com/gngdb/pytorch-cifar-sgdr
    3. http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    Email : autuanliu@163.com
    Date：2018/04/02
"""
from .utils.utils_imports import *
from .vislib.vis_imports import *


class BaseNet:
    """A abstract class for establish network.

    train_m and test_m must be implement in the subclass.

    Attribute:
    ----------
    config: dict
        The config of model.
    """

    def __init__(self, config):
        """initial the network
        
        Parameters:
        ----------
        config : dict
            The configs of the network
        """
        self.config = config
        self.loss = []
        self.accuracy = []
        self.best_model_wts = None
        self.best_acc = 0.
        self.cost_time = 0.

    @property
    def config(self):
        """return a dict of model's configure."""
        return self.config
    
    @config.setter
    def set_config(self, value):
        if not isinstance(value, dict):
            raise ValueError('configure must be a Dict!')
        self.config = value

    @property
    def loss(self):
        """return a list of model's loss."""
        return self.loss

    @property
    def accuracy(self):
        """return a list of model's accuracy.(valid)"""
        return self.accuracy
    
    @property
    def best_model(self):
        """return a dict which save the best weights of model."""
        return self.best_model_wts

    def train_m(self):
        since = time.time()
        # save best model weights
        self.best_model_wts = copy.deepcopy(self.config['model'].state_dict())
        for epoch in range(self.config['epochs']):
            # Each epoch has a training and validation phase
            for phrase in ['train', 'valid']:
                if phrase == 'train':
                    if 'lrs_decay' in self.config:
                        # update learning rates
                        self.config['lrs_decay'].step()
                    self.config['model'].train(True)
                else:
                    self.config['model'].train(False)
                
                # record the current epoch loss and corrects
                cur_loss, cur_corrects = 0., 0

                # train over minibatch
                for batch_idx, (data, target) in enumerate(self.config['dataloaders'][phrase]):
                    if gpu:
                        data, target = Variable(data.cuda()), Variable(target.cuda())
                    else:
                        data, target = Variable(data), Variable(target)
                    
                    # zero the buffer of parameters' gradient
                    self.config['opt'].zero_grad()

                    # forward
                    out = self.config['model'](data)
                    _, y_pred = torch.max(out.data, 1)
                    loss = self.config['criterion'](out, target)

                    # backward in training phrase
                    if phrase == 'train':
                        loss.backward()
                        self.config['opt'].step()
                    
                    # statistics
                    cur_loss += loss.data[0] * data.size(0)
                    cur_corrects += torch.sum(y_pred == traget.data)

                epoch_loss = cur_loss / self.config['data_sz'][phrase]
                epoch_acc = cur_corrects / self.config['data_sz'][phrase]
                if (epoch + 1) % self.config['prt_freq'] == 0:
                    print(f'Epoch {epoch}/{self.config['epochs'] - 1}.\n', '*-*' * 20)
                    print(f'{phrase}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # save the best model
                if phrase == 'valid' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.config['model'].state_dict())
        self.cost_time = time.time() - since
        print(f'Training complete in {int(self.cost_time // 60)}m{self.cost_time % 60}s.')
        print(f'Best val acc: {self.best_acc:.4f}')
        return self.load_model()
                        
    def test_m(self):
        self.config['model'].eval()

    def visualize(self):
        raise NotImplementedError
    
    def load_model(self):
        self.config['model'].load_state_dict(self.best_model_wts)
        return self.config['model']
