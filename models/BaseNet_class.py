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

    train_m and test_m must function can be reimplement in the subclass.

    Attribute:
    ----------
    config: dict
        The config of model.
    """

    def __init__(self, configs):
        """initial the network
        
        Parameters:
        ----------
        configs : dict
            The configs of the network
        """
        self.model = configs['model']
        self.opt = configs['opt']
        self.criterion = configs['criterion']
        self.dataloaders = configs['dataloaders']
        self.lrs_decay = configs['lrs_decay']
        self.prt_freq = configs['prt_freq']
        self.epochs = configs['epochs']
        self.checkpoint = configs['checkpoint']
        self.best_model_wts = None
        self.best_acc = 0.
        self.cost_time = 0.
        self.res = {}

    @property
    def best_model(self):
        """return a dict which save the best weights of model."""
        return self.best_model_wts

    @property
    def res_model(self):
        """return a dict which save the loss and acc of model training and valid."""
        return self.res

    def train_m(self):
        since = time.time()
        loss_t, acc_t, loss_val, acc_val = [], [], [], []
        # save best model weights
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.epochs):
            # Each epoch has a training and validation phase
            for phrase in ['train', 'valid']:
                if phrase == 'train':
                    if 'lrs_decay' is not None:
                        # update learning rates
                        self.lrs_decay.step()
                    self.model.train(True)
                else:
                    self.model.train(False)

                # record the current epoch loss and corrects
                cur_loss, cur_corrects = 0., 0

                # train over minibatch
                for _, (data, target) in enumerate(self.dataloaders[phrase]):
                    if gpu:
                        data, target = Variable(data.cuda()), Variable(target.cuda())
                        self.model = self.model.cuda()
                    else:
                        data, target = Variable(data), Variable(target)

                    # zero the buffer of parameters' gradient
                    self.opt.zero_grad()

                    # forward
                    out = self.model(data)
                    _, y_pred = torch.max(out.data, 1)
                    loss = self.criterion(out, target)

                    # backward in training phrase
                    if phrase == 'train':
                        loss.backward()
                        self.opt.step()

                    # statistics
                    cur_loss += loss.data[0] * data.size(0)
                    cur_corrects += torch.sum(y_pred == target.data)

                epoch_loss = cur_loss / len(self.dataloaders[phrase].dataset)
                epoch_acc = cur_corrects / len(self.dataloaders[phrase].dataset)
                # save loss and acc
                if phrase == 'train':
                    loss_t.append(epoch_loss)
                    acc_t.append(epoch_acc)
                else:
                    loss_val.append(epoch_loss)
                    acc_val.append(epoch_acc)
                self.res.setdefault('loss_train', loss_t)
                self.res.setdefault('acc_train', acc_t)
                self.res.setdefault('loss_val', loss_val)
                self.res.setdefault('acc_val', acc_val)
                if (epoch + 1) % self.prt_freq == 0:
                    print(f'Epoch {epoch + 1}/{self.epochs} ---> {phrase}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # save the best model
                if phrase == 'valid' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
        # save model
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_acc,
                'optimizer': self.opt.state_dict(),
            }
            self.save_checkpoint(checkpoint, f'checkpoint{epoch + 1}.pth.tar')
        self.save_checkpoint(self.best_model_wts, 'best_model.pkl')
        self.cost_time = time.time() - since
        print(f'Training complete in {int(self.cost_time // 60)}m{self.cost_time % 60}s.')
        print(f'Best val acc: {self.best_acc:.4f}')

    def test_m(self):
        self.model = self.load_model()
        self.model.eval()
        test_loss, correct = 0, 0
        for _, (data, target) in enumerate(self.dataloaders['test']):
            if gpu:
                data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
                self.model = self.model.cuda()
            else:
                data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            # sum up batch loss
            test_loss += self.criterion(output, target).data[0]
            # get the index of the max
            _, y_pred = output.data.max(1, keepdim=True)
            correct += y_pred.eq(target.data.view_as(y_pred)).cpu().sum()

        test_loss /= len(self.dataloaders['test'].dataset)
        len1 = len(self.dataloaders['test'].dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len1} ({100. * correct / len1:.0f}%)\n')

    def visualize(self):
        raise NotImplementedError

    def load_model(self):
        self.model.load_state_dict(self.best_model_wts)
        # or
        # self.model.load_state_dict(torch.load(self.checkpoint/'best_model.pkl'))
        return self.model

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        # shutil.copyfile(filename, self.checkpoint / filename)
        shutil.move(filename, self.checkpoint / filename)
