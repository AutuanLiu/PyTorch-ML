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

# from .utils.logger import Logger


class BaseNet:
    """An abstract class for establish network.

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
        self.visual_dir = configs['visual_dir']
        self.prt_dir = configs['prt_dir']
        self.data_sz = configs['data_sz']
        self.best_model_wts = None
        self.best_acc = 0.
        self.cost_time = 0.
        self.res = {}
        if self.visual_dir is not None:
            self.writer = SummaryWriter(self.visual_dir)
            # self.writer = Logger(self.visual_dir)

    @property
    def best_model(self):
        """return a dict which save the best weights of model."""
        return self.best_model_wts

    @property
    def res_model(self):
        """return a dict which save the loss and acc of model training and valid."""
        return self.res

    def train_m(self):
        """Train and valid(model training and validing each epoch)."""
        # create a file
        if is_file_exist(self.prt_dir):
            os.remove(self.prt_dir)
        with open(self.prt_dir, 'w') as f:
            f.write("Model's training logs\n")

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
                    if phrase == 'train':
                        # zero the buffer of parameters' gradient
                        self.model, data, target = self.trans2gpu(self.model, data, target, volatile=False)
                        self.opt.zero_grad()
                    else:
                        self.model, data, target = self.trans2gpu(self.model, data, target, volatile=True)
                    # forward
                    out = self.model(data)
                    _, y_pred = torch.max(out.data, 1)
                    loss = self.criterion(out, target)

                    # backward in training phrase
                    if phrase == 'train':
                        # zero the buffer of parameters' gradient
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                    # statistics
                    cur_loss += loss.data[0] * data.size(0)
                    cur_corrects += torch.sum(y_pred == target.data)
                epoch_loss = cur_loss / self.data_sz[phrase]
                epoch_acc = cur_corrects / self.data_sz[phrase]
                # save loss and acc
                if phrase == 'train':
                    loss_t.append(epoch_loss)
                    acc_t.append(epoch_acc)
                    # self.visualize({'train_loss': epoch_loss, 'train_acc': epoch_acc}, epoch)
                    # self.visualize({'train_loss': epoch_loss, 'train_acc': epoch_acc}, epoch, scaler=False)
                    self.writer.add_scalars('train', {'train_loss': epoch_loss, 'train_acc': epoch_acc}, epoch)
                else:
                    loss_val.append(epoch_loss)
                    acc_val.append(epoch_acc)
                    # self.visualize({'valid_loss': epoch_loss, 'valid_acc': epoch_acc}, epoch)
                    # self.writer.add_scalars('valid', {'valid_loss': epoch_loss, 'valid_acc': epoch_acc}, epoch)
                self.res.setdefault('loss_train', loss_t)
                self.res.setdefault('acc_train', acc_t)
                self.res.setdefault('loss_val', loss_val)
                self.res.setdefault('acc_val', acc_val)
                f = open(self.prt_dir, 'a')
                if (epoch + 1) % self.prt_freq == 0:
                    print(f'Epoch {(epoch + 1):5d}/{self.epochs} ---> {phrase}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    f.write(f'Epoch {(epoch + 1):5d}/{self.epochs} ---> {phrase}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

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
        f.write(f'Best val acc: {self.best_acc:.4f}\n')
        f.close()

    def test_m(self):
        """Using the best model weights to test the test-dataset."""
        self.model = self.load_model()
        self.model.eval()
        test_loss, correct = 0, 0
        for _, (data, target) in enumerate(self.dataloaders['test']):
            self.model, data, target = self.trans2gpu(self.model, data, target, volatile=True)
            output = self.model(data)
            # sum up batch loss
            test_loss += self.criterion(output, target).data[0]
            # get the index of the max
            _, y_pred = torch.max(output.data, 1)
            correct += torch.sum(y_pred == target.data)

        test_loss /= len(self.dataloaders['test'].dataset)
        len1 = len(self.dataloaders['test'].dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len1} ({100. * correct / len1:.0f}%)')
        with open(self.prt_dir, 'a') as f:
            f.write(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len1} ({100. * correct / len1:.0f}%)')

    def visualize(self, info, epoch, scaler=True):
        """Visualizing the model(training loss, acc etc.)."""
        if scaler:
            for tag, value in info.items():
                self.writer.scalar_summary(tag, value, epoch + 1)
        else:
            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                self.writer.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                self.writer.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

    def load_model(self):
        self.model.load_state_dict(self.best_model_wts)
        # or
        # self.model.load_state_dict(torch.load(self.checkpoint/'best_model.pkl'))
        return self.model

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Save checkpoint and best model."""
        torch.save(state, filename)
        # shutil.copyfile(filename, self.checkpoint / filename)
        shutil.move(filename, self.checkpoint / filename)

    def trans2gpu(self, mod, data, target, volatile=False):
        """ If volatile is setted False, it will ensure that no intermediate states are saved."""
        if gpu:
            inputs, outputs = Variable(data.cuda(), volatile=volatile), Variable(target.cuda())
            model = mod.cuda()
        else:
            inputs, outputs = Variable(data, volatile=volatile), Variable(target)
            model = mod
        return model, inputs, outputs
