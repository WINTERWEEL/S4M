import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self,args,patience=7, verbose=False, delta=1E-3,more_save=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.more_save = more_save
        self.args = args

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if self.more_save:
            dic = {}
            dic['model'] = model.state_dict()
            if "Sea" in self.args.model:
                dic['memory1'] =model.mem_net1.memory.memory
                dic['repre1'] = model.mem_net1.memory.repre
                dic['num_dic1'] =model.mem_net1.memory.num_dic   
                dic['memory2'] = model.mem_net2.memory.memory
                dic['repre2'] =model.mem_net2.memory.repre
                dic['num_dic2'] = model.mem_net2.memory.num_dic   
            else:
                dic['memory'] = model.mem_net.memory.memory
                dic['repre'] = model.mem_net.memory.repre
                dic['num_dic'] =model.mem_net.memory.num_dic   
            torch.save(dic, path + '/' + 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
            
        self.val_loss_min = val_loss
        


def remasker(seq_x,ratio):
    num = seq_x.shape[0]*seq_x.shape[1]*seq_x.shape[2]
    remask = torch.zeros(num)
    remask[torch.randperm(num)[:int(num*ratio)]] = 1
    remask = remask.reshape(seq_x.shape)
    return remask

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true,pre,mask,name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(12,7))
    idx = np.arange(true.shape[0])
    idx1 = idx[-pre.shape[0]:]
    
    plt.plot(idx1,pre, label='Prediction', linewidth=1,color = "orange")
    
    plot_dash(true,mask,color = "blue",label="GroundTruth")
    # plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    
    
def plot_dash(data,mask,color,label="GroundTruth"):
    holder = []
    pre_mask=  mask[0]
    idx = []
    for i in range(mask.shape[0]):
        if mask[i]==pre_mask:
            holder.append(data[i])
            idx.append(i)
        else:
            if pre_mask==0:
                plt.plot(idx,holder,linestyle='--',color=color,linewidth=1)
            else:
                plt.plot(idx,holder,linestyle='-',color=color,linewidth=1)
            holder=[holder[-1]]
            idx = [idx[-1]]
            pre_mask = mask[i]
            holder.append(data[i])
            idx.append(i)
    if pre_mask==0:
        plt.plot(idx,holder,linestyle='--',color=color,linewidth=1)
    else:
        plt.plot(idx,holder,linestyle='-',color=color,linewidth=1,label="GroundTruth")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
