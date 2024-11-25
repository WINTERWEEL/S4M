import random
from thop import profile

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric,MAE,MSE,RMSE,MAPE,MSPE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import wandb
import warnings
import numpy as np
import pandas as pd

from torchstat import stat

warnings.filterwarnings('ignore')

def save_model1(i,iter,model,path):
    path = path + '/' + 'checkpoint.pth'
    path = path[:-4]+str(i)+"_"+str(iter)+path[-4:]
    print("save_model1",path)
    dic = {}
    # dic['model'] = model.state_dict()
    # dic['memory'] = model.mem_net.memory.memory
    dic['repre'] = model.mem_net.memory.repre
    dic['num_dic'] = model.mem_net.memory.num_dic
    torch.save(dic,path)    
    
def load_model1(i,iter,model,path):
    path = path[:-4]+str(i)+"_"+str(iter)+path[-4:]
    print(path)
    dic = torch.load(path)
    # model.load_state_dict(dic['model'],strict=False)
    # model.mem_net.memory.memory = dic['memory']
    model.mem_net.memory.repre = dic['repre']
    model.mem_net.memory.num_dic = dic['num_dic']
    return model





class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.SGD(self.model.parameters(), lr = self.args.learning_rate, momentum=0.9,weight_decay=1e-5)
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def change_mode(self,is_training=1):
        self.model.mem_net.is_training=is_training
        self.model.is_training=is_training
            
    def save_model(self,path):
        dic = {}
        dic['model'] = self.model.state_dict()

        dic['memory'] = self.model.mem_net.memory.memory
        dic['repre'] = self.model.mem_net.memory.repre
        dic['num_dic'] = self.model.mem_net.memory.num_dic   
        torch.save(dic,path)
        
    def load_model(self,path):
        dic = torch.load(path)
        self.model.load_state_dict(dic['model'],strict=False)
        self.model.mem_net.memory.memory = dic['memory']
        self.model.mem_net.memory.repre = dic['repre']
        self.model.mem_net.memory.num_dic = dic['num_dic']
        return self.model

    def vali(self, vali_data, vali_loader, criterion,epoch=0):
        total_loss = []
        f_dim = -1 if self.args.features == 'MS' else 0
        self.model.eval()
        self.change_mode(0)
        with torch.no_grad():
            for i, (seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value) in enumerate(vali_loader):
                seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value \
                    = seq_x.float().to(self.device),seq_x_mask.float().to(self.device),seq_y.float().to(self.device),seq_y_mask.float().to(self.device),seq_true_x.float().to(self.device),seq_true_y.float().to(self.device),max_idx.float().to(self.device),min_idx.float().to(self.device),max_value.float().to(self.device),min_value.float().to(self.device)
                if (seq_y_mask).sum()==0:
                    continue
                seq_y = seq_y[:, -self.args.pred_len:, f_dim:]
                outputs = self.model(seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = outputs.detach()
                seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:].detach()
                batch_y = seq_y.detach()
                loss = criterion(pred[seq_y_mask==1.0], batch_y[seq_y_mask==1.0])
                total_loss.append(loss.item())
        total_loss = sum(total_loss)/len(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # print(train_data.data_x)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        f_dim = -1 if self.args.features == 'MS' else 0
    
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(args =self.args,patience=self.args.patience, verbose=True,more_save=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            pred = []
            self.model.train()
            self.change_mode(1)
            if not self.args.no_renew:
                if "Sea" in self.args.model:
                    self.model.mem_net1.memory.init_bank()
                    self.model.mem_net2.memory.init_bank()
                else:
                    self.model.mem_net.memory.init_bank()                
            epoch_time = time.time()
            self.model.count=0
            print("lr: ",model_optim.state_dict()['param_groups'][0]['lr'])
            for i, (seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value) in enumerate(train_loader):
                seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value \
                    = seq_x.float().to(self.device),seq_x_mask.float().to(self.device),seq_y.float().to(self.device),seq_y_mask.float().to(self.device),seq_true_x.float().to(self.device),seq_true_y.float().to(self.device),max_idx.float().to(self.device),min_idx.float().to(self.device),max_value.float().to(self.device),min_value.float().to(self.device)
                iter_count += 1
                model_optim.zero_grad()
                if i==0 and epoch==0:
                    outputs =self.model.warmup(seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value)
                    continue
                else:
                    outputs = self.model(seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value)

                seq_y = seq_y[:, -self.args.pred_len:, f_dim:]
                seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                seq_y_mask = seq_y_mask.detach()
                
                loss = criterion(outputs[seq_y_mask==1.0], seq_y[seq_y_mask==1.0])
                train_loss.append(loss.item())
                if (i + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    if self.args.analysis:
                        save_model1(epoch,i,self.model,path)
                loss.backward()
                model_optim.step()       
                
            # save_model1(epoch,self.model,path)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion,epoch)
            wandb.log({'valid_loss':vali_loss,'train_loss':train_loss})
            wandb.log({"num_bank1":self.model.mem_net.memory.num_bank()})
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        best_model_path = path + '/' + 'checkpoint.pth'
        wandb.save(best_model_path)

        self.load_model(best_model_path)
        return self.model

    def test(self, setting, test=0,epoch=0):
        test_data, test_loader = self._get_data(flag='test')
        f_dim = -1 if self.args.features == 'MS' else 0
        criterion = self._select_criterion()
        
        if test:
            print('loading model')
            self.load_model(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        folder_path = '/home/project/S4M/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.change_mode(0)
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value) in enumerate(test_loader):
                seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value \
                    = seq_x.float().to(self.device),seq_x_mask.float().to(self.device),seq_y.float().to(self.device),seq_y_mask.float().to(self.device),seq_true_x.float().to(self.device),seq_true_y.float().to(self.device),max_idx.float().to(self.device),min_idx.float().to(self.device),max_value.float().to(self.device),min_value.float().to(self.device)

                seq_y = seq_y[:, -self.args.pred_len:, f_dim:]
                seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                outputs = self.model(seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = outputs.detach().cpu()
                seq_y = seq_y[:,-self.args.pred_len:, f_dim:].detach().cpu().numpy()
                seq_y_true = seq_true_y[:,-self.args.pred_len:,f_dim:].detach().cpu()

                loss = criterion(pred, seq_y_true)
                total_loss.append(loss.item())
                seq_y_true = seq_y_true.numpy()
                pred = pred.cpu().numpy()
                trues.append(seq_y_true)
                preds.append(pred)
                
                seq_true_x = seq_true_x.detach().cpu().numpy()
                if i%100==0:
                    if test_data.scale:
                        pred = pred[0,:,:]
                        pred = test_data.inverse_transform1(pred)
                        seq_true_y = seq_y_true[0,:,:]
                        seq_true_y = test_data.inverse_transform5(seq_true_y)
                        seq_true_x = seq_true_x[0,:,:]
                        seq_true_x = test_data.inverse_transform5(seq_true_x)
                    seq_x_mask = seq_x_mask.detach().cpu().numpy()
                    seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                    for j in [-1,0,1]:
                        true_xy = np.concatenate((seq_true_x[:,j],seq_true_y[:,j]),axis=0)
                        pre = pred[:,j]
                        mask = np.concatenate((seq_x_mask[0,:,j],seq_y_mask[0,:,j]),axis=0)
                        visual(true_xy,pre,mask, os.path.join(folder_path, str(i) + '_'+str(j)+'.png'))
                        wandb.save(os.path.join(folder_path, str(i) + '_'+str(j)+'.png'))
                self.args.plot=0
                    
        mae, mse, rmse, mape, mspe = 0.0,0.0,0.0,0.0,0.0
        
        for i in range(len(preds)):
            mae+= MAE(preds[i],trues[i])
            mse+=MSE(preds[i],trues[i])
            rmse+=RMSE(preds[i],trues[i])
            mape+=MAPE(preds[i],trues[i])
            mspe+=MSPE(preds[i],trues[i])
            
        mae,mse,rmse,mape,mspe = mae/len(preds), mse/len(preds), rmse/len(preds), mape/len(preds), mspe/len(preds)
        total_loss = np.average(total_loss)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        wandb.log({"mse":mse,'mae':mae,'rmse':rmse,'mape':mape,'mspe':mspe,"total_loss":total_loss})

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        return
    
    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        inputs = []
        for i, data_i in enumerate(test_loader):
            input = data_i[0].detach().cpu().numpy()
            inputs.append((input))
        folder_path = './results/' + setting + '/'
        np.save(folder_path + 'input.npy', inputs)