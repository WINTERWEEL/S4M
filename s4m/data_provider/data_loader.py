import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')




class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,features='S', data_path='ETTh1.csv',target='OT', scale=True, timeenc=0, freq='h',mask=None,pad='zero'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.mask = mask
        self.pad = pad

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.mask:
            mask = df_raw.copy()
            mask.iloc[:,1:]=1.0
            mask = mask.where(~df_raw.isna(),0.0)
        if self.pad=='zero':
            df_raw = df_raw.where(~df_raw.isna(),0.0)

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            if self.mask:
                mask = mask[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            if self.mask:
                mask = mask[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            if self.mask:
                mask = mask[border1s[0]:border2s[0]]
                mask = mask.values()
        else:
            data = df_data.values
            if self.mask:
                mask = mask.values()

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data[border1:border2]
        if self.mask:
            self.mask_x = mask[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        if self.mask:
            seq_mask = self.mask_x[s_begin:s_end]
            return seq_x,seq_mask,seq_y,seq_x_mark,seq_y_mark
        else:
            return seq_x,seq_y,seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def parse_delta(masks):
    deltas = []
    l,d = masks.shape[0],masks.shape[1]
    for h in range(l):
        if h == 0:
            deltas.append(np.zeros(d))
        else:
            deltas.append(masks[h-1]*np.ones(d) + (1 - masks[h-1]) * (deltas[-1]+np.ones(d)))
    deltas = np.array(deltas)
    # deltas = (deltas - np.min(deltas,axis=0,keepdims=True))/(np.max(deltas,axis=0,keepdims=True)-np.min(deltas,axis=0,keepdims=True))
    return deltas


class Dataset_Custom2(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.mask = mask
        self.args = args
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()
        self.scaler5 = StandardScaler()
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        if "electricity" in self.data_path:
            true_path = "/home/project/S4M/data/electricity/electricity.csv"
        elif "exchange" in self.data_path:
            true_path = "/home/project/S4M/data/exchange_rate/exchange_rate.csv"
        elif "Solar" in self.data_path:
            true_path = "/home/project/S4M/data/Solar/solar_AL.csv"
        elif "traffic" in self.data_path:
            true_path = "/home/project/S4M/data/traffic/traffic.csv"
        elif "weather" in self.data_path:
            true_path = "/home/project/S4M/data/weather/weather.csv"
        elif "simulation" in self.data_path:
            true_path = "/home/project/S4M/data/simulation/simulation.csv"
        elif "ETTh1" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh1.csv'
        elif "ETTh2" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh2.csv'
        elif 'climate' in self.data_path:
            true_path = '/home/project/S4M/data/climate/climate.csv'
        
        df_true = pd.read_csv(true_path)
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
            
        if "simulation" not in self.data_path:
            cols = list(df_raw.columns)
            cols.remove('date')
            if 'climate' not in self.data_path:
                cols.remove(self.target)
                df_raw = df_raw[['date'] + cols +[self.target]]    
            else:
                df_raw = df_raw[['date']+cols] 
            df_mask = df_raw.copy()
            df_mask.iloc[:,1:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)

        else:
            df_mask = df_raw.copy()
            df_mask.iloc[:,:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)        
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if "simulation" in self.data_path:
            df_data= df_raw
            df_true_data = df_true
        else:
            if self.features == 'M' or self.features == 'MS':
                
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
                df_mask = df_mask[cols_data]   
                df_true_data = df_true[cols_data] 
                    
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
                df_mask = df_mask[[self.target]]
                df_true_data = df_true[[self.target]]
        
        df_cumsum = df_data.cumsum(axis=0)
        df_cummean = df_cumsum/df_mask.cumsum(axis=0)
        df_cumstd = ((df_data-df_cummean)).cumsum(axis=0)/df_mask.cumsum(axis=0)
        df_cumstd = df_cumstd**(1/2)
        df_cumstd.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="backfill",inplace=True,axis=0)

        df_f = df_data.fillna(method="ffill",axis=0)
        df_f = df_f.fillna(method="backfill",axis=0)
        
        mask_data = df_mask.values
        delta = parse_delta(mask_data)
    
        if self.scale:
            train1 = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train1.values)
            data = self.scaler1.transform(df_data.values)
            
            train2 = df_cummean[border1s[0]:border2s[0]]
            self.scaler2.fit(train2.values)
            em_mean = self.scaler2.transform(df_cummean.values)
            
            train3 = df_cumstd[border1s[0]:border2s[0]]
            self.scaler3.fit(train3.values)
            em_std = self.scaler3.transform(df_cumstd.values)

            train4 = df_f[border1s[0]:border2s[0]]
            self.scaler4.fit(train4.values)
            data_f = self.scaler4.transform(df_f.values)
            
            train5 = df_data[border1s[0]:border2s[0]]
            self.scaler5.fit(train5.values)
            true_data = self.scaler5.transform(df_true_data.values)
        else:
            data = df_data.values
            em_mean = df_cummean.values
            em_std  = df_cumstd.values
            data_f = df_f.values
            
            
        global_mean = np.nanmean(data[border1s[0]:border2s[0]],axis=0,keepdims=True).repeat(data.shape[0],axis=0)
        data_df = pd.DataFrame(data)
        data = (data_df.fillna(0,axis=0)).values
        
        if "simulation" in self.data_path:
            data_stamp = np.arange(border1,border2)
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        
        
        
        self.mask_data = mask_data[border1:border2]
        self.delta = delta[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = data[border1:border2]
        self.em_mean = em_mean[border1:border2]
        self.data_f = data_f[border1:border2]
        self.em_std = em_std[border1:border2]
        self.global_mean = global_mean[border1:border2]
        
        self.true_data = true_data[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.args.mean_type=="global_mean":
            seq_mean = self.global_mean[s_begin:s_end]
        else:
            seq_mean = self.em_mean[s_begin:s_end]
        # idx = list(range(s_begin,s_end))
        # idx_ls = [list(range(self.config.))]
        
        seq_true_x = self.true_data[s_begin:s_end]
        seq_true_y = self.true_data[r_begin:r_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_std = self.em_std[s_begin:s_end]
        seq_x_f = self.data_f[s_begin:s_end]
        seq_delta = self.delta[s_begin:s_end]
        seq_x_mask = self.mask_data[s_begin:s_end]
        seq_y_mask = self.mask_data[r_begin:r_end]
        
        if self.args.model=='BiaTCGNet':
            seq_x = np.expand_dims(seq_x,axis=-1)
            seq_x_mask = np.expand_dims(seq_x_mask,axis=-1)
        return seq_x,seq_x_f,seq_x_mask,seq_mean,seq_delta,seq_x_mark,seq_y,seq_y_mask,seq_std,seq_true_x,seq_true_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform1(self, data):
        return self.scaler1.inverse_transform(data)

    def inverse_transform5(self, data):
        return self.scaler5.inverse_transform(data)
    


class Dataset_CRU(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.mask = mask
        self.args = args
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()
        self.scaler5 = StandardScaler()
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        if "electricity" in self.data_path:
            true_path = "/home/project/S4M/data/electricity/electricity.csv"
        elif "exchange" in self.data_path:
            true_path = "/home/project/S4M/data/exchange_rate/exchange_rate.csv"
        elif "Solar" in self.data_path:
            true_path = "/home/project/S4M/data/Solar/solar_AL.csv"
        elif "traffic" in self.data_path:
            true_path = "/home/project/S4M/data/traffic/traffic.csv"
        elif "weather" in self.data_path:
            true_path = "/home/project/S4M/data/weather/weather.csv"
        elif "simulation" in self.data_path:
            true_path = "/home/project/S4M/data/simulation/simulation.csv"
        elif "ETTh1" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh1.csv'
        elif "ETTh2" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh2.csv'
        
        df_true = pd.read_csv(true_path)
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
            
        if "simulation" not in self.data_path:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols +[self.target]]    
            df_mask = df_raw.copy()
            df_mask.iloc[:,1:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)

        else:
            df_mask = df_raw.copy()
            df_mask.iloc[:,:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)        
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if "simulation" in self.data_path:
            df_data= df_raw
            df_true_data = df_true
        else:
            if self.features == 'M' or self.features == 'MS':
                
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
                df_mask = df_mask[cols_data]   
                df_true_data = df_true[cols_data] 
                    
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
                df_mask = df_mask[[self.target]]
                df_true_data = df_true[[self.target]]
        
        df_cumsum = df_data.cumsum(axis=0)
        df_cummean = df_cumsum/df_mask.cumsum(axis=0)
        df_cumstd = ((df_data-df_cummean)).cumsum(axis=0)/df_mask.cumsum(axis=0)
        df_cumstd = df_cumstd**(1/2)
        df_cumstd.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="backfill",inplace=True,axis=0)

        df_f = df_data.fillna(method="ffill",axis=0)
        df_f = df_f.fillna(method="backfill",axis=0)
        
        mask_data = df_mask.values
        delta = parse_delta(mask_data)
    
        if self.scale:
            train1 = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train1.values)
            data = self.scaler1.transform(df_data.values)
            
            train2 = df_cummean[border1s[0]:border2s[0]]
            self.scaler2.fit(train2.values)
            em_mean = self.scaler2.transform(df_cummean.values)
            
            train3 = df_cumstd[border1s[0]:border2s[0]]
            self.scaler3.fit(train3.values)
            em_std = self.scaler3.transform(df_cumstd.values)

            train4 = df_f[border1s[0]:border2s[0]]
            self.scaler4.fit(train4.values)
            data_f = self.scaler4.transform(df_f.values)
            
            train5 = df_data[border1s[0]:border2s[0]]
            self.scaler5.fit(train5.values)
            true_data = self.scaler5.transform(df_true_data.values)
        else:
            data = df_data.values
            em_mean = df_cummean.values
            em_std  = df_cumstd.values
            data_f = df_f.values
            
            
        global_mean = np.nanmean(data[border1s[0]:border2s[0]],axis=0,keepdims=True).repeat(data.shape[0],axis=0)
        data_df = pd.DataFrame(data)
        data = (data_df.fillna(0,axis=0)).values
        # print(df_raw['date'][border1:border2])
        date = pd.to_datetime(df_raw['date'][border1:border2])
        timestep = date.apply(lambda x: (x-date.iloc[0]).total_seconds()/3600)
        
        obs_valid = np.sum(mask_data[border1:border2], axis=-1) > 0
        
        
        self.mask_data = mask_data[border1:border2]
        self.data_stamp = timestep.to_numpy()
        self.obs_valid = obs_valid
        self.target_obs = data[border1:border2]
        self.obs = data[border1:border2]
        self.true_data = true_data[border1:border2]
        self.target_mask = mask_data[border1:border2]
        
        
        # self.delta = delta[border1:border2]
        # self.data_y = data[border1:border2]
        # self.data_x = data[border1:border2]
        # self.em_mean = em_mean[border1:border2]
        # self.data_f = data_f[border1:border2]
        # self.em_std = em_std[border1:border2]
        # self.global_mean = global_mean[border1:border2]
        # self.true_data = true_data[border1:border2]
        


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.obs[s_begin:r_end].copy()
        truth = self.target_obs[s_begin:r_end]
        seq_x[self.seq_len:] = 0
        obs_valid = self.obs_valid[s_begin:r_end].copy()
        obs_valid[self.seq_len:] = False
        obs_mask = self.mask_data[s_begin:r_end]==1.0
        target_mask = (self.target_mask[s_begin:r_end]==1.0)*1
        obs_times = self.data_stamp[s_begin:r_end]
        
        # idx_ls = [list(range(self.config.))]
        
        seq_true_x = self.true_data[s_begin:s_end]
        seq_true_y = self.true_data[r_begin:r_end]
        
        return seq_x,truth,obs_valid, obs_times, target_mask, obs_mask,seq_true_x,seq_true_y

    def __len__(self):
        return len(self.obs) - self.seq_len - self.pred_len + 1

    def inverse_transform1(self, data):
        return self.scaler1.inverse_transform(data)

    def inverse_transform5(self, data):
        return self.scaler5.inverse_transform(data)
    
    

class Dataset_Grafiti(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.mask = mask
        self.args = args
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()
        self.scaler5 = StandardScaler()
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        if "electricity" in self.data_path:
            true_path = "/home/project/S4M/data/electricity/electricity.csv"
        elif "exchange" in self.data_path:
            true_path = "/home/project/S4M/data/exchange_rate/exchange_rate.csv"
        elif "Solar" in self.data_path:
            true_path = "/home/project/S4M/data/Solar/solar_AL.csv"
        elif "traffic" in self.data_path:
            true_path = "/home/project/S4M/data/traffic/traffic.csv"
        elif "weather" in self.data_path:
            true_path = "/home/project/S4M/data/weather/weather.csv"
        elif "simulation" in self.data_path:
            true_path = "/home/project/S4M/data/simulation/simulation.csv"
        elif "ETTh1" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh1.csv'
        elif "ETTh2" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh2.csv'
        
        df_true = pd.read_csv(true_path)
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
            
        if "simulation" not in self.data_path:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols +[self.target]]    
            df_mask = df_raw.copy()
            df_mask.iloc[:,1:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)

        else:
            df_mask = df_raw.copy()
            df_mask.iloc[:,:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)        
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if "simulation" in self.data_path:
            df_data= df_raw
            df_true_data = df_true
        else:
            if self.features == 'M' or self.features == 'MS':
                
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
                df_mask = df_mask[cols_data]   
                df_true_data = df_true[cols_data] 
                    
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
                df_mask = df_mask[[self.target]]
                df_true_data = df_true[[self.target]]
        
        df_cumsum = df_data.cumsum(axis=0)
        df_cummean = df_cumsum/df_mask.cumsum(axis=0)
        df_cumstd = ((df_data-df_cummean)).cumsum(axis=0)/df_mask.cumsum(axis=0)
        df_cumstd = df_cumstd**(1/2)
        df_cumstd.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="backfill",inplace=True,axis=0)

        df_f = df_data.fillna(method="ffill",axis=0)
        df_f = df_f.fillna(method="backfill",axis=0)
        
        mask_data = df_mask.values
        delta = parse_delta(mask_data)
    
        if self.scale:
            train1 = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train1.values)
            data = self.scaler1.transform(df_data.values)
            
            train2 = df_cummean[border1s[0]:border2s[0]]
            self.scaler2.fit(train2.values)
            em_mean = self.scaler2.transform(df_cummean.values)
            
            train3 = df_cumstd[border1s[0]:border2s[0]]
            self.scaler3.fit(train3.values)
            em_std = self.scaler3.transform(df_cumstd.values)

            train4 = df_f[border1s[0]:border2s[0]]
            self.scaler4.fit(train4.values)
            data_f = self.scaler4.transform(df_f.values)
            
            train5 = df_data[border1s[0]:border2s[0]]
            self.scaler5.fit(train5.values)
            true_data = self.scaler5.transform(df_true_data.values)
        else:
            data = df_data.values
            em_mean = df_cummean.values
            em_std  = df_cumstd.values
            data_f = df_f.values
            
            
        global_mean = np.nanmean(data[border1s[0]:border2s[0]],axis=0,keepdims=True).repeat(data.shape[0],axis=0)
        data_df = pd.DataFrame(data)
        data = (data_df.fillna(0,axis=0)).values
        # print(df_raw['date'][border1:border2])
        date = pd.to_datetime(df_raw['date'][border1:border2])
        timestep = date.apply(lambda x: (x-date.iloc[0]).total_seconds()/3600)
        
        obs_valid = np.sum(mask_data[border1:border2], axis=-1) > 0
        
        
        self.mask_data = mask_data[border1:border2]
        self.data_stamp = timestep.to_numpy()
        self.obs_valid = obs_valid
        
        self.target_obs = data[border1:border2]
        self.obs = data[border1:border2]
        self.true_data = true_data[border1:border2]
        self.target_mask = mask_data[border1:border2]
        
        
        # self.delta = delta[border1:border2]
        # self.data_y = data[border1:border2]
        # self.data_x = data[border1:border2]
        # self.em_mean = em_mean[border1:border2]
        # self.data_f = data_f[border1:border2]
        # self.em_std = em_std[border1:border2]
        # self.global_mean = global_mean[border1:border2]
        # self.true_data = true_data[border1:border2]
        


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.obs[s_begin:r_end]
        truth = self.target_obs[s_begin:r_end]
        seq_x[self.seq_len:] = 0
        truth[:self.seq_len] = 0
        
        obs_mask = self.mask_data[s_begin:r_end]
        obs_mask1 = obs_mask.copy()
        obs_mask1[self.seq_len:]=0
        obs_mask1 = (obs_mask1!=0.0)
        
        target_mask = self.mask_data[s_begin:r_end]
        target_mask1 = target_mask.copy()
        target_mask1[:self.seq_len]=0
        if self.flag=='test':
            target_mask1[self.seq_len:] = 1.0
        obs_times = self.data_stamp[s_begin:r_end]
        target_mask1 = (target_mask1!=0)
        # idx_ls = [list(range(self.config.))]
        
        seq_true_x = self.true_data[s_begin:s_end]
        seq_true_y = self.true_data[r_begin:r_end]
        return obs_times, seq_x, obs_mask1, obs_times, truth, target_mask1, seq_true_x, seq_true_y

    def __len__(self):
        return len(self.obs) - self.seq_len - self.pred_len + 1

    def inverse_transform1(self, data):
        return self.scaler1.inverse_transform(data)

    def inverse_transform5(self, data):
        return self.scaler5.inverse_transform(data)
    

class Dataset_Brits(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.mask = mask
        self.args = args
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()
        self.scaler5 = StandardScaler()
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        if "electricity" in self.data_path:
            true_path = "/home/project/S4M/data/electricity/electricity.csv"
        elif "exchange" in self.data_path:
            true_path = "/home/project/S4M/data/exchange_rate/exchange_rate.csv"
        elif "Solar" in self.data_path:
            true_path = "/home/project/S4M/data/Solar/solar_AL.csv"
        elif "traffic" in self.data_path:
            true_path = "/home/project/S4M/data/traffic/traffic.csv"
        elif "weather" in self.data_path:
            true_path = "/home/project/S4M/data/weather/weather.csv"
        elif "simulation" in self.data_path:
            true_path = "/home/project/S4M/data/simulation/simulation.csv"
        elif "ETTh1" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh1.csv'
        elif "ETTh2" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh2.csv'
        
        df_true = pd.read_csv(true_path)
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
            
        if "simulation" not in self.data_path:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols +[self.target]]    
            df_mask = df_raw.copy()
            df_mask.iloc[:,1:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)

        else:
            df_mask = df_raw.copy()
            df_mask.iloc[:,:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if "simulation" in self.data_path:
            df_data= df_raw
            df_true_data = df_true
        else:
            if self.features == 'M' or self.features == 'MS':
                
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
                df_mask = df_mask[cols_data]   
                df_true_data = df_true[cols_data] 
                    
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
                df_mask = df_mask[[self.target]]
                df_true_data = df_true[[self.target]]
        
        df_cumsum = df_data.cumsum(axis=0)
        df_cummean = df_cumsum/df_mask.cumsum(axis=0)
        df_cumstd = ((df_data-df_cummean)).cumsum(axis=0)/df_mask.cumsum(axis=0)
        df_cumstd = df_cumstd**(1/2)
        df_cumstd.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="backfill",inplace=True,axis=0)

        df_f = df_data.fillna(method="ffill",axis=0)
        df_f = df_f.fillna(method="backfill",axis=0)
        
        mask_data = df_mask.values
        delta = parse_delta(mask_data)
    
        if self.scale:
            train1 = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train1.values)
            data = self.scaler1.transform(df_data.values)
            
            train2 = df_cummean[border1s[0]:border2s[0]]
            self.scaler2.fit(train2.values)
            em_mean = self.scaler2.transform(df_cummean.values)
            
            train3 = df_cumstd[border1s[0]:border2s[0]]
            self.scaler3.fit(train3.values)
            em_std = self.scaler3.transform(df_cumstd.values)

            train4 = df_f[border1s[0]:border2s[0]]
            self.scaler4.fit(train4.values)
            data_f = self.scaler4.transform(df_f.values)
            
            train5 = df_data[border1s[0]:border2s[0]]
            self.scaler5.fit(train5.values)
            true_data = self.scaler5.transform(df_true_data.values)
        else:
            data = df_data.values
            em_mean = df_cummean.values
            em_std  = df_cumstd.values
            data_f = df_f.values
            
            
        global_mean = np.nanmean(data[border1s[0]:border2s[0]],axis=0,keepdims=True).repeat(data.shape[0],axis=0)
        data_df = pd.DataFrame(data)
        data = (data_df.fillna(0,axis=0)).values
        
        if "simulation" in self.data_path:
            data_stamp = np.arange(border1,border2)
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        
        
        
        self.mask_data = mask_data[border1:border2]
        self.delta = delta[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = data[border1:border2]
        self.em_mean = em_mean[border1:border2]
        self.data_f = data_f[border1:border2]
        self.em_std = em_std[border1:border2]
        self.global_mean = global_mean[border1:border2]
        
        self.true_data = true_data[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        if self.args.mean_type=="global_mean":
            seq_mean = self.global_mean[s_begin:s_end]
        else:
            seq_mean = self.em_mean[s_begin:s_end]
        # idx = list(range(s_begin,s_end))
        # idx_ls = [list(range(self.config.))]
        
        seq_true_x = self.true_data[s_begin:s_end]
        seq_true_y = self.true_data[r_begin:r_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_std = self.em_std[s_begin:s_end]
        seq_x_f = self.data_f[s_begin:s_end]
        seq_delta = self.delta[s_begin:s_end]
        seq_x_mask = self.mask_data[s_begin:s_end]
        seq_y_mask = self.mask_data[r_begin:r_end]
        return seq_x,seq_x_mask,seq_delta,seq_y,seq_y_mask,seq_true_y,seq_true_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform1(self, data):
        return self.scaler1.inverse_transform(data)

    def inverse_transform5(self, data):
        return self.scaler5.inverse_transform(data)



class Dataset_Custom4(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.mask = mask
        self.args = args
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()
        self.scaler5 = StandardScaler()
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        if "electricity" in self.data_path:
            true_path = "/home/project/S4M/data/electricity/electricity.csv"
        elif "exchange" in self.data_path:
            true_path = "/home/project/S4M/data/exchange_rate/exchange_rate.csv"
        elif "Solar" in self.data_path:
            true_path = "/home/project/S4M/data/Solar/solar_AL.csv"
        elif "traffic" in self.data_path:
            true_path = "/home/project/S4M/data/traffic/traffic.csv"
        elif "weather" in self.data_path:
            true_path = "/home/project/S4M/data/weather/weather.csv"
        elif "simulation" in self.data_path:
            true_path = "/home/project/S4M/data/simulation/simulation.csv"
        elif "ETTh1" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh1.csv'
        elif "ETTh2" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh2.csv'
        
        df_true = pd.read_csv(true_path)
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
            
        if "simulation" not in self.data_path:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols +[self.target]]    
            df_mask = df_raw.copy()
            df_mask.iloc[:,1:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)

        else:
            df_mask = df_raw.copy()
            df_mask.iloc[:,:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)        
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if "simulation" in self.data_path:
            df_data= df_raw
            df_true_data = df_true
        else:
            if self.features == 'M' or self.features == 'MS':
                
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
                df_mask = df_mask[cols_data]   
                df_true_data = df_true[cols_data] 
                    
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
                df_mask = df_mask[[self.target]]
                df_true_data = df_true[[self.target]]
        
        df_cumsum = df_data.cumsum(axis=0)
        df_cummean = df_cumsum/df_mask.cumsum(axis=0)
        df_cumstd = ((df_data-df_cummean)).cumsum(axis=0)/df_mask.cumsum(axis=0)
        df_cumstd = df_cumstd**(1/2)
        df_cumstd.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="backfill",inplace=True,axis=0)

        df_f = df_data.fillna(method="ffill",axis=0)
        df_f = df_f.fillna(method="backfill",axis=0)
        
        mask_data = df_mask.values
        delta = parse_delta(mask_data)
    
        if self.scale:
            train1 = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train1.values)
            data = self.scaler1.transform(df_data.values)
            
            train2 = df_cummean[border1s[0]:border2s[0]]
            self.scaler2.fit(train2.values)
            em_mean = self.scaler2.transform(df_cummean.values)
            
            train3 = df_cumstd[border1s[0]:border2s[0]]
            self.scaler3.fit(train3.values)
            em_std = self.scaler3.transform(df_cumstd.values)

            train4 = df_f[border1s[0]:border2s[0]]
            self.scaler4.fit(train4.values)
            data_f = self.scaler4.transform(df_f.values)
            
            train5 = df_data[border1s[0]:border2s[0]]
            self.scaler5.fit(train5.values)
            true_data = self.scaler5.transform(df_true_data.values)
        else:
            data = df_data.values
            em_mean = df_cummean.values
            em_std  = df_cumstd.values
            data_f = df_f.values
            
            
        global_mean = np.nanmean(data[border1s[0]:border2s[0]],axis=0,keepdims=True).repeat(data.shape[0],axis=0)
        data_df = pd.DataFrame(data)
        data = (data_df.fillna(0,axis=0)).values
        
        if "simulation" in self.data_path:
            data_stamp = np.arange(border1,border2)
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        
        
        
        self.mask_data = mask_data[border1:border2]
        self.delta = delta[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = data[border1:border2]
        self.em_mean = em_mean[border1:border2]
        self.data_f = data_f[border1:border2]
        self.em_std = em_std[border1:border2]
        self.global_mean = global_mean[border1:border2]
        
        self.true_data = true_data[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.args.mean_type=="global_mean":
            seq_mean = self.global_mean[s_begin:s_end]
        else:
            seq_mean = self.em_mean[s_begin:s_end]
        # idx = list(range(s_begin,s_end))
        # idx_ls = [list(range(self.config.))]
        max_idx = np.argmax(seq_x,axis=0)
        min_idx = np.argmin(seq_x,axis=0)
        idx = np.expand_dims(np.arange(seq_x.shape[0]),axis=1).repeat(seq_x.shape[1],axis=1)    
        max_idx = np.abs(max_idx-idx)
        min_idx = np.abs(min_idx-idx)    
        
        value = np.max(seq_x,axis=0)
        idx = np.argmax(seq_x,axis=0)
        max_value = np.expand_dims(value,axis=0).repeat(seq_x.shape[0],axis=0)
        value = np.min(seq_x,axis=0)
        min_value = np.expand_dims(value,axis=0).repeat(seq_x.shape[0],axis=0)
                
        # print("max_value",max_value)
        # print("min_value",min_value)
        # print("max_idx",max_idx)
        # print("min_idx",min_idx)
                
        seq_true_x = self.true_data[s_begin:s_end]
        seq_true_y = self.true_data[r_begin:r_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        # seq_std = self.em_std[s_begin:s_end]
        seq_x_f = self.data_f[s_begin:s_end]
        # seq_delta = self.delta[s_begin:s_end]
        seq_x_mask = self.mask_data[s_begin:s_end]
        seq_y_mask = self.mask_data[r_begin:r_end]
        
        if self.args.model=='BiaTCGNet':
            seq_x = np.expand_dims(seq_x,axis=-1)
            seq_x_mask = np.expand_dims(seq_x_mask,axis=-1)
        # print("nan_sim",np.sum(np.isnan(seq_x)),np.sum(np.isnan(max_idx)),np.sum(np.isnan(min_idx)),np.sum(np.isnan(max_value)),np.sum(np.isnan(min_value)))
        return seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform1(self, data):
        return self.scaler1.inverse_transform(data)

    def inverse_transform5(self, data):
        return self.scaler5.inverse_transform(data)


class Dataset_Transformer(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.mask = mask
        self.args = args
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()
        self.scaler5 = StandardScaler()
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        if "electricity" in self.data_path:
            true_path = "/home/project/S4M/data/electricity/electricity.csv"
        elif "exchange" in self.data_path:
            true_path = "/home/project/S4M/data/exchange_rate/exchange_rate.csv"
        elif "Solar" in self.data_path:
            true_path = "/home/project/S4M/data/Solar/solar_AL.csv"
        elif "traffic" in self.data_path:
            true_path = "/home/project/S4M/data/traffic/traffic.csv"
        elif "weather" in self.data_path:
            true_path = "/home/project/S4M/data/weather/weather.csv"
        elif "simulation" in self.data_path:
            true_path = "/home/project/S4M/data/simulation/simulation.csv"
        elif "ETTh1" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh1.csv'
        elif "ETTh2" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh2.csv'
        elif "climate" in self.data_path:
            true_path = '/home/project/S4M/data/climate/climate.csv'
        df_true = pd.read_csv(true_path)
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
            
        if "simulation" not in self.data_path:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols +[self.target]]    
            df_mask = df_raw.copy()
            df_mask.iloc[:,1:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)

        else:
            df_mask = df_raw.copy()
            df_mask.iloc[:,:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)        
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if "simulation" in self.data_path:
            df_data= df_raw
            df_true_data = df_true
        else:
            if self.features == 'M' or self.features == 'MS':
                
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
                df_mask = df_mask[cols_data]   
                df_true_data = df_true[cols_data] 
                    
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
                df_mask = df_mask[[self.target]]
                df_true_data = df_true[[self.target]]
        
        df_cumsum = df_data.cumsum(axis=0)
        df_cummean = df_cumsum/df_mask.cumsum(axis=0)
        df_cumstd = ((df_data-df_cummean)).cumsum(axis=0)/df_mask.cumsum(axis=0)
        df_cumstd = df_cumstd**(1/2)
        df_cumstd.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="ffill",inplace=True,axis=0)
        df_f = df_data.fillna(method="ffill",axis=0)
        
        mask_data = df_mask.values
        delta = parse_delta(mask_data)
    
        if self.scale:
            train1 = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train1.values)
            data = self.scaler1.transform(df_data.values)
            
            train2 = df_cummean[border1s[0]:border2s[0]]
            self.scaler2.fit(train2.values)
            em_mean = self.scaler2.transform(df_cummean.values)
            
            train3 = df_cumstd[border1s[0]:border2s[0]]
            self.scaler3.fit(train3.values)
            em_std = self.scaler3.transform(df_cumstd.values)

            train4 = df_f[border1s[0]:border2s[0]]
            self.scaler4.fit(train4.values)
            data_f = self.scaler4.transform(df_f.values)
            
            train5 = df_data[border1s[0]:border2s[0]]
            self.scaler5.fit(train5.values)
            true_data = self.scaler5.transform(df_true_data.values)
        else:
            data = df_data.values
            em_mean = df_cummean.values
            em_std  = df_cumstd.values
            data_f = df_f.values
            
            
        global_mean = np.nanmean(data[border1s[0]:border2s[0]],axis=0,keepdims=True).repeat(data.shape[0],axis=0)
        data_df = pd.DataFrame(data)
        data = (data_df.fillna(0,axis=0)).values
        
        if "simulation" in self.data_path:
            data_stamp = np.arange(border1,border2)
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        
        
        
        self.mask_data = mask_data[border1:border2]
        self.delta = delta[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = data[border1:border2]
        self.em_mean = em_mean[border1:border2]
        self.data_f = data_f[border1:border2]
        self.em_std = em_std[border1:border2]
        self.global_mean = global_mean[border1:border2]
        
        self.true_data = true_data[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.args.mean_type=="global_mean":
            seq_mean = self.global_mean[s_begin:s_end]
        else:
            seq_mean = self.em_mean[s_begin:s_end]
        # idx = list(range(s_begin,s_end))
        # idx_ls = [list(range(self.config.))]
        
        seq_true_x = self.true_data[s_begin:s_end]
        seq_true_y = self.true_data[r_begin:r_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_std = self.em_std[s_begin:s_end]
        seq_x_f = self.data_f[s_begin:s_end]
        seq_delta = self.delta[s_begin:s_end]
        seq_x_mask = self.mask_data[s_begin:s_end]
        seq_y_mask = self.mask_data[r_begin:r_end]
        return seq_x,seq_x_mask,seq_mean,seq_x_mark,seq_y,seq_y_mask,seq_true_x,seq_true_y,seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform1(self, data):
        return self.scaler1.inverse_transform(data)

    def inverse_transform5(self, data):
        return self.scaler5.inverse_transform(data)



class Dataset_Transformer_Impute(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.mask = mask
        self.args = args
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()
        self.scaler5 = StandardScaler()
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        if "electricity" in self.data_path:
            true_path = "/home/project/S4M/data/electricity/electricity.csv"
            # impute_path = "/home/project/S4M/data/electricity/electricity5_3_imputed.csv" if "chunk_missing" in self.root_path else "/home/project/S4M/data/electricity/electricity5_3_imputed_random.csv"
        elif "exchange" in self.data_path:
            true_path = "/home/project/S4M/data/exchange_rate/exchange_rate.csv"
        elif "Solar" in self.data_path:
            true_path = "/home/project/S4M/data/Solar/solar_AL.csv"
        elif "traffic" in self.data_path:
            true_path = "/home/project/S4M/data/traffic/traffic.csv"
            # impute_path = "/home/project/S4M/data/traffic/traffic5_3_imputed.csv" if "chunk_missing" in self.root_path else "/home/project/S4M/data/traffic/traffic5_3_imputed_random.csv"
        elif "weather" in self.data_path:
            true_path = "/home/project/S4M/data/weather/weather.csv"
            # impute_path = "/home/project/S4M/data/weather/weather5_3_imputed.csv" if "chunk_missing" in self.root_path else "/home/project/S4M/data/weather/weather5_3_imputed_random.csv"
        elif "simulation" in self.data_path:
            true_path = "/home/project/S4M/data/simulation/simulation.csv"
        elif "ETTh1" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh1.csv'
            # impute_path = '/home/project/S4M/data/ETT-small/ETTh1_5_3_imputed.csv' if "chunk_missing" in self.root_path else '/home/project/S4M/data/ETT-small/ETTh1_5_3_imputed_random.csv'
        elif "ETTh2" in self.data_path:
            true_path = '/home/project/S4M/data/ETT-small/ETTh2.csv'
        impute_path = self.args.impute_path
        print(impute_path)
        df_true = pd.read_csv(true_path)
        
        df_impute = pd.read_csv(impute_path)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
            
        if "simulation" not in self.data_path:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols +[self.target]]    
            df_mask = df_raw.copy()
            df_mask.iloc[:,1:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)

        else:
            df_mask = df_raw.copy()
            df_mask.iloc[:,:]=1.0
            df_mask = df_mask.where(~df_raw.isna(),0.0)        
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if "simulation" in self.data_path:
            df_data= df_raw
            df_true_data = df_true
        else:
            if self.features == 'M' or self.features == 'MS':
                
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
                df_mask = df_mask[cols_data]   
                df_true_data = df_true[cols_data] 
                df_impute = df_impute[cols_data]
                    
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
                df_mask = df_mask[[self.target]]
                df_true_data = df_true[[self.target]]
                df_impute = df_impute[[self.target]]
        
        df_cumsum = df_data.cumsum(axis=0)
        df_cummean = df_cumsum/df_mask.cumsum(axis=0)
        df_cumstd = ((df_data-df_cummean)).cumsum(axis=0)/df_mask.cumsum(axis=0)
        df_cumstd = df_cumstd**(1/2)
        df_cumstd.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="ffill",inplace=True,axis=0)
        df_f = df_data.fillna(method="ffill",axis=0)
        
        mask_data = df_mask.values
        delta = parse_delta(mask_data)
    
        if self.scale:
            train1 = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train1.values)
            data = self.scaler1.transform(df_data.values)
            
            train2 = df_cummean[border1s[0]:border2s[0]]
            self.scaler2.fit(train2.values)
            em_mean = self.scaler2.transform(df_cummean.values)
            
            train3 = df_cumstd[border1s[0]:border2s[0]]
            self.scaler3.fit(train3.values)
            em_std = self.scaler3.transform(df_cumstd.values)

            train4 = df_f[border1s[0]:border2s[0]]
            self.scaler4.fit(train4.values)
            data_f = self.scaler4.transform(df_f.values)
            
            train5 = df_data[border1s[0]:border2s[0]]
            self.scaler5.fit(train5.values)
            true_data = self.scaler5.transform(df_true_data.values)
            
            impute_data = self.scaler1.transform(df_impute.values)
        else:
            data = df_data.values
            em_mean = df_cummean.values
            em_std  = df_cumstd.values
            data_f = df_f.values
            impute_data = df_impute.values
            
            
        global_mean = np.nanmean(data[border1s[0]:border2s[0]],axis=0,keepdims=True).repeat(data.shape[0],axis=0)
        data_df = pd.DataFrame(data)
        data = (data_df.fillna(0,axis=0)).values
        
        if "simulation" in self.data_path:
            data_stamp = np.arange(border1,border2)
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        
        
        
        self.mask_data = mask_data[border1:border2]
        self.delta = delta[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = data[border1:border2]
        self.em_mean = em_mean[border1:border2]
        self.data_f = data_f[border1:border2]
        self.em_std = em_std[border1:border2]
        self.global_mean = global_mean[border1:border2]
        self.impute = impute_data[border1:border2]
        
        self.true_data = true_data[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.args.mean_type=="global_mean":
            seq_mean = self.global_mean[s_begin:s_end]
        else:
            seq_mean = self.em_mean[s_begin:s_end]
        # idx = list(range(s_begin,s_end))
        # idx_ls = [list(range(self.config.))]
        seq_impute = self.impute[s_begin:s_end]
        seq_true_x = self.true_data[s_begin:s_end]
        seq_true_y = self.true_data[r_begin:r_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_std = self.em_std[s_begin:s_end]
        seq_x_f = self.data_f[s_begin:s_end]
        seq_delta = self.delta[s_begin:s_end]
        seq_x_mask = self.mask_data[s_begin:s_end]
        seq_y_mask = self.mask_data[r_begin:r_end]
        return seq_x,seq_x_mask,seq_impute,seq_x_mark,seq_y,seq_y_mask,seq_true_x,seq_true_y,seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform1(self, data):
        return self.scaler1.inverse_transform(data)

    def inverse_transform5(self, data):
        return self.scaler5.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):
         # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.args = args
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Solar1(Dataset):
    def __init__(self, root_path, args,flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None):

        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.args = args

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))  
        
        df_mask = df_raw.copy()
        df_mask.iloc[:,:]=1.0
        df_mask = df_mask.where(~df_raw.isna(),0.0)          

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw
        
        df_cumsum = df_data.cumsum(axis=0)
        df_cummean = df_cumsum/df_mask.cumsum(axis=0)
        df_cumstd = ((df_data-df_cummean)).cumsum(axis=0)/df_mask.cumsum(axis=0)
        df_cumstd = df_cumstd**(1/2)
        df_cumstd.fillna(method="ffill",inplace=True,axis=0)
        df_cummean.fillna(method="ffill",inplace=True,axis=0)

        mask_data = df_mask.values
        delta = parse_delta(mask_data)
        
        if self.scale:
            train1 = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train1.values)
            data = self.scaler1.transform(df_data.values)
            
            train2 = df_cummean[border1s[0]:border2s[0]]
            self.scaler2.fit(train2.values)
            em_mean = self.scaler2.transform(df_cummean.values)
            
            train3 = df_cumstd[border1s[0]:border2s[0]]
            self.scaler3.fit(train3.values)
            em_std = self.scaler3.transform(df_cumstd.values)
        else:
            data = df_data.values
            em_mean = df_cummean.values
            em_std  = df_cumstd.values
            
        data_df = pd.DataFrame(data)
        data_f = (data_df.fillna(axis=0,method="ffill")).values
        data = (data_df.fillna(0,axis=0)).values
    
            
        self.mask_data = mask_data[border1:border2]
        self.delta = delta[border1:border2]
        self.data_y = data[border1:border2]
        self.data_x = data[border1:border2]
        self.em_mean = em_mean[border1:border2]
        self.data_f = data_f[border1:border2]
        self.em_std = em_std[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_mean = self.em_mean[s_begin:s_end]
        seq_std = self.em_std[s_begin:s_end]
        seq_x_f = self.data_f[s_begin:s_end]
        seq_delta = self.delta[s_begin:s_end]
        seq_x_mask = self.mask_data[s_begin:s_end]
        seq_y_mask = self.mask_data[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x,seq_x_f,seq_x_mask,seq_mean,seq_delta,seq_x_mark,seq_y_mark,seq_y,seq_y_mask,seq_std

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
