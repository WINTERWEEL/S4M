import torch
import torch.nn as nn
import torch.nn.functional as F
from S4.s4.models.s4.s4 import S4Block as S4
from S4.s4.models.s4.s4_mask1 import S4Block as S4_mask
from model.Bank import MTNet
import os
import math
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = torch.nn.parameter.Parameter(torch.Tensor(output_size, input_size))
        self.b = torch.nn.parameter.Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * torch.autograd.Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
    


class Model(nn.Module):
    """
    Mamba
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.activation = F.relu 
        self.dropout = nn.Dropout(configs.dropout)
        self.mask = configs.mask
        self.decay1 = nn.Linear(configs.d_var,configs.d_var)
        self.decay2 = nn.Linear(configs.d_var,configs.d_var)
        self.classification = configs.classification
        self.plot = configs.plot
        self.d_model = configs.d_model


        self.mem_net = MTNet(configs)

        
        self.linear1 = nn.Linear(configs.en_conv_hidden_size,self.d_model)



        self.norm = nn.LayerNorm(configs.d_model)
    
        self.s4_layers = nn.ModuleList(
        [
            S4_Layer(
                S4_mask(configs.en_conv_hidden_size, transposed=False),
                S4_mask(configs.en_conv_hidden_size, transposed=False),
                configs.en_conv_hidden_size,
                configs
            )
        ]
        +
        [
            S4_Layer(
                S4(configs.d_model, transposed=False),
                S4(configs.d_model, transposed=False),
                configs.d_model,
                configs
            )
            for i in range(configs.e_layers-1)
        ]
        )
        self.mask_trans = nn.Linear(configs.d_var,configs.en_conv_hidden_size)
        if not self.classification:
            self.out_proj=nn.Linear(configs.d_model, configs.d_var, bias=True)
        else:
            self.out_proj=nn.Linear(configs.d_model,configs.num_class)
        self.hist_pool = []
        self.q_pool = []
        
    def embedding(self,x):
        return self.mem_net.encoder_m(x)
        
    def warmup(self,seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value):

        self.mem_net.warm(seq_x)
        return None

    def forward(self,seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value):

        max_idx = (max_idx-torch.mean(max_idx,dim=1,keepdims=True))/torch.std(max_idx,dim=1,keepdims=True)
        min_idx = (max_idx-torch.mean(min_idx,dim=1,keepdims=True))/torch.std(min_idx,dim=1,keepdims=True)
        seq_mask = seq_x_mask[:,:,0:1]
        gamma2 = torch.exp(-1*(self.decay1(max_idx))) 
        gamma1 = torch.exp(-1*(self.decay2(min_idx))) 
        gamma = gamma1+gamma2
        gamma1 = gamma1/gamma
        gamma2 = gamma2/gamma
        z_s = (gamma1*max_value+gamma2*min_value)*(1-seq_x_mask)+seq_x*seq_x_mask# decay imputation
        
        
        # memory bank
        c_s = self.mem_net(z_s)
        mask = self.mask_trans(seq_x_mask)
        # print("c_s nan sum",torch.sum(torch.isnan(c_s)))
        x = self.s4_layers[0](c_s,mask)
        x = self.linear1(x)
        for i in range(1,self.configs.e_layers):
            # print("***********",x.shape)
            x = self.s4_layers[i](x)
        x = self.norm(x)
        if self.classification:
            out = x.mean(dim=1)
            out = self.out_proj(out)
            out = F.softmax(out, dim=1)
        else:
            out = self.out_proj(x)
        # print("out nan sum",torch.sum(torch.isnan(out)))

        return out

class S4_Layer(nn.Module):
    def __init__(self, s4_1,s4_2,d_model,configs):
        super(S4_Layer, self).__init__()
        self.s4_1 = s4_1
        self.s4_2 = s4_2
        d_ff = configs.d_ff
        # self.bidirectional = configs.bidirectional
        self.activation = F.relu 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, x,m=None):
        if m is None:
            new_x = self.s4_1(x)[0]         
        else:
            new_x = self.s4_1(x,m)[0]        
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)