import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from S4.s4.models.s4.s4 import S4Block as S4
from model.Memory import FeatureMemory
import numpy as np


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                # print(z.shape)
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
    


class Encoder2(nn.Module):
    def __init__(self, config,short_len,n=None):
        super(Encoder2, self).__init__()
        if n is None:
            self.n = config.n
        else:
            self.n = n
        self.config = config
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=config.en_conv_hidden_size,
                               kernel_size=(config.W, config.d_var),
                               stride=1,
                               padding=0)  # Assuming padding='VALID' equivalent to padding=0
        self.dropout = nn.Dropout(p=1-config.input_keep_prob)
        self.short_len =short_len
        self.s4 = S4_Layer(d_model = config.en_conv_hidden_size,dropout =1-config.output_keep_prob)  # B,L,D
        self.attention_layer = AttentionLayer(
                        FullAttention(False, config.factor, attention_dropout=config.dropout,
                                      output_attention=config.output_attention), config.en_conv_hidden_size, config.n_heads)
        
        Tc = self.short_len - self.config.W + 1
        self.Tc = Tc
        # Dropout for RNN layers

        self.linear = nn.Linear(Tc,1)

    def forward(self, x):   #
        b,n,l,d = x.shape
        x = x.reshape(-1,1,self.short_len,self.config.d_var)                # Reshape input_x : <batch_size * n,1, T, D>
        batch_size = int(x.shape[0]/n)
        # print("Tag1",x.shape)
        
        last_rnn_hidden_size = self.config.en_rnn_hidden_sizes[-1]


        # CNN layer
        h_conv1 = F.relu(self.conv1(x))  # (batch_size*n,en_conv_hidden_size,Tc,1)
        # print("tag2",h_conv1.shape)
        # print("h_conv1.shape",h_conv1.shape)

        if self.config.input_keep_prob < 1:
            h_conv1 = self.dropout(h_conv1)


        input = h_conv1.reshape(-1,self.Tc,self.config.en_conv_hidden_size) # (batch_size*n,Tc,en_conv_hidden_size)
        # print("input.shape",input.shape)
        att_output,_ = self.attention_layer(input,input,input,attn_mask=None)
        # print("att_output.shape",att_output.shape)
        att_output = self.s4(att_output)
        
        y = self.linear(att_output.transpose(2,1))
        y = y.transpose(2,1)
        # print("y.shape",y.shape)
        y = y.reshape(batch_size,n,-1)
        return y  #  # <batch_size, n, last_rnn_hidden_size>


class S4_Layer(nn.Module):
    def __init__(self, d_model,dropout):
        super(S4_Layer, self).__init__()
        self.s4_1 = S4(d_model, transposed=False)
        d_ff = d_model
        # self.bidirectional = configs.bidirectional
        self.activation = F.relu 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        # print("***************x.shape",x.shape)
        new_x = self.s4_1(x)[0]               
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


    
class MTNet(nn.Module):
    def __init__(self, config):
        super(MTNet, self).__init__()
        self.config = config


        self.encoder_m = Encoder2(config,short_len=config.short_len)
        self.encoder_u = Encoder2(config,short_len= config.short_len) 
        
        for param in self.encoder_m.parameters():
            param.requires_grad = False
                      
        self.dropout_u = nn.Dropout(p=1-config.input_keep_prob)
    
        last_rnn_hid_size = config.en_conv_hidden_size
        self.pred_w = nn.Parameter(torch.randn(last_rnn_hid_size * 2 + config.d_var,last_rnn_hid_size))
        self.pred_b = nn.Parameter(torch.randn(last_rnn_hid_size))

        self.dropout = nn.Dropout(1 - config.input_keep_prob)
        self.unfold = torch.nn.Unfold( kernel_size = (config.short_len,config.d_var),padding=(config.short_len-1,0),stride =(1,config.d_var))

        self.memory = FeatureMemory(config)

        self.K = config.K
        self.momentum = config.momentum
        self.count = 0
        self.is_training = config.is_training

    @torch.no_grad()
    def update_k_encoder_weights(self):
        """ manually update key encoder weights with momentum and no_grad"""
        # update k_encoder.parameters 0.99
        for p_u, p_m in zip(self.encoder_u.parameters(), self.encoder_m.parameters()):
            p_m.data = p_m.data*self.momentum + (1.0 - self.momentum)*p_u.data
            p_m.requires_grad = False
            
    def warm(self,X):
        with torch.no_grad():
            B,L,D = X.shape
            Q = int(L/self.config.short_len)
            X = X[:,-Q*self.config.short_len:,:].view(B,-1,self.config.short_len,D)
            print("X.shape",X.shape)
            # X = X.reshape(B,-1,self.config.short_len,D)
            m = self.encoder_m(X) 
            self.memory.cluster(m)

    def forward(self,Q):# Q->z_s 
        # Q (B,L,D)    (B*L,1,D)
        B,L,D = Q.shape
        M_Q = Q.detach()
        Q = Q.unsqueeze(0)
        
        unfold_q = self.unfold(Q)[:,:,:L]  # B*L*D -> B*L*L1*D
        
        sliding_q = unfold_q.view(B,self.config.short_len,D,L).permute(0,3,1,2) # B L l D
        u = self.encoder_u(sliding_q)  # B,L,H

        m = self.memory.repre.clone().detach().cuda() # fetch memory
        
        
        if m.shape[0]>self.config.topM:   # random sample topM memory
            ids = torch.randperm(m.size(0))[:self.config.topM]
            m = m[ids,:]
            

        
        H = u.shape[-1]
        u = u.reshape(-1,H)  #n_u, H
        u = nn.functional.normalize(u,dim=1)  # normalize query
        # print("m.shape,u.shape",m.shape,u.shape)
        #*****************************normalize
        i = torch.matmul(m,u.transpose(1,0)) # n_memory n_u* 
        
        if self.config.topK<m.shape[0]:   # get the topK most correlated memory
            values,idxs = torch.topk(i,k=self.config.topK,dim=0)
            selec_m = m[idxs,:]
            p = F.softmax(values, dim=0)

            o = torch.sum(p.unsqueeze(-1)*selec_m,dim=0)  # n_memory n_u 1 * n_memory 1 H -> n_memory n_u H
        else:
            p = F.softmax(i, dim=0)    # n_memory n_us
            o = torch.sum(p.unsqueeze(-1)*m.unsqueeze(1),dim=0)  # n_memory n_u 1 * n_memory 1 H -> n_memory n_u H
        

        o = o.reshape(B,L,H)
        o = torch.cat([o,u.reshape(B,L,H),Q.squeeze(0)],dim=-1) # B,L,2*H+D
        o = torch.matmul(o,self.pred_w)+self.pred_b

        # update memory bank
        with torch.no_grad():
            Q = int(M_Q.shape[1]/self.config.short_len)
            lim = Q*self.config.short_len
            q = M_Q[:,-lim:,:].reshape(B,-1,self.config.short_len,D)
            q = self.encoder_m(q)
            q = torch.nn.functional.normalize(q.reshape(-1,H),dim=1)
            q = q.reshape(B,-1,H)
            if self.is_training:
                self.update_k_encoder_weights()
                self.memory.add_features_from_sample_learned(q)
        
        o = o+u.reshape(B,L,H)
        return o  # (B,L,D)