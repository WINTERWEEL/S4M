"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import numpy as np
import random
from sklearn.cluster import KMeans
from torch_kmeans import KMeans
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Bio.Cluster import kcluster
from Bio.Cluster import clustercentroids
import numpy as np
import os


"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import numpy as np
import random
from sklearn.cluster import KMeans


class FeatureMemory:
    def __init__(self,config):
        self.config = config
        self.memory_size = config.memory_size  # cluster 9000
        self.memory = None  # 3 num_cluster * num_pre * H
        self.K = config.K # 
        self.num_dic = [0 for i in range(self.memory_size)]
        self.M = config.M  
        self.per_mem_size= config.per_mem_size # queue 
        self.thres1 = config.thres1  # sim>thres1 joint i cluster
        self.thres2 = config.thres2  # sim<thre2 add cluster
        self.count = 0
        self.idx_arr = []
        self.repre=None
            
    def init_bank(self):
        self.num_dic = [0 for i in range(self.memory_size)]
        self.memory = None
        self.repre = None
        
    def num_bank(self):
        if self.memory is None:
            return 0
        else:
            return len(self.memory)
    
    def bank_centroid(self):
        re = []
        for i in range(len(self.memory)):
            re.append(torch.mean(self.memory[i],dim=0))
        
        self.repre = torch.stack(re)
        return self.repre # NUM_CLUSTER * H
    
    def cluster(self,x):
        b,l,h = x.shape
        x = x.reshape(-1,h)
        data = x.detach().cpu().numpy()
        coef = []
        C = [3,4,5,6,7,8,9,10,15,20]
        for clusters in C:
            clusterid, error, nfound = kcluster(data, clusters, dist='u', npass=100)
            silhouette_avg = silhouette_score(data, clusterid, metric='euclidean')
            coef.append(silhouette_avg)
        n = C[np.argmax(coef)]
        clusterid, error, nfound = kcluster(data, n, dist='u', npass=100)    
        memorys = []
        for i in set(clusterid):
            memorys.append(torch.from_numpy(data[clusterid==i,:]))
        for key,value in Counter(clusterid).items():
            self.num_dic[key]=value
        self.memory = memorys
        self.bank_centroid()
                               
    def add_features_from_sample_learned(self,q): # 
        q = q.detach().cpu()
        sim = torch.nn.CosineSimilarity(dim = -1, eps = 1e-6)
        self.count+=1
    
        H = q.size(-1)
        q = q.reshape(-1,H)
        cur_bank = self.repre   # current memory centroid
        num_bank =len(self.memory)
        
        ids = torch.randperm(q.size(0))
        M = self.M if ids.shape[0]>self.M else ids.shape[0]  # sample M query from all of the queries to update memory
        for i in range(M):
            idq = ids[i]
            qi = q[idq:idq+1,:]
            sims = sim(qi,cur_bank)
            top1 = torch.topk(sims,1)
            id2 = top1[1].item()  # idx
            max_sim = top1[0].item()  # value
            num = self.num_dic[id2]
            mem = self.memory[id2]
            if max_sim>=self.thres1:
                if num>=self.per_mem_size:
                    self.memory[id2]= torch.cat([mem[-self.per_mem_size+1:],qi],dim=0)
                else:
                    self.memory[id2] = torch.cat([mem,qi],dim=0)
                    self.num_dic[id2]=num+1
                cur_bank[id2,:] = torch.mean(self.memory[id2],dim=0)
            elif max_sim<self.thres2:
                self.memory.append(qi)
                # self.num_dic[num_bank-1] =1
                cur_bank = torch.cat([cur_bank,torch.mean(self.memory[num_bank-1],dim=0,keepdims=True)],dim=0)
                if len(self.memory)>self.memory_size:
                    self.memory = self.memory[1:]
                    cur_bank = cur_bank[1:,:]
                    self.num_dic=self.num_dic[1:]+[1]
                else:
                    self.num_dic[num_bank] = 1
                    num_bank+=1

        self.repre = cur_bank
