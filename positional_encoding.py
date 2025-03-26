import torch
import torch.nn as nn
class PositionalEncoding:
    def __init__(self,hidden_dim,drop_out_num,max_len=1000):
        
        self.drop_out = nn.Dropout(drop_out_num)
        
        
        self.P = torch.zeros((1,max_len,hidden_dim))
        
        X = torch.arange(max_len,dtype=torch.float32)