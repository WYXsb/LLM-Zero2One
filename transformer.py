import math
import torch

import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


class SelfAttv1(nn.Module):

    def __init__(self, hidden_dim):
        super(SelfAttv1, self).__init__()


        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, X):

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        
        attention_value = torch.matmul(Q, K.transpose(-1, -2))
        
        attention_weight = torch.softmax(attention_value/math.sqrt(self.hidden_dim), dim=-1)
        
        print(attention_weight)
        output = torch.matmul(attention_weight, V)
        return output

word_embeddings = {
    "I":    torch.tensor([1.0, 0.0, 0.0, 0.0]),  # v1
    "love": torch.tensor([0.0, 1.0, 0.0, 0.0]),  # v2
    "you":  torch.tensor([0.0, 0.0, 1.0, 0.0])   # v3
}

# "I love you"
sentence1 = torch.stack([word_embeddings["I"], word_embeddings["love"], word_embeddings["you"]])  # shape: [3, 4]

# "you love I"
sentence2 = torch.stack([word_embeddings["you"], word_embeddings["love"], word_embeddings["I"]])  # shape: [3, 4]



simple_attention = SelfAttv1(4)


out1 = simple_attention(sentence1)
out2 = simple_attention(sentence2)

print("输出1 (I love you):")
print(out1)

print("\n输出2 (you love I):")
print(out2)
