import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

# 下载 NLTK 停用词（如没有下载，可以运行下面的命令）
# nltk.download('stopwords')

# 自定义 SelfAttV4 注意力机制
class SelfAttV4(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.att_drop = nn.Dropout(0.1)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask=None):
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)
        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        ret = self.output_proj(output)
        return ret

# 数据预处理
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

# 创建文本数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text]
        indices = indices[:self.max_len] + [self.vocab['<PAD>']] * (self.max_len - len(indices))
        return torch.tensor(indices), torch.tensor(label)

# 文本分类模型
class TextAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len):
        super(TextAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attn = SelfAttV4(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, attention_mask=None):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_out = self.attn(lstm_out, attention_mask)
        out = self.fc(attn_out[:, -1, :])  # 取 LSTM 输出的最后一个时间步
        return out

# 准备数据
texts = ["I love programming", "PyTorch is amazing", "Natural language processing is fun", 
         "I enjoy machine learning", "Deep learning is powerful", "I hate bugs"]
labels = [1, 1, 0, 1, 0, 0]

processed_texts = [preprocess_text(text) for text in texts]

# 创建词汇表,啥意思？为啥就创建词汇表了
all_words = [word for text in processed_texts for word in text]
vocab = Counter(all_words)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}
vocab['<PAD>'] = 0
vocab['<UNK>'] = len(vocab)

# 数据集划分
train_texts, test_texts, train_labels, test_labels = train_test_split(processed_texts, labels, test_size=0.33, random_state=42)

train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset = TextDataset(test_texts, test_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 实例化模型
vocab_size = len(vocab)
embedding_dim = 50
hidden_dim = 64
output_dim = 2
max_len = 10

model = TextAttentionModel(vocab_size, embedding_dim, hidden_dim, output_dim, max_len)

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 测试模型
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        
        predictions.extend(predicted.numpy())
        true_labels.extend(labels.numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")
