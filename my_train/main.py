from SpamDataset import SpamDataset
import tiktoken
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def get_balanced_dataset(data):
    # 随机抽取出与spam数量相同的ham数据
    num_spam = data[data['Label'] == 'spam'].shape[0]
    ham_subset = data[data['Label'] == 'ham'].sample(num_spam,random_state=42)
    
    # 合并数据
    balanced_data = pd.concat([ham_subset, data[data['Label'] == 'spam']], ignore_index=True)
    return balanced_data

def random_split(data,tran_frac,validation_frac):
    data = data.sample(frac=1,random_state=42).reset_index(drop=True)
    train_end = int(len(data)*tran_frac)
    validation_end = int(len(data)*(tran_frac+validation_frac))
    
    train_data = data[:train_end]
    validation_data = data[train_end:validation_end]
    test_data = data[validation_end:]
    return train_data,validation_data,test_data


if __name__ == '__main__':
    data = pd.read_csv("sms_spam_collection/sms_spam_collection.tsv",sep='\t',header=None,names=['Label','Message'])
    
    # 查看数据大体样貌
    print(data.head())
    print(data["Label"].value_counts())
    balanced_dataset = get_balanced_dataset(data)
    
    
    # 把spam和ham标签对应为0和1
    balanced_dataset["Label"] = balanced_dataset["Label"].map({"ham": 0, "spam": 1})
    print(balanced_dataset["Label"].value_counts())
    
    # 划分数据集
    train_data,validation_data,test_data = random_split(balanced_dataset,0.7,0.1)
    train_data.to_csv("sms_spam_collection/train.csv",index=False)
    validation_data.to_csv("sms_spam_collection/validation.csv",index=False)
    test_data.to_csv("sms_spam_collection/test.csv",index=False)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    
    train_dataset = SpamDataset(
        csv_file="sms_spam_collection/train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file="sms_spam_collection/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="sms_spam_collection/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    
    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    
    print("Train loader:")
    for input_batch, target_batch in train_loader:
        pass

    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)
    
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])