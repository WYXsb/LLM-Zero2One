
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request

# 创建一个字典用于存储config
MASTER_CONFIG = {
    # 参数放这里
}
def get_vocab(path):
    # 读数据
    lines = open(path, 'r').read()
    # 创建简易版词表（字符级）
    vocab = sorted(list(set(lines)))
    print('----创建词表----')
    print('词表前{}个:'.format(head_num), vocab[:head_num])
    print('词表大小:', len(vocab))
    return vocab
# 编码器（青春版）
def encode(stoi,s):
    return [stoi[ch] for ch in s]

# 解码器（青春版）
def decode(itos,l):
    return ''.join([itos[i] for i in l])

# 构建batch
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    # 切分训练集，验证集，测试集，比例为，训练80%，验证10%，测试10%
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    # 将全部的训练数据作为batch，验证集，测试集也换个变量存储（单纯为了方便看）
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # 这里需要学习torch.randint，生成大小为batch_size，内部数值为随机整数的tensor。生成随机数数值域为[0,训练集字符数量-滑动窗口大小-1]之间的整数
    # 详情可以参考官方文档，或者这个博客：https://blog.csdn.net/qq_41813454/article/details/136326473
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    print('ix输出:')
    print(ix)


    # 这里需要学习torch.stack，执行操作类似于python的zip关键字，只不过操作对象是tensor张量，指定任意维度的张量进行组合
    # 详情参考官方文档，或者这个博客：https://blog.csdn.net/dongjinkun/article/details/132590205

    # 这里x作为特征，y作为预测值，因为文本生成任务是根据前n个字符，去推理后面的1个字符，因此y的构造会使窗口在保持原大小的基础上向后移一位
    # 通过滑动窗口，对batch_data中的训练数据，进行随机取样，相当于随机选择训练数据。
    # 在原65万多个字符中，随机选取一个字符作为开始，并以这个开始点，向后选取滑动窗口个数的字符，作为训练数据，向后移一位就是其目标值。  因此ix的构造不能超出index。
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    # 返回特征值，目标值
    return x, y

# 在进行分析LlaMa架构分析之前，我们从最简单的文本生成模型开始创建，然后在最简单的文本生成模型的基础上，把LlaMa的RSM，Rope等一点点添加进去。为此我们先：
# 创建一个有毛病的模型架构
# 分析一下这个架构（其实也没什么分析的）
class SimpleBrokenModel(nn.Module):
    # init里的跟上面一样，没变化
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )



      # 添加前向传播函数
    def forward(self, idx, targets=None):
        # 实例化embedding层，输入映射为id的数据，输出嵌入后的数据
        x = self.embedding(idx)

        logits = self.linear(x)
        

        # 如果有目标值（也就是我们前面的y），则计算通过交叉熵损失计算loss结果。给输出的概率矩阵变个形状，再给目标值变个形状。  统一一下输入输出，然后计算loss。其中最后一维代表着一条数据。
        # 此处需要了解tensor.view()函数，带上几何空间想象力去想一下矩阵的形状。
        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        # 如果没有目标值，则只返回概率分布的结果
        else:
            return logits

        # 查看参数量
        print("模型参数量：", sum([m.numel() for m in self.parameters()]))

@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    # 评估结果存储变量
    out = {}

    # 将模型置为评估模式
    model.eval()

    # 分别会在训练集和验证集里通过get_batchs()函数取评估数据
    for split in ["train", "val"]:

        losses = []

        # 评估10个batch
        for _ in range(10):
            # 拿到特征值（输入数据），以及目标值（输出数据）
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])

            # 把拿到的数据丢进模型，得到loss值
            _, loss = model(xb, yb)

            # 更新loss存储
            losses.append(loss.item())

        # 这里就是大家经常在控制台看到的 "train_loss"  "valid_loss"由来
        out[split] = np.mean(losses)

    # 评估完了，别忘了把模型再置回训练状态，下一个epoch还要继续训练呢
    model.train()

    return out
# 构建训练函数
def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    # loss存储
    losses = []

    # 训练时间记录开始时间
    start_time = time.time()

    # 循环训练指定epoch的轮数
    for epoch in range(config['epochs']):
        # 优化器要初始化啊，否则每次训练都是基于上一次训练结果进行优化，效果甚微
        optimizer.zero_grad()

        # 获取训练数据
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])

        # 前向传播计算概率矩阵与loss
        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        optimizer.step()

        # 如果提供学习率调度器，那么学习率会通过调度器进行修改，比如学习率周期性变化，或者梯度减小，增加，具体策略需要综合考虑进行设置，详情自行查询，关键字：lr_scheduler
        if scheduler:
            scheduler.step()

        # 打印log
        if epoch % config['log_interval'] == 0:
            # 训练时间
            batch_time = time.time() - start_time

            # 执行评估函数，在训练集和验证集上计算loss
            x = evaluate_loss(model)

            # Store the validation loss
            losses += [x]

            # 打印进度日志
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")

            # 重置开始时间，用于计算下一轮的训练时间
            start_time = time.time()

            # 打印下一轮的学习率，如果使用了lr_scheduler
            if scheduler:
                print("lr: ", scheduler.get_lr())

    # 上面所有epoch训练结束，打印最终的结果
    print("Validation loss: ", losses[-1]['val'])

    # 返还每一步loss值的列表，因为我们要画图，返还的是loss迭代的图像
    return pd.DataFrame(losses).plot()


if __name__ == "__main__":
    
    # 查看词表前n个字符
    head_num=50
    data_path = 'xiyouji.txt'
    content = open(data_path, 'r').read()
    vocab = get_vocab(data_path)
    
    # 将词表编码成为数字，普通的整数
    itos = {i: ch for i, ch in enumerate(vocab)}

    # 双向映射
    stoi = {ch: i for i, ch in enumerate(vocab)}
    
    
    print(encode(stoi,"悟空空"))
    print(decode(itos,[1317,2691]))
    # # 对全文进行编码，并映射成为tensor
    dataset = torch.tensor(encode(stoi,content), dtype=torch.int16)

    # # 看一下形状，实际上就是多少个字符，一共65万个字符
    print(dataset.shape)
    print(dataset)
    MASTER_CONFIG.update({
        'batch_size': 8,          # 不解释
        'context_window': 16,      # 滑动窗口采样，设置采样大小
        'vocab_size':4325         # 咱们的西游记数据集，一共包含4325个不重复的汉字，标点符号
    })

    xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
    print(xs, ys)
    # # 因为是随机生成的采样，我们可以看一下数据，其中每个采样数据，来自于原文随机的起始点，每个元组为一个（x,y），可以观察每个x和y的首位去直观感受一下滑动窗口执行的操作
    decoded_samples = [(decode(itos,xs[i].tolist()), decode(itos,ys[i].tolist())) for i in range(len(xs))]
    
    # MASTER_CONFIG.update({
    #     'd_model': 128,
    # })
    # MASTER_CONFIG.update({
    #     'epochs': 1000,
    #     'log_interval': 10,      # 每10个batch打印一次log
    #     'batch_size': 32,
    # })
    # model = SimpleBrokenModel(MASTER_CONFIG)
    # print("咱们的模型这么多参数量:", sum([m.numel() for m in model.parameters()]))
    # optimizer = torch.optim.Adam(
    #     model.parameters(),      # 优化器执行优化全部的模型参数
    # )
    

    # # 启动训练
    # train(model, optimizer,print_logs=True)