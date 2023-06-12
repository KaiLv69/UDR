import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import Dataset,DataLoader

import torch.distributed as dist



class TMPDataset(Dataset):
    #初始化，定义数据内容和标签
    def __init__(self,l):
        # self.l = 1000
        self.data = torch.ones(size=[l,512])
    #返回数据集大小
    def __len__(self):
        return len(self.data)
    #得到数据内容和标签
    def __getitem__(self, index):
        return self.data[index]

import torch.nn as nn


data = TMPDataset(10000)

data = DataLoader(data,batch_size=16)

model = nn.Linear(in_features=512,out_features=512)
optimizer = optim.Adam(params=model.parameters(),lr=2e-5)




accelerator = Accelerator()
device = accelerator.device

model, optimizer, data = accelerator.prepare(model, optimizer, data)

if accelerator.is_main_process:
    data = tqdm.tqdm(data)

print('local_rank:{} device: {}'.format(dist.get_rank(), device))
# print()
# print(accelerator.local_rank)

for epoch in range(10):
    if accelerator.is_main_process:
        print('epoch {}'.format(epoch))
    for source in data:

        optimizer.zero_grad()

        output = model(source)
        loss = torch.sum(output)
        accelerator.backward(loss)
        optimizer.step()