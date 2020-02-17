import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from dataset import MyDataset
from word import Words
from transformer import TransformerEF

EPOCH = 100

w = Words()

dataset = MyDataset(w)


trainModel = TransformerEF(w, w.engNum, w.fraNum).cuda()

opt = optim.Adam(trainModel.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
cost = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    dataset_t = tqdm(dataset)
    dataset_t.set_description_str(' Training {}'.format(epoch+1))
    loss_all = 0
    for d in dataset_t:
        eng = d['eng'].cuda()
        fra = d['fra'].cuda()
        output = trainModel(eng, fra)
        output = output[:-1]
        fra = fra[1:]
        loss = cost(output, fra)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_all += loss.data
        
    # scheduler.step()

    print('loss is {}'.format(loss_all))
    print('Saving model...')
    torch.save(trainModel.state_dict(), 'transformer.pkl')

