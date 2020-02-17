from torch.utils.data import Dataset
import torch
from word import Words
from nltk.tokenize import TreebankWordTokenizer

class MyDataset(Dataset):
    def __init__(self, w):
        super(MyDataset, self).__init__()
        sents = w.sents
        self.sents_index = []
        for sent in sents:
            eng = sent[0]
            fra = sent[1]
            eng_l = []
            for word in eng:
                eng_l.append(w.getEngIndex(word))
            fra_l = []
            for word in fra:
                fra_l.append(w.getFraIndex(word))
            self.sents_index.append([eng_l, fra_l])

    def __len__(self):
        return len(self.sents_index)

    def __getitem__(self, index):
        lst = self.sents_index[index]
        eng = lst[0]
        fra = lst[1]
        eng = torch.LongTensor(eng)
        fra = torch.LongTensor(fra)
        return {'eng':eng, 'fra':fra}

if __name__ == "__main__":
    w = Words()
    dataset = MyDataset(w)
    for d in dataset:
        print(d['eng'], d['fra'])
