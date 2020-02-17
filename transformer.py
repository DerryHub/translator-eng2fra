import torch
from torch import nn
import math
from nltk.tokenize import TreebankWordTokenizer
from word import Words

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEF(nn.Module):
    def __init__(self, w, engNum=14785, fraNum=29511, dim=128, dropout=0):
        super(TransformerEF, self).__init__()
        self.w = w
        self.dim = dim
        self.tokenizer_twt = TreebankWordTokenizer()
        self.position = PositionalEncoding(dim, dropout=dropout)
        self.engEmbedding = nn.Embedding(engNum, dim)
        self.fraEmbedding = nn.Embedding(fraNum, dim)
        self.transformer = nn.Transformer(d_model=dim, dropout=dropout)
        self.classification = nn.Sequential(
            nn.Linear(dim, fraNum)
        )

    def forward(self, eng, fra=None):
        if fra is not None and self.training == True:
            engEmb = self.engEmbedding(eng)
            fraEmb = self.fraEmbedding(fra)
            engEmb = engEmb.view(-1, 1, self.dim)
            fraEmb = fraEmb.view(-1, 1, self.dim)
            engEmb = self.position(engEmb)
            fraEmb = self.position(fraEmb)
            tgt_mask = self.transformer.generate_square_subsequent_mask(len(fra)).cuda()
            output = self.transformer(engEmb, fraEmb, tgt_mask=tgt_mask)
            output = output.view(-1, self.dim)
            output = self.classification(output)
            return output

        elif fra is None and self.training == False:
            words = self.tokenizer_twt.tokenize(eng)
            words_input = []
            for word in words:
                index = self.w.getEngIndex(word)
                words_input.append(index)
            words_input = torch.LongTensor(words_input)
            engEmb = self.engEmbedding(words_input)
            engEmb = engEmb.view(-1, 1, self.dim)
            engEmb = self.position(engEmb)
            memory = self.transformer.encoder(engEmb, mask=None, src_key_padding_mask=None)
            s_list = ['[sos]']
            index = self.w.getFraIndex(s_list[0])
            index_list = [index]
            i = 0
            while s_list[-1] != '[eos]' and i < 100:
                i += 1
                index_tensor = torch.LongTensor(index_list)
                fraEmb = self.fraEmbedding(index_tensor)
                fraEmb = fraEmb.view(-1, 1, self.dim)
                fraEmb = self.position(fraEmb)
                output = self.transformer.decoder(fraEmb, memory, tgt_mask=None, memory_mask=None,
                              tgt_key_padding_mask=None,
                              memory_key_padding_mask=None)
                output = output[-1]
                output = self.classification(output)
                output = output[0]
                index = torch.argmax(output)
                index_list.append(index)
                word = self.w.getFraWord(int(index))
                s_list.append(word)
            return ' '.join(s_list)

        else:
            raise RuntimeError('error')



if __name__ == "__main__":
    w = Words()
    model = TransformerEF(w, w.engNum, w.fraNum)
    # model.eval()
    a = torch.tensor([4,3,12,5])
    b = torch.tensor([21,4,5,4,6,7,6,8,8,9])
    output = model(a, b)
    print(output.size())