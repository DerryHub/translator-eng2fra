import torch
from torch import nn
from dataset import MyDataset
from nltk.tokenize import TreebankWordTokenizer
from word import Words

class Encoder(nn.Module):
    def __init__(self, engNum=14785):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(engNum, 128)
        self.rnn = nn.GRU(128, 256, 4)

    def forward(self, input):
        embedding = self.embedding(input)
        embedding = embedding.view(-1, 1, 128)
        output, hidden = self.rnn(embedding)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, fraNum=29511):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(fraNum, 128)
        self.rnn = nn.GRU(128, 256, 4)
        self.linear = nn.Sequential(
            nn.Linear(256, fraNum)
        )
        self.fraNum = fraNum

    def forward(self, input, hidden):
        embedding = self.embedding(input)
        embedding = embedding.view(-1, 1, 128)
        output, hidden = self.rnn(embedding, hidden)
        output = output.view(-1, 256)
        output = self.linear(output)
        return output, hidden

class TrainModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TrainModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, eng, fra):
        encode_output, encode_hidden = self.encoder(eng)
        decode_output, _ = self.decoder(fra, encode_hidden)
        return decode_output

class EvalModel(nn.Module):
    def __init__(self, encoder, decoder, w):
        super(EvalModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.w = w
        self.tokenizer_twt = TreebankWordTokenizer()

    def forward(self, eng):
        words = self.tokenizer_twt.tokenize(eng)
        # words.append('[eos]')
        words_input = []
        for word in words:
            index = self.w.getEngIndex(word)
            words_input.append(index)
        words_input = torch.LongTensor(words_input)
        encode_output, encode_hidden = self.encoder(words_input)
        s_list = ['[sos]']
        index = self.w.getFraIndex(s_list[0])
        index_list = [index]
        i = 0
        while s_list[-1] != '[eos]' and i < 100:
            i += 1
            index_tensor = torch.LongTensor(index_list)
            output, encode_hidden = self.decoder(index_tensor, encode_hidden)
            output = output[-1]
            index = torch.argmax(output)
            index_list.append(index)
            word = self.w.getFraWord(int(index))
            s_list.append(word)
        return ' '.join(s_list)


if __name__ == "__main__":
    w = Words('nlp_task_now/data/eng-fra.txt')
    evalModel = EvalModel(Encoder(w.engNum), Decoder(w.fraNum), w)
    output = evalModel('go')
    print(output)