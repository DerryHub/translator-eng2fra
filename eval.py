from rnn import EvalModel, Encoder, Decoder
from transformer import TransformerEF
import torch
from word import Words

w = Words()

'''rnn'''
# encoder = Encoder(w.engNum)
# decoder = Decoder(w.fraNum)

# evalModel = EvalModel(encoder, decoder, w)

# evalModel.load_state_dict(torch.load('model.pkl'))

# evalModel.eval()

# eng = 'We need a plan.'
# fra = evalModel(eng)

# print('ENG: {}'.format(eng))
# print('FRA: {}'.format(fra))

'''transformer'''
transformer = TransformerEF(w, w.engNum, w.fraNum)
transformer.load_state_dict(torch.load('transformer.pkl'))

transformer.eval()

eng = 'I\'m sure you\'ll succeed'
fra = transformer(eng)

print('ENG: {}'.format(eng))
print('FRA: {}'.format(fra))