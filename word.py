from tqdm import tqdm
from nltk.tokenize import TreebankWordTokenizer
import random

class Words:
    def __init__(self, file='nlp_task_now/data/eng-fra.txt'):
        tokenizer_twt = TreebankWordTokenizer()
        with open(file, 'r') as f:
            textList = f.readlines()


        self.engCountDic = {}
        self.fraCountDic = {'[eos]':0, '[sos]':0}
        self.engW2IDic = {}
        self.fraW2IDic = {'[eos]':0, '[sos]':1}
        self.engI2WDic = {}
        self.fraI2WDic = {0:'[eos]', 1:'[sos]'}
        self.sents = []

        engIndex = 0
        fraIndex = 2

        # textList = textList[:1000]
        textList_t = tqdm(textList)
        textList_t.set_description_str(" Dealing words")
        for item in textList_t:
            lst = item.strip().split('\t')
            eng = lst[0]
            fra = lst[1]
            engWords = tokenizer_twt.tokenize(eng)
            fraWords = tokenizer_twt.tokenize(fra)
            fraWords.append('[eos]')
            fraWords.insert(0, '[sos]')
            # engWords.append('[eos]')
            self.sents.append([engWords, fraWords])
            for word in engWords:
                word = word.lower()
                if word in self.engCountDic.keys():
                    self.engCountDic[word] += 1
                else:
                    self.engCountDic[word] = 1
                    self.engI2WDic[engIndex] = word
                    self.engW2IDic[word] = engIndex
                    engIndex += 1
            for word in fraWords:
                word = word.lower()
                if word in self.fraCountDic.keys():
                    self.fraCountDic[word] += 1
                else:
                    self.fraCountDic[word] = 1
                    self.fraI2WDic[fraIndex] = word
                    self.fraW2IDic[word] = fraIndex
                    fraIndex += 1
        random.shuffle(self.sents)
        self.engNum = len(self.engW2IDic)
        self.fraNum = len(self.fraW2IDic)

    def getEngCount(self, word):
        word = word.lower()
        try:
            return self.engCountDic[word]
        except:
            raise Exception("{} is not in the dictionary".format(word))
    
    def getFraCount(self, word):
        word = word.lower()
        try:
            return self.fraCountDic[word]
        except:
            raise Exception("{} is not in the dictionary".format(word))

    def getEngIndex(self, word):
        word = word.lower()
        try:
            return self.engW2IDic[word]
        except:
            raise Exception("{} is not in the dictionary".format(word))

    def getFraIndex(self, word):
        word = word.lower()
        try:
            return self.fraW2IDic[word]
        except:
            raise Exception("{} is not in the dictionary".format(word))

    def getEngWord(self, index):
        try:
            return self.engI2WDic[index]
        except:
            raise Exception("{} is out of the range".format(index))

    def getFraWord(self, index):
        # try:
        return self.fraI2WDic[index]
        # except:
        #     raise Exception("{} is out of the range".format(index))

if __name__ == "__main__":
    w = Words('nlp_task_now/data/eng-fra.txt')
    print(w.getEngIndex('go'))

    
