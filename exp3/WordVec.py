class WordVec():
    EOS_token = 1
    SOS_token = 0

    def __init__(self, name):
        self.name = name
        self.word2Index = {}
        self.word2Count = {}
        self.index2Count = {0: "EOS", 1: "SOS"}
        self.num_words = 2

    def addWord(self, word):
        if (word not in self.word2Count.keys()):
            self.index2Count[self.num_words] = word
            self.word2Index[word] = self.num_words
            self.num_words += 1
            self.word2Count[word] = 1
        else:
            self.word2Count[word] += 1

    def addSentence(self,sentence:str):
        for word in sentence.split(' '):
            self.addWord(word)
