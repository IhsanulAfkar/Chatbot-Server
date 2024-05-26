import numpy as np
from nlp_id.tokenizer import Tokenizer 
from nlp_id.lemmatizer import Lemmatizer 
from nlp_id.stopword import StopWord 

class Helper:
    def __init__(self) -> None:
        self.lemmatizer = Lemmatizer()
        self.tokenizer = Tokenizer()
        self.stopword = StopWord()
        self.punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    def tokenize(self, sentence): 
        return self.tokenizer.tokenize(sentence)

    def stem(self, word): 
        return self.lemmatizer.lemmatize(word.lower())

    def bag_of_words(self, tokenized_sentence, words):
        # stem each word
        sentence_words = [self.stem(word) for word in tokenized_sentence]
        # initialize bag with 0 for each word
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words: 
                bag[idx] = 1

        return bag

    def stopword_removal(self,sentence:str) -> str:
        return self.stopword.remove_stopword(sentence)
    
    def slang_cleaning(self, sentence, slang_df):
        words = self.tokenize(sentence)
        for i, word in enumerate(words):
            if word in slang_df['slang'].values:
                words[i] = slang_df.loc[slang_df['slang'] == word,'formal'].iloc[0]

        return " ".join(words)
    def remove_punctuations(self, sentence:str) -> str:
        return ''.join([char for char in sentence if char not in self.punctuation])

        